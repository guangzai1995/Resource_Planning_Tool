import asyncio
import json

import pytest
from aiohttp import web

from vllm_bench.pd_proxy import (
    Endpoint,
    PdProxy,
    build_nixl_prefill_body,
    build_p2p_prefill_body,
    build_p2p_request_id,
    inject_kv_transfer_params,
    parse_endpoint,
)


def test_parse_args_accepts_builtin_proxy_command():
    from vllm_bench.pd_proxy import parse_args

    args = parse_args([
        "--connector",
        "p2p_nccl",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--prefill",
        json.dumps({
            "name": "p1",
            "url": "http://p1:30000",
            "kv_address": "p1:21001",
        }),
        "--decode",
        json.dumps({
            "name": "d1",
            "url": "http://d1:31000",
            "kv_address": "d1:22001",
        }),
    ])

    assert args.connector == "p2p_nccl"
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.prefill[0].name == "p1"


def test_parse_endpoint_accepts_json_with_kv_address():
    endpoint = parse_endpoint(json.dumps({
        "name": "p1",
        "url": "http://10.0.0.11:30000",
        "kv_address": "10.0.0.11:21001",
    }))

    assert endpoint == Endpoint(
        name="p1",
        url="http://10.0.0.11:30000",
        kv_address="10.0.0.11:21001",
    )


def test_build_p2p_request_id_matches_vllm_parser_format():
    request_id = build_p2p_request_id(
        "10.0.0.11:21001",
        "10.0.0.21:22001",
        request_uuid="abc",
    )

    assert request_id == (
        "___prefill_addr_10.0.0.11:21001"
        "___decode_addr_10.0.0.21:22001_abc"
    )


def test_build_p2p_prefill_body_forces_one_token_non_streaming():
    body = {"model": "m", "prompt": "hello", "max_tokens": 32, "stream": True}
    prefill_body = build_p2p_prefill_body(body, request_id="rid")

    assert prefill_body["max_tokens"] == 1
    assert prefill_body["stream"] is False
    assert prefill_body["request_id"] == "rid"
    assert body["max_tokens"] == 32


@pytest.mark.parametrize("builder", [
    build_p2p_prefill_body,
    build_nixl_prefill_body,
])
def test_prefill_body_strips_stream_options(builder):
    # prefill 强制 stream=False，必须移除 stream_options，否则 vLLM 会以
    # "Stream options can only be defined when `stream=True`" 拒绝（400）。
    body = {
        "model": "m",
        "messages": [],
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    prefill_body = (
        builder(body, request_id="rid")
        if builder is build_p2p_prefill_body
        else builder(body)
    )

    assert prefill_body["stream"] is False
    assert "stream_options" not in prefill_body
    # 原始请求体不应被修改
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("builder", [
    build_p2p_prefill_body,
    build_nixl_prefill_body,
])
def test_prefill_body_clamps_max_completion_tokens(builder):
    # chat completions 用 max_completion_tokens，vLLM 里它优先于 max_tokens。
    # prefill 只应产出 1 个 token，否则 producer 会继续 decode 进而触发
    # p2p_nccl_connector 的 `assert req_id in self.chunked_prefill` 崩溃。
    body = {
        "model": "m",
        "messages": [],
        "max_completion_tokens": 1024,
        "stream": True,
    }
    prefill_body = (
        builder(body, request_id="rid")
        if builder is build_p2p_prefill_body
        else builder(body)
    )

    assert prefill_body["max_tokens"] == 1
    assert prefill_body["max_completion_tokens"] == 1
    assert body["max_completion_tokens"] == 1024


def test_build_nixl_prefill_body_injects_remote_decode_marker():
    body = {"model": "m", "messages": [], "max_tokens": 32, "stream": True}
    prefill_body = build_nixl_prefill_body(body)

    assert prefill_body["max_tokens"] == 1
    assert prefill_body["stream"] is False
    assert prefill_body["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


def test_inject_kv_transfer_params_preserves_original_body():
    body = {"model": "m", "prompt": "hello"}
    params = {"remote_engine_id": "engine-a"}
    decode_body = inject_kv_transfer_params(body, params)

    assert decode_body == {
        "model": "m",
        "prompt": "hello",
        "kv_transfer_params": {"remote_engine_id": "engine-a"},
    }
    assert "kv_transfer_params" not in body


class FakeResponse:
    def __init__(self, payload, *, status=200, headers=None):
        self._payload = payload
        self.status = status
        self.headers = headers or {"content-type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode("utf-8")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def test_p2p_proxy_sends_prefill_then_decode():
    async def run_case():
        session = FakeSession([
            FakeResponse({"choices": []}),
            FakeResponse({"choices": [{"text": "ok"}]}),
        ])
        proxy = PdProxy(
            connector="p2p_nccl",
            prefill=[Endpoint("p1", "http://p1:30000", "p1:21001")],
            decode=[Endpoint("d1", "http://d1:31000", "d1:22001")],
            session=session,
        )

        status, headers, payload = await proxy.handle_json_completion(
            "/v1/completions",
            {"model": "m", "prompt": "hi", "max_tokens": 8},
        )
        return session, status, headers, payload

    session, status, headers, payload = asyncio.run(run_case())

    assert status == 200
    assert headers["content-type"] == "application/json"
    assert payload == b'{"choices": [{"text": "ok"}]}'
    assert session.calls[0][1] == "http://p1:30000/v1/completions"
    assert session.calls[1][1] == "http://d1:31000/v1/completions"
    prefill_json = session.calls[0][2]["json"]
    decode_json = session.calls[1][2]["json"]
    prefill_headers = session.calls[0][2]["headers"]
    decode_headers = session.calls[1][2]["headers"]
    assert prefill_json["max_tokens"] == 1
    assert decode_json["request_id"] == prefill_json["request_id"]
    assert prefill_headers["X-Request-Id"] == prefill_json["request_id"]
    assert decode_headers["X-Request-Id"] == prefill_json["request_id"]


def test_p2p_proxy_stops_when_prefill_fails():
    async def run_case():
        session = FakeSession([
            FakeResponse({"error": "bad prefill"}, status=400),
            FakeResponse({"choices": [{"text": "should not decode"}]}),
        ])
        proxy = PdProxy(
            connector="p2p_nccl",
            prefill=[Endpoint("p1", "http://p1:30000", "p1:21001")],
            decode=[Endpoint("d1", "http://d1:31000", "d1:22001")],
            session=session,
        )

        with pytest.raises(web.HTTPBadGateway) as exc_info:
            await proxy.handle_json_completion(
                "/v1/completions",
                {"model": "m", "prompt": "hi", "max_tokens": 8},
            )
        return session, exc_info.value

    session, exc = asyncio.run(run_case())

    assert len(session.calls) == 1
    assert session.calls[0][1] == "http://p1:30000/v1/completions"
    assert exc.reason == "prefill p1 failed (400)"
    assert exc.text == 'prefill p1 failed (400): {"error": "bad prefill"}'


def test_nixl_proxy_forwards_prefill_transfer_params_to_decode():
    async def run_case():
        params = {"remote_engine_id": "engine-a", "remote_block_ids": [1]}
        session = FakeSession([
            FakeResponse({"kv_transfer_params": params}),
            FakeResponse({"choices": [{"message": {"content": "ok"}}]}),
        ])
        proxy = PdProxy(
            connector="nixl",
            prefill=[Endpoint("p1", "http://p1:30000")],
            decode=[Endpoint("d1", "http://d1:31000")],
            session=session,
        )

        status, headers, payload = await proxy.handle_json_completion(
            "/v1/chat/completions",
            {"model": "m", "messages": [], "max_tokens": 8},
        )
        return params, session, status, headers, payload

    params, session, status, headers, payload = asyncio.run(run_case())

    assert status == 200
    assert headers["content-type"] == "application/json"
    assert payload == b'{"choices": [{"message": {"content": "ok"}}]}'
    assert session.calls[0][1] == "http://p1:30000/v1/chat/completions"
    assert session.calls[1][1] == "http://d1:31000/v1/chat/completions"
    assert session.calls[1][2]["json"]["kv_transfer_params"] == params
