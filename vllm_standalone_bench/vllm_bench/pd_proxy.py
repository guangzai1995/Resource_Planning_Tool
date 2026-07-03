from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from aiohttp import ClientSession, web


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

MAX_PREFILL_ERROR_CHARS = 2048


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str
    kv_address: str | None = None


def parse_endpoint(value: str) -> Endpoint:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "endpoint must be a JSON object"
        ) from exc
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("endpoint must be a JSON object")
    name = raw.get("name")
    url = raw.get("url")
    kv_address = raw.get("kv_address")
    if not isinstance(name, str) or not name:
        raise argparse.ArgumentTypeError("endpoint.name must be a non-empty string")
    if not isinstance(url, str) or not url:
        raise argparse.ArgumentTypeError("endpoint.url must be a non-empty string")
    if kv_address is not None and not isinstance(kv_address, str):
        raise argparse.ArgumentTypeError("endpoint.kv_address must be a string")
    return Endpoint(name=name, url=url.rstrip("/"), kv_address=kv_address)


def build_p2p_request_id(
    prefill_kv_address: str,
    decode_kv_address: str,
    *,
    request_uuid: str | None = None,
) -> str:
    suffix = request_uuid or uuid.uuid4().hex
    return (
        f"___prefill_addr_{prefill_kv_address}"
        f"___decode_addr_{decode_kv_address}_{suffix}"
    )


def _one_token_body(body: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(body)
    copied["max_tokens"] = 1
    copied["stream"] = False
    return copied


def build_p2p_prefill_body(
    body: dict[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    copied = _one_token_body(body)
    copied["request_id"] = request_id
    return copied


def build_nixl_prefill_body(body: dict[str, Any]) -> dict[str, Any]:
    copied = _one_token_body(body)
    copied["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return copied


def inject_kv_transfer_params(
    body: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    copied = copy.deepcopy(body)
    copied["kv_transfer_params"] = copy.deepcopy(params)
    return copied


class PdProxy:
    def __init__(
        self,
        *,
        connector: str,
        prefill: Iterable[Endpoint],
        decode: Iterable[Endpoint],
        session: Any,
    ) -> None:
        if connector not in {"p2p_nccl", "nixl"}:
            raise ValueError(f"unsupported connector: {connector}")
        self.connector = connector
        self.prefill = tuple(prefill)
        self.decode = tuple(decode)
        if not self.prefill or not self.decode:
            raise ValueError("prefill and decode endpoints are required")
        self._prefill_cycle = itertools.cycle(self.prefill)
        self._decode_cycle = itertools.cycle(self.decode)
        self.session = session

    async def handle_json_completion(
        self,
        path: str,
        body: dict[str, Any],
    ) -> tuple[int, dict[str, str], bytes]:
        decode, decode_body = await self.prepare_decode_request(path, body)
        return await self._post_json(decode, path, decode_body)

    async def prepare_decode_request(
        self,
        path: str,
        body: dict[str, Any],
    ) -> tuple[Endpoint, dict[str, Any]]:
        prefill = next(self._prefill_cycle)
        decode = next(self._decode_cycle)
        if self.connector == "p2p_nccl":
            request_id = build_p2p_request_id(
                _required_kv_address(prefill),
                _required_kv_address(decode),
            )
            prefill_body = build_p2p_prefill_body(body, request_id=request_id)
            status, _headers, payload = await self._post_json(
                prefill,
                path,
                prefill_body,
            )
            _raise_for_prefill_error(prefill, status, payload)
            decode_body = copy.deepcopy(body)
            decode_body["request_id"] = request_id
            return decode, decode_body

        prefill_body = build_nixl_prefill_body(body)
        status, _headers, prefill_payload = await self._post_json(
            prefill,
            path,
            prefill_body,
        )
        _raise_for_prefill_error(prefill, status, prefill_payload)
        prefill_json = json.loads(prefill_payload.decode("utf-8"))
        params = prefill_json.get("kv_transfer_params")
        if not isinstance(params, dict):
            raise web.HTTPBadGateway(
                reason="prefill response missing kv_transfer_params"
            )
        return decode, inject_kv_transfer_params(body, params)

    async def _post_json(
        self,
        endpoint: Endpoint,
        path: str,
        body: dict[str, Any],
    ) -> tuple[int, dict[str, str], bytes]:
        async with self.session.post(f"{endpoint.url}{path}", json=body) as response:
            payload = await response.read()
            return response.status, dict(response.headers), payload


def _required_kv_address(endpoint: Endpoint) -> str:
    if endpoint.kv_address is None:
        raise web.HTTPBadGateway(reason=f"{endpoint.name} missing kv_address")
    return endpoint.kv_address


def _raise_for_prefill_error(
    endpoint: Endpoint,
    status: int,
    payload: bytes,
) -> None:
    if 200 <= status < 300:
        return
    detail = _compact_error_payload(payload)
    message = f"prefill {endpoint.name} failed ({status})"
    if detail:
        message = f"{message}: {detail}"
    raise web.HTTPBadGateway(
        reason=f"prefill {endpoint.name} failed ({status})",
        text=message,
    )


def _compact_error_payload(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="replace").strip()
    text = " ".join(text.split())
    return text[:MAX_PREFILL_ERROR_CHARS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM PD proxy")
    parser.add_argument("--connector", choices=("p2p_nccl", "nixl"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--prefill",
        action="append",
        type=parse_endpoint,
        required=True,
    )
    parser.add_argument(
        "--decode",
        action="append",
        type=parse_endpoint,
        required=True,
    )
    return parser.parse_args(argv)


async def health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def root_v1(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app(proxy: PdProxy) -> web.Application:
    app = web.Application()
    app["proxy"] = proxy
    app.router.add_get("/health", health)
    app.router.add_get("/v1", root_v1)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/completions", handle_completion)
    app.router.add_post("/v1/chat/completions", handle_completion)
    return app


async def handle_completion(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(reason="request body must be a JSON object")
    proxy: PdProxy = request.app["proxy"]
    if body.get("stream") is True:
        return await _stream_completion(request, proxy, request.path, body)
    status, headers, payload = await proxy.handle_json_completion(
        request.path,
        body,
    )
    return web.Response(
        body=payload,
        status=status,
        headers=_response_headers(headers),
    )


async def _stream_completion(
    request: web.Request,
    proxy: PdProxy,
    path: str,
    body: dict[str, Any],
) -> web.StreamResponse:
    decode, decode_body = await proxy.prepare_decode_request(path, body)
    async with proxy.session.post(f"{decode.url}{path}", json=decode_body) as response:
        stream = web.StreamResponse(
            status=response.status,
            headers=_response_headers(response.headers),
        )
        await stream.prepare(request)
        async for chunk in response.content.iter_any():
            await stream.write(chunk)
        await stream.write_eof()
        return stream


async def handle_models(request: web.Request) -> web.StreamResponse:
    proxy: PdProxy = request.app["proxy"]
    endpoint = proxy.decode[0] if proxy.decode else proxy.prefill[0]
    async with proxy.session.get(f"{endpoint.url}/v1/models") as response:
        payload = await response.read()
        return web.Response(
            body=payload,
            status=response.status,
            headers=_response_headers(response.headers),
        )


def _response_headers(headers: Any) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).lower() not in HOP_BY_HOP_HEADERS
    }


async def _async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    async with ClientSession() as session:
        proxy = PdProxy(
            connector=args.connector,
            prefill=args.prefill,
            decode=args.decode,
            session=session,
        )
        runner = web.AppRunner(create_app(proxy))
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        await site.start()
        await asyncio.Event().wait()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
