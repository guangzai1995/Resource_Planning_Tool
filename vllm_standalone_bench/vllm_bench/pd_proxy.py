from __future__ import annotations

import argparse
import copy
import itertools
import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from aiohttp import web


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str
    kv_address: str | None = None


def parse_endpoint(value: str) -> Endpoint:
    raw = json.loads(value)
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
        prefill = next(self._prefill_cycle)
        decode = next(self._decode_cycle)
        if self.connector == "p2p_nccl":
            request_id = build_p2p_request_id(
                _required_kv_address(prefill),
                _required_kv_address(decode),
            )
            prefill_body = build_p2p_prefill_body(body, request_id=request_id)
            await self._post_json(prefill, path, prefill_body)
            decode_body = copy.deepcopy(body)
            decode_body["request_id"] = request_id
            return await self._post_json(decode, path, decode_body)

        prefill_body = build_nixl_prefill_body(body)
        _status, _headers, prefill_payload = await self._post_json(
            prefill,
            path,
            prefill_body,
        )
        prefill_json = json.loads(prefill_payload.decode("utf-8"))
        params = prefill_json.get("kv_transfer_params")
        if not isinstance(params, dict):
            raise web.HTTPBadGateway(
                reason="prefill response missing kv_transfer_params"
            )
        return await self._post_json(
            decode,
            path,
            inject_kv_transfer_params(body, params),
        )

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
