"""HTTP request construction helpers for benchmark clients."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any


_TEMPLATE_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _sample_get(sample: dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _base_url(config: dict[str, Any]) -> str:
    return str(config.get("base_url", "")).rstrip("/")


def _headers(config: dict[str, Any]) -> dict[str, str]:
    headers = config.get("headers") or {}
    return {str(key): str(value) for key, value in headers.items() if value not in (None, "")}


def _replace_template(value: Any, sample: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value == "${prompt}":
            return _sample_get(sample, "prompt", "")
        if value == "${max_tokens}":
            return int(_sample_get(sample, "expected_output_len", None) or 128)

        def replace_match(match: re.Match[str]) -> str:
            replacement = _sample_get(sample, match.group(1), "")
            return "" if replacement is None else str(replacement)

        return _TEMPLATE_PATTERN.sub(replace_match, value)
    if isinstance(value, list):
        return [_replace_template(item, sample) for item in value]
    if isinstance(value, dict):
        return {key: _replace_template(item, sample) for key, item in value.items()}
    return value


def build_request(config: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol")
    headers = _headers(config)

    if protocol == "openai_chat":
        body = {
            "model": config["model"],
            "messages": [{"role": "user", "content": _sample_get(sample, "prompt", "")}],
        }
        body.update(config.get("request") or {})
        return {
            "method": "POST",
            "url": f"{_base_url(config)}/chat/completions",
            "headers": headers,
            "json": body,
        }

    if protocol == "openai_completion":
        body = {
            "model": config["model"],
            "prompt": _sample_get(sample, "prompt", ""),
        }
        body.update(config.get("request") or {})
        return {
            "method": "POST",
            "url": f"{_base_url(config)}/completions",
            "headers": headers,
            "json": body,
        }

    if protocol == "openai_asr":
        multipart = {"model": config["model"]}
        multipart.update(config.get("request") or {})
        if _sample_get(sample, "prompt"):
            multipart["prompt"] = _sample_get(sample, "prompt")
        return {
            "method": "POST",
            "url": f"{_base_url(config)}/audio/transcriptions",
            "headers": headers,
            "multipart": multipart,
            "multipart_file": _sample_get(sample, "audio"),
        }

    if protocol == "generic_http":
        request = {
            "method": str(config.get("method", "POST")).upper(),
            "url": config["url"],
            "headers": headers,
        }
        if "body_template" in config:
            request["json"] = _replace_template(config["body_template"], sample)
        return request

    raise ValueError(f"Unsupported protocol: {protocol}")


def make_curl(request: dict[str, Any]) -> str:
    headers = dict(request.get("headers") or {})
    parts = ["curl", "-X", str(request.get("method", "POST")).upper()]

    if "json" in request:
        headers.setdefault("Content-Type", "application/json")

    for key, value in headers.items():
        if value not in (None, ""):
            parts.extend(["-H", f"{key}: {value}"])

    if "json" in request:
        payload = json.dumps(request["json"], ensure_ascii=False)
        parts.extend(["--data", payload])

    if "multipart" in request:
        for key, value in request["multipart"].items():
            if value not in (None, ""):
                parts.extend(["-F", f"{key}={value}"])
        if request.get("multipart_file"):
            parts.extend(["-F", f"file=@{request['multipart_file']}"])

    parts.append(request["url"])
    return " ".join(shlex.quote(str(part)) for part in parts)


def classify_error(status_code: int | None, exception: BaseException | None) -> str:
    if status_code in (401, 403):
        return "auth_error"
    if status_code == 404:
        return "not_found"
    if status_code in (400, 422):
        return "bad_request"
    if status_code is not None and 500 <= status_code <= 599:
        return "http_5xx"

    if exception is not None:
        message = f"{type(exception).__name__}: {exception}".lower()
        if "timeout" in message or "timed out" in message:
            return "timeout"
        if "broken pipe" in message:
            return "broken_pipe"
        if "disconnect" in message or "disconnected" in message:
            return "disconnect"
        if "connection reset" in message:
            return "connection_reset"

    return "unknown_error"
