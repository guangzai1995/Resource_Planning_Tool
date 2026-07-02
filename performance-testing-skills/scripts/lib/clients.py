"""HTTP request construction helpers for benchmark clients."""

from __future__ import annotations

import json
import re
import shlex
import time
import urllib.error
import urllib.request
import uuid
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


def _summarize_text(text: str, limit: int = 500) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _extract_response_summary(payload: Any, fallback_text: str = "") -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return _summarize_text(str(message["content"]))
                if first.get("text") is not None:
                    return _summarize_text(str(first["text"]))
        if payload.get("text") is not None:
            return _summarize_text(str(payload["text"]))
    if fallback_text:
        return _summarize_text(fallback_text)
    return ""


def _extract_usage(payload: Any) -> tuple[int, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return 0, 0

    usage = payload["usage"]
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _encode_json_request(request: dict[str, Any], headers: dict[str, str]) -> bytes | None:
    if "json" not in request:
        return None

    headers.setdefault("Content-Type", "application/json")
    return json.dumps(request["json"]).encode("utf-8")


def _encode_multipart_request(request: dict[str, Any], headers: dict[str, str]) -> bytes:
    boundary = f"----benchmark-{uuid.uuid4().hex}"
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    chunks: list[bytes] = []
    for key, value in (request.get("multipart") or {}).items():
        if value in (None, ""):
            continue
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])

    multipart_file = request.get("multipart_file")
    if multipart_file:
        filename = str(multipart_file).rsplit("/", 1)[-1]
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
            b"Content-Type: application/octet-stream\r\n\r\n",
        ])
        with open(multipart_file, "rb") as file_obj:
            chunks.append(file_obj.read())
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def _build_urllib_request(request: dict[str, Any]) -> urllib.request.Request:
    headers = dict(request.get("headers") or {})
    data = _encode_json_request(request, headers)
    if "multipart" in request or request.get("multipart_file"):
        data = _encode_multipart_request(request, headers)

    return urllib.request.Request(
        url=str(request["url"]),
        data=data,
        headers=headers,
        method=str(request.get("method", "POST")).upper(),
    )


def _base_result(request_id: str, start: float) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "success": False,
        "status_code": None,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "input_tokens": 0,
        "output_tokens": 0,
        "response_summary": "",
    }


def _parse_response_body(body: bytes) -> tuple[Any, str]:
    text = body.decode("utf-8", errors="replace")
    if not text:
        return None, ""
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        return None, text


def send_request(request_id: str, request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    """Send a benchmark request and return normalized timing/result metadata."""
    start = time.perf_counter()
    try:
        http_request = _build_urllib_request(request)
        with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
            body = response.read()
            status_code = response.getcode()

        payload, text = _parse_response_body(body)
        input_tokens, output_tokens = _extract_usage(payload)
        result = _base_result(request_id, start)
        result.update({
            "success": 200 <= status_code <= 299,
            "status_code": status_code,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_summary": _extract_response_summary(payload, text),
        })
        return result
    except urllib.error.HTTPError as exc:
        body = exc.read()
        payload, text = _parse_response_body(body)
        input_tokens, output_tokens = _extract_usage(payload)
        result = _base_result(request_id, start)
        result.update({
            "status_code": exc.code,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_summary": _extract_response_summary(payload, text),
            "error_type": classify_error(exc.code, exc),
            "message": str(exc),
        })
        return result
    except Exception as exc:
        result = _base_result(request_id, start)
        result.update({
            "error_type": classify_error(None, exc),
            "message": str(exc),
        })
        return result
