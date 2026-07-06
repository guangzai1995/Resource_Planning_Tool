import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace


def prepare_request(target, sample):
    target = _as_mapping(target)
    sample = _as_mapping(sample)
    target_type = target.get("type", "generic-http")
    method = target.get("method", "POST").upper()
    headers = {"Content-Type": "application/json", **dict(target.get("headers") or {})}
    body = _build_body(target_type, target, sample)
    return {
        "method": method,
        "url": _join_url(target.get("base_url", ""), target.get("endpoint", "")),
        "headers": headers,
        "body": body,
    }


def request_json(prepared, timeout_sec=30):
    body_bytes = json.dumps(prepared.get("body") or {}).encode("utf-8")
    request = urllib.request.Request(
        prepared["url"],
        data=body_bytes,
        headers=prepared.get("headers") or {},
        method=prepared.get("method", "POST"),
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            content = response.read()
            latency = time.perf_counter() - started
            json_body = _decode_json(content)
            return SimpleNamespace(
                ok=200 <= response.status < 400,
                status_code=response.status,
                latency_sec=latency,
                headers=dict(response.headers),
                body=content.decode("utf-8", errors="replace"),
                json_body=json_body,
                error="",
            )
    except urllib.error.HTTPError as exc:
        content = exc.read()
        latency = time.perf_counter() - started
        return SimpleNamespace(
            ok=False,
            status_code=exc.code,
            latency_sec=latency,
            headers=dict(exc.headers),
            body=content.decode("utf-8", errors="replace"),
            json_body=_decode_json(content),
            error=str(exc),
        )
    except Exception as exc:
        latency = time.perf_counter() - started
        return SimpleNamespace(
            ok=False,
            status_code=0,
            latency_sec=latency,
            headers={},
            body="",
            json_body=None,
            error=str(exc),
        )


def response_to_record(index, prepared, response):
    return {
        "index": index,
        "ok": bool(response.ok),
        "status_code": int(response.status_code),
        "latency_sec": round(float(response.latency_sec), 6),
        "error": response.error,
        "url": prepared["url"],
        "method": prepared["method"],
    }


def _build_body(target_type, target, sample):
    prompt = sample.get("prompt") or sample.get("text") or ""
    model = target.get("model")
    extra_body = dict(target.get("extra_body") or {})

    if target_type == "openai-chat":
        body = {
            "model": model,
            "messages": [
                {
                    "role": sample.get("role", "user"),
                    "content": prompt,
                }
            ],
        }
        body.update(extra_body)
        return body

    if target_type == "openai-completion":
        body = {"model": model, "prompt": prompt}
        body.update(extra_body)
        return body

    if target_type == "openai-asr":
        body = {
            "model": model,
            "audio_path": sample.get("audio_path"),
            "prompt": prompt,
        }
        body.update(extra_body)
        return body

    template = target.get("body_template")
    if template:
        return _render_template(template, sample)
    body = dict(sample)
    body.update(extra_body)
    return body


def _render_template(value, sample):
    if isinstance(value, dict):
        return {key: _render_template(item, sample) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_template(item, sample) for item in value]
    if isinstance(value, str):
        return value.format_map(_SafeFormat(sample))
    return value


class _SafeFormat(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _join_url(base_url, endpoint):
    base_url = str(base_url).rstrip("/")
    endpoint = str(endpoint or "")
    if endpoint and not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{base_url}{endpoint}"


def _decode_json(content):
    if not content:
        return None
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _as_mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value.__dict__)
