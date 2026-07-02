"""Manual interface performance testing CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.lib.clients import build_request, make_curl, send_request
from scripts.lib.config import load_config, resolve_package_path
from scripts.lib.datasets import load_dataset


DEFAULT_CONFIG = "configs/openai_chat.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually validate a model API request or print its curl form."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Config file path. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--mode",
        choices=("request", "curl"),
        default="request",
        help="Run a request smoke test or only print curl. Defaults to request.",
    )
    parser.add_argument("--input", help="Text input overriding the sample prompt.")
    parser.add_argument(
        "--audio-file",
        help="Audio file path overriding the sample audio file for ASR configs.",
    )
    parser.add_argument(
        "--request-count",
        type=_positive_int,
        default=1,
        help="Number of requests to send. Defaults to 1.",
    )
    parser.add_argument(
        "--print-curl",
        action="store_true",
        help="Print the equivalent curl command before sending requests.",
    )
    parser.add_argument(
        "--save-response",
        type=Path,
        help="Optional path where response JSON will be saved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and print the request without sending network traffic.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=60.0,
        help="Request timeout in seconds. Defaults to 60.",
    )
    return parser.parse_args(argv)


def _load_sample(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rows = load_dataset(config.get("dataset"))
    sample = dict(rows[0])

    if args.input is not None:
        sample["prompt"] = args.input
    if args.audio_file is not None:
        sample["audio"] = args.audio_file

    return sample


def _print_dry_run(config: dict[str, Any], request: dict[str, Any], curl: str) -> None:
    print("DRY RUN")
    print(f"protocol: {config.get('protocol')}")
    print(f"method: {request.get('method', 'POST')}")
    print(f"url: {request.get('url')}")
    print("curl:")
    print(curl)


def _format_latency(result: dict[str, Any]) -> str:
    latency = result.get("latency_ms")
    if isinstance(latency, (int, float)):
        return f"{latency:.2f}"
    return ""


def _print_result(result: dict[str, Any]) -> None:
    pieces = [
        f"request_id={result.get('request_id')}",
        f"success={result.get('success')}",
        f"status_code={result.get('status_code')}",
        f"latency_ms={_format_latency(result)}",
        f"response_summary={result.get('response_summary', '')}",
    ]
    if result.get("error_type"):
        pieces.append(f"error_type={result.get('error_type')}")
    if result.get("error_message"):
        pieces.append(f"error_message={result.get('error_message')}")
    print(" ".join(pieces))


def _save_response(path: Path, results: list[dict[str, Any]]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"requests": results}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    config = load_config(resolve_package_path(args.config))
    sample = _load_sample(config, args)
    request = build_request(config, sample)
    curl = make_curl(request)

    if args.mode == "curl":
        print(curl)
        if args.save_response:
            _save_response(args.save_response, [])
        return 0

    if args.dry_run:
        _print_dry_run(config, request, curl)
        if args.save_response:
            _save_response(args.save_response, [])
        return 0

    if args.print_curl:
        print(curl)

    results: list[dict[str, Any]] = []
    for index in range(1, args.request_count + 1):
        request_id = f"manual-{index}"
        result = send_request(request_id, request, args.timeout_seconds)
        results.append(result)
        _print_result(result)

    if args.save_response:
        _save_response(args.save_response, results)

    return 0 if all(result.get("success") for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
