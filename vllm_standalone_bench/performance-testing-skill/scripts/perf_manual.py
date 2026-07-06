#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.clients import request_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Send one manual HTTP JSON request.")
    parser.add_argument("--url", required=True, help="Full request URL.")
    parser.add_argument("--method", default="POST", help="HTTP method. Default: POST.")
    parser.add_argument("--header", action="append", default=[], help="Header as 'Name: value'.")
    parser.add_argument("--body", default="{}", help="JSON body string.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print prepared request only.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        body = json.loads(args.body)
    except json.JSONDecodeError as exc:
        print(f"error: --body must be valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(body, dict):
        print("error: --body must be a JSON object", file=sys.stderr)
        return 2

    try:
        headers = _parse_headers(args.header)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    headers.setdefault("Content-Type", "application/json")

    prepared = {
        "method": args.method.upper(),
        "url": args.url,
        "headers": headers,
        "body": body,
    }
    if args.dry_run:
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        return 0

    response = request_json(prepared, timeout_sec=args.timeout)
    print(
        json.dumps(
            {
                "ok": response.ok,
                "status_code": response.status_code,
                "latency_sec": round(response.latency_sec, 6),
                "body": response.json_body if response.json_body is not None else response.body,
                "error": response.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if response.ok else 1


def _parse_headers(values):
    headers = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"header must use 'Name: value' format: {value}")
        name, header_value = value.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"header name is empty: {value}")
        headers[name] = header_value.strip()
    return headers


if __name__ == "__main__":
    raise SystemExit(main())
