#!/usr/bin/env python3
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.clients import prepare_request, request_json, response_to_record
from lib.config import ConfigError, config_to_dict, load_config
from lib.datasets import DatasetError, expand_samples, load_dataset
from lib.metrics import summarize
from lib.reporters import write_reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run standalone HTTP performance tests.")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Write request plan without sending traffic.")
    parser.add_argument("--output-dir", help="Override report output directory.")
    parser.add_argument("--limit", type=int, help="Override request count.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        samples = load_dataset(config.dataset.path)
    except (ConfigError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    request_count = args.limit if args.limit is not None else config.request_count
    if request_count <= 0:
        print("error: --limit must be positive", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.output_dir
    prepared_requests = [
        prepare_request(config.target, sample)
        for sample in expand_samples(samples, request_count + config.warmup_requests)
    ]

    if args.dry_run:
        _write_dry_run_plan(output_dir, config, prepared_requests, request_count)
        print(f"dry-run plan written to {output_dir / 'dry_run_plan.json'}")
        return 0

    for prepared in prepared_requests[: config.warmup_requests]:
        request_json(prepared, timeout_sec=config.timeout_sec)

    measured = prepared_requests[config.warmup_requests :]
    started_at = time.time()
    records = _execute_requests(measured, config.concurrency, config.timeout_sec)
    ended_at = time.time()
    summary = summarize(records, started_at, ended_at, run_name=config.run.name)
    written = write_reports(output_dir, summary, records)
    print(f"reports written: {', '.join(str(path) for path in written)}")
    return 0 if summary["failed_requests"] == 0 else 1


def _execute_requests(prepared_requests, concurrency, timeout_sec):
    records = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_item = {
            executor.submit(request_json, prepared, timeout_sec): (index, prepared)
            for index, prepared in enumerate(prepared_requests)
        }
        for future in as_completed(future_to_item):
            index, prepared = future_to_item[future]
            response = future.result()
            records.append(response_to_record(index, prepared, response))
    return sorted(records, key=lambda record: record["index"])


def _write_dry_run_plan(output_dir, config, prepared_requests, request_count):
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "config": config_to_dict(config),
        "target_type": config.target.type,
        "request_count": request_count,
        "concurrency": config.concurrency,
        "warmup_requests": config.warmup_requests,
        "requests": prepared_requests[:request_count],
    }
    (output_dir / "dry_run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
