"""Automated performance testing CLI."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.lib.clients import build_request, send_request
from scripts.lib.config import load_config, resolve_package_path
from scripts.lib.datasets import load_dataset
from scripts.lib.metrics import aggregate_results, analyze_bottleneck
from scripts.lib.reporters import write_reports


DEFAULT_CONFIG = "configs/openai_chat.json"
DEFAULT_OUTPUT_DIR = "reports/latest"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _error_rate_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number from 0 to 1") from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be a finite number from 0 to 1")
    return parsed


def _parse_concurrency(values: list[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                raise argparse.ArgumentTypeError("must contain positive integers")
            parsed.append(_positive_int(chunk))

    if not parsed:
        raise argparse.ArgumentTypeError("must contain at least one positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated API performance tests.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Config file path. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset path overriding the dataset path in the config.",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        default=["1"],
        metavar="N",
        help="Comma or space separated positive concurrency values. Defaults to 1.",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=1,
        help="Requests per worker for each concurrency value. Defaults to 1.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=_positive_float,
        help="Optional finite positive duration for time-based load generation.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed concurrency tier.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=_error_rate_threshold,
        help="Optional maximum allowed error rate, from 0 to 1 inclusive.",
    )
    parser.add_argument(
        "--max-p90-latency-ms",
        type=_positive_float,
        help="Optional maximum allowed p90 latency in milliseconds.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=60.0,
        help="Request timeout in seconds. Defaults to 60.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the test plan without sending network traffic.",
    )

    args = parser.parse_args(argv)
    try:
        args.concurrency = _parse_concurrency(args.concurrency)
    except argparse.ArgumentTypeError as exc:
        parser.error(f"argument --concurrency: {exc}")
    return args


def _apply_dataset_override(config: dict[str, Any], dataset_path: str | None) -> dict[str, Any]:
    if dataset_path is None:
        return config

    updated = dict(config)
    dataset_config = dict(updated.get("dataset") or {})
    dataset_config.setdefault("type", "text_prompts")
    dataset_config["path"] = dataset_path
    updated["dataset"] = dataset_config
    return updated


def _dataset_path(config: dict[str, Any]) -> str:
    dataset_config = config.get("dataset")
    if isinstance(dataset_config, dict) and dataset_config.get("path"):
        return str(resolve_package_path(dataset_config["path"]))
    return "(inline default sample)"


def _resolve_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return PACKAGE_ROOT / output_path


def _print_dry_run(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> None:
    print("DRY RUN")
    print(f"config path: {config_path}")
    print(f"dataset path: {_dataset_path(config)}")
    print(f"output dir: {output_dir}")
    print(f"timeout_seconds={args.timeout_seconds:g}")
    if args.duration_seconds is not None:
        print(f"duration_seconds={args.duration_seconds:g}")
    for concurrency in args.concurrency:
        request_count = concurrency * args.epochs
        print(f"concurrency={concurrency} request_count={request_count}")


def _sample_for_index(dataset: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return dataset[(index - 1) % len(dataset)]


def _run_one_request(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    concurrency: int,
    index: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    sample = _sample_for_index(dataset, index)
    request = build_request(config, sample)
    result = send_request(
        request_id=f"auto-c{concurrency}-{index}",
        request=request,
        timeout_seconds=timeout_seconds,
    )
    result["concurrency"] = concurrency
    if sample.get("duration_s") is not None:
        result["audio_duration_s"] = sample["duration_s"]
    return result


def _run_fixed_count(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    concurrency: int,
    request_count: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _run_one_request,
                config,
                dataset,
                concurrency,
                index,
                timeout_seconds,
            )
            for index in range(1, request_count + 1)
        ]
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    return rows, time.perf_counter() - start


def _run_for_duration(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    concurrency: int,
    duration_seconds: float,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    next_index = 1
    start = time.perf_counter()
    deadline = start + duration_seconds

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: set[concurrent.futures.Future[dict[str, Any]]] = set()

        for _ in range(concurrency):
            futures.add(
                executor.submit(
                    _run_one_request,
                    config,
                    dataset,
                    concurrency,
                    next_index,
                    timeout_seconds,
                )
            )
            next_index += 1

        while futures:
            done, futures = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                rows.append(future.result())

            while time.perf_counter() < deadline and len(futures) < concurrency:
                futures.add(
                    executor.submit(
                        _run_one_request,
                        config,
                        dataset,
                        concurrency,
                        next_index,
                        timeout_seconds,
                    )
                )
                next_index += 1

    return rows, time.perf_counter() - start


def _format_summary(metrics: dict[str, Any]) -> str:
    return (
        f"concurrency={metrics['concurrency']} "
        f"request_count={metrics['n_requests']} "
        f"success_rate={metrics['success_rate']:.3f} "
        f"p90={metrics['latency_p90_ms']:.2f} "
        f"throughput={metrics['request_throughput_req_s']:.3f}"
    )


def _thresholds_violated(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    violations: list[str] = []
    if metrics.get("n_requests", 0) == 0:
        violations.append("no requests completed")
    elif metrics.get("success_rate", 0.0) == 0.0:
        violations.append("all requests failed")
    if (
        args.max_error_rate is not None
        and metrics.get("error_rate", 0.0) > args.max_error_rate
    ):
        violations.append(
            f"error_rate {metrics.get('error_rate', 0.0):.3f} > {args.max_error_rate:.3f}"
        )
    if (
        args.max_p90_latency_ms is not None
        and metrics.get("latency_p90_ms", 0.0) > args.max_p90_latency_ms
    ):
        violations.append(
            f"p90 {metrics.get('latency_p90_ms', 0.0):.2f}ms > "
            f"{args.max_p90_latency_ms:.2f}ms"
        )
    return violations


def _should_fail_fast(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    if not args.fail_fast:
        return False
    if metrics.get("n_requests", 0) == 0:
        return True
    if metrics.get("success_rate", 0.0) == 0.0:
        return True
    return bool(
        args.max_error_rate is not None
        and metrics.get("error_rate", 0.0) > args.max_error_rate
    )


def run(args: argparse.Namespace) -> int:
    config_path = resolve_package_path(args.config)
    config = _apply_dataset_override(load_config(config_path), args.dataset)
    output_dir = _resolve_output_dir(args.output_dir)

    if args.dry_run:
        _print_dry_run(args, config, config_path, output_dir)
        return 0

    dataset = load_dataset(config.get("dataset"))
    metrics_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    threshold_violations: list[str] = []

    for concurrency in args.concurrency:
        if args.duration_seconds is None:
            rows, duration_s = _run_fixed_count(
                config,
                dataset,
                concurrency,
                concurrency * args.epochs,
                args.timeout_seconds,
            )
        else:
            rows, duration_s = _run_for_duration(
                config,
                dataset,
                concurrency,
                args.duration_seconds,
                args.timeout_seconds,
            )

        request_rows.extend(rows)
        error_rows.extend(row for row in rows if not row.get("success"))

        metrics = aggregate_results(rows, concurrency=concurrency, duration_s=duration_s)
        metrics_rows.append(metrics)
        print(_format_summary(metrics))

        violations = _thresholds_violated(metrics, args)
        threshold_violations.extend(
            f"concurrency={concurrency}: {violation}" for violation in violations
        )

        if _should_fail_fast(metrics, args):
            print(f"fail_fast_stop concurrency={concurrency}")
            break

    write_reports(
        output_dir,
        metrics_rows,
        request_rows,
        error_rows,
        analyze_bottleneck(metrics_rows),
    )

    for violation in threshold_violations:
        print(f"threshold_violation {violation}", file=sys.stderr)

    return 1 if threshold_violations else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
