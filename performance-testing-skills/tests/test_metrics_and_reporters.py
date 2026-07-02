import csv
import json

from scripts.lib.metrics import aggregate_results, analyze_bottleneck
from scripts.lib.reporters import write_reports


def test_aggregate_results_computes_latency_and_success_rate():
    rows = [
        {"success": True, "latency_ms": 100.0, "output_tokens": 10, "audio_duration_s": 30.0},
        {"success": True, "latency_ms": 200.0, "output_tokens": 20, "audio_duration_s": 30.0},
        {"success": False, "latency_ms": 300.0, "output_tokens": 0, "audio_duration_s": 30.0},
    ]

    metrics = aggregate_results(rows, concurrency=4, duration_s=10.0)

    assert metrics["concurrency"] == 4
    assert metrics["n_success"] == 2
    assert metrics["n_failed"] == 1
    assert metrics["success_rate"] == 2 / 3
    assert metrics["latency_p50_ms"] == 200.0
    assert metrics["request_throughput_req_s"] == 0.2
    assert metrics["audio_rtfx"] == 9.0


def test_analyze_bottleneck_finds_stable_and_overloaded_concurrency():
    metrics = [
        {"concurrency": 16, "success_rate": 1.0, "request_throughput_req_s": 0.50, "latency_p90_ms": 30000.0},
        {"concurrency": 32, "success_rate": 1.0, "request_throughput_req_s": 0.61, "latency_p90_ms": 58000.0},
        {"concurrency": 64, "success_rate": 0.93, "request_throughput_req_s": 0.65, "latency_p90_ms": 120000.0},
    ]

    analysis = analyze_bottleneck(metrics)

    assert analysis["stable_concurrency"] == 16
    assert analysis["peak_throughput_concurrency"] == 64
    assert analysis["overload_starts_at"] == 32


def test_analyze_bottleneck_treats_all_failed_runs_as_invalid_not_overloaded():
    metrics = [
        {"concurrency": 16, "success_rate": 1.0, "request_throughput_req_s": 0.50, "latency_p90_ms": 30000.0},
        {"concurrency": 32, "success_rate": 0.0, "request_throughput_req_s": 0.0, "latency_p90_ms": 0.0},
    ]

    analysis = analyze_bottleneck(metrics)

    assert analysis["stable_concurrency"] == 16
    assert analysis["overload_starts_at"] is None
    assert analysis["all_failed_at"] == 32


def test_write_reports_creates_summary_json_and_csv(tmp_path):
    metrics = [{"concurrency": 1, "n_success": 3, "n_failed": 0, "success_rate": 1.0, "request_throughput_req_s": 0.1}]
    requests = [{"request_id": "req-1", "success": True, "latency_ms": 100.0}]
    errors = []

    write_reports(tmp_path, metrics, requests, errors, {"stable_concurrency": 1})

    assert (tmp_path / "summary.md").exists()
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))[0]["concurrency"] == 1
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["concurrency"] == "1"


def test_write_reports_warns_that_all_failed_runs_are_not_bottlenecks(tmp_path):
    metrics = [{"concurrency": 32, "n_success": 0, "n_failed": 3, "success_rate": 0.0, "request_throughput_req_s": 0.0}]
    requests = [{"request_id": "req-1", "success": False}]
    errors = [{"request_id": "req-1", "error": "401 Unauthorized"}]
    analysis = {"stable_concurrency": None, "overload_starts_at": None, "all_failed_at": 32}

    write_reports(tmp_path, metrics, requests, errors, analysis)

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "All requests failed at: 32" in summary
    assert "not a performance bottleneck" in summary
    assert "API configuration, authentication, or request shape" in summary
