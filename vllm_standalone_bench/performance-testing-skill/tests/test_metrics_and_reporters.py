import csv
import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarizes_latency_and_throughput():
    metrics = load_module("skill_metrics", "scripts/lib/metrics.py")
    records = [
        {"ok": True, "latency_sec": 0.10, "status_code": 200},
        {"ok": True, "latency_sec": 0.20, "status_code": 200},
        {"ok": True, "latency_sec": 0.30, "status_code": 200},
        {"ok": False, "latency_sec": 0.40, "status_code": 500},
    ]

    summary = metrics.summarize(records, started_at=10.0, ended_at=12.0)

    assert summary["total_requests"] == 4
    assert summary["success_requests"] == 3
    assert summary["failed_requests"] == 1
    assert summary["throughput_rps"] == 2.0
    assert summary["latency_sec"]["p50"] == 0.25
    assert summary["latency_sec"]["p95"] == 0.385


def test_writes_json_csv_and_markdown_reports(tmp_path):
    reporters = load_module("skill_reporters", "scripts/lib/reporters.py")
    records = [
        {
            "index": 0,
            "ok": True,
            "status_code": 200,
            "latency_sec": 0.12,
            "error": "",
        }
    ]
    summary = {
        "run_name": "smoke",
        "total_requests": 1,
        "success_requests": 1,
        "failed_requests": 0,
        "throughput_rps": 8.33,
        "latency_sec": {"p50": 0.12, "p95": 0.12, "p99": 0.12},
    }

    written = reporters.write_reports(tmp_path, summary, records)

    assert {path.name for path in written} == {"results.json", "results.csv", "summary.md"}
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["summary"] == summary

    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status_code"] == "200"

    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "# Performance Test Summary" in markdown
    assert "| total_requests | 1 |" in markdown
