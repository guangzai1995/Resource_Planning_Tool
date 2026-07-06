import csv
import json
from pathlib import Path


RESULT_FIELDS = ["index", "ok", "status_code", "latency_sec", "error", "method", "url"]


def write_reports(output_dir, summary, records):
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "results.json"
    csv_path = output_path / "results.csv"
    markdown_path = output_path / "summary.md"

    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, records)
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    return [json_path, csv_path, markdown_path]


def _write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _render_markdown(summary):
    lines = [
        "# Performance Test Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "run_name",
        "total_requests",
        "success_requests",
        "failed_requests",
        "duration_sec",
        "throughput_rps",
    ]:
        if key in summary:
            lines.append(f"| {key} | {summary[key]} |")
    latency = summary.get("latency_sec", {})
    for key in ["min", "p50", "p90", "p95", "p99", "max"]:
        if key in latency:
            lines.append(f"| latency_{key}_sec | {latency[key]} |")
    lines.append("")
    return "\n".join(lines)
