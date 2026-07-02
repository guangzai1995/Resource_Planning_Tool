"""Report writers for benchmark outputs."""

import csv
import json
from pathlib import Path


def write_reports(output_dir, metrics, requests, errors, analysis):
    """Write benchmark summary, aggregate metrics, request rows, and errors."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = list(metrics)
    requests = list(requests)
    errors = list(errors)

    (output_path / "summary.md").write_text(
        _build_summary(metrics, analysis),
        encoding="utf-8",
    )
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_path / "metrics.csv", metrics)
    _write_jsonl(output_path / "requests.jsonl", requests)
    _write_jsonl(output_path / "errors.jsonl", errors)


def _build_summary(metrics, analysis):
    stable = analysis.get("stable_concurrency")
    peak = analysis.get("peak_throughput_concurrency")
    overload = analysis.get("overload_starts_at")
    all_failed = analysis.get("all_failed_at")

    lines = [
        "# Benchmark Summary",
        "",
        "## Conclusion",
        "",
        f"- Stable concurrency: {stable}",
        f"- Peak throughput concurrency: {peak}",
        f"- Overload starts at: {overload}",
    ]
    if all_failed is not None:
        lines.extend(
            [
                f"- All requests failed at: {all_failed}",
                "",
                "All requests failed at this concurrency, which is not a performance bottleneck. "
                "Check API configuration, authentication, or request shape before interpreting performance.",
            ]
        )

    lines.extend(["", "## Metrics", ""])

    if not metrics:
        lines.append("No metrics were recorded.")
        return "\n".join(lines) + "\n"

    headers = _fieldnames(metrics)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in metrics:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")

    return "\n".join(lines) + "\n"


def _write_csv(path, rows):
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _fieldnames(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
