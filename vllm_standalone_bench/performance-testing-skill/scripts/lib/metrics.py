def summarize(records, started_at, ended_at, run_name="benchmark"):
    total = len(records)
    success = sum(1 for record in records if record.get("ok"))
    failed = total - success
    elapsed = max(float(ended_at) - float(started_at), 0.0)
    latencies = sorted(float(record.get("latency_sec", 0.0)) for record in records)
    return {
        "run_name": run_name,
        "total_requests": total,
        "success_requests": success,
        "failed_requests": failed,
        "duration_sec": round(elapsed, 6),
        "throughput_rps": round(total / elapsed, 6) if elapsed > 0 else 0.0,
        "latency_sec": {
            "min": _round(latencies[0]) if latencies else 0.0,
            "p50": _round(_percentile(latencies, 50)),
            "p90": _round(_percentile(latencies, 90)),
            "p95": _round(_percentile(latencies, 95)),
            "p99": _round(_percentile(latencies, 99)),
            "max": _round(latencies[-1]) if latencies else 0.0,
        },
    }


def _percentile(sorted_values, percentile):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _round(value):
    return round(float(value), 6)
