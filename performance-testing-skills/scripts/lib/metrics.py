"""Metrics helpers for benchmark results."""


def percentile(values, pct):
    """Return the nearest-rank percentile used by the benchmark reports."""
    if not values:
        return 0.0

    sorted_values = sorted(float(value) for value in values)
    raw_index = (len(sorted_values) - 1) * pct / 100
    index = int(round(raw_index + 1e-12))
    return sorted_values[index]


def aggregate_results(rows, concurrency, duration_s):
    """Aggregate per-request rows into one metrics record."""
    rows = list(rows)
    duration_s = float(duration_s)
    total_requests = len(rows)
    success_rows = [row for row in rows if row.get("success")]
    failed_rows = [row for row in rows if not row.get("success")]

    n_success = len(success_rows)
    n_failed = len(failed_rows)
    success_rate = n_success / total_requests if total_requests else 0.0
    error_rate = n_failed / total_requests if total_requests else 0.0
    safe_duration_s = duration_s if duration_s > 0 else 0.0

    latencies = [float(row.get("latency_ms", 0.0)) for row in success_rows]
    output_tokens_total = sum(float(row.get("output_tokens", 0.0)) for row in success_rows)

    metrics = {
        "concurrency": concurrency,
        "n_requests": total_requests,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p90_ms": percentile(latencies, 90),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "request_throughput_req_s": n_success / safe_duration_s if safe_duration_s else 0.0,
        "output_tokens_total": output_tokens_total,
        "output_throughput_tok_s": output_tokens_total / safe_duration_s if safe_duration_s else 0.0,
    }

    audio_durations = [
        float(row["audio_duration_s"])
        for row in rows
        if row.get("audio_duration_s") is not None
    ]
    if audio_durations:
        audio_duration_s_total = sum(audio_durations)
        metrics.update(
            {
                "audio_duration_s_total": audio_duration_s_total,
                "audio_duration_s_avg": audio_duration_s_total / len(audio_durations),
                "audio_rtfx": audio_duration_s_total / safe_duration_s if safe_duration_s else 0.0,
            }
        )

    return metrics


def analyze_bottleneck(metrics):
    """Infer stable, peak, and overloaded concurrency from aggregate metrics."""
    sorted_metrics = sorted(metrics, key=lambda item: item.get("concurrency", 0))
    if not sorted_metrics:
        return {
            "stable_concurrency": None,
            "peak_throughput_concurrency": None,
            "overload_starts_at": None,
        }

    peak = max(sorted_metrics, key=lambda item: item.get("request_throughput_req_s", 0.0))
    overload_starts_at = None
    stable_concurrency = None

    for index, current in enumerate(sorted_metrics):
        if overload_starts_at is not None:
            break

        current_concurrency = current.get("concurrency")
        current_success_rate = current.get("success_rate", 0.0)
        overloaded = current_success_rate < 1.0

        if index > 0:
            previous = sorted_metrics[index - 1]
            previous_throughput = previous.get("request_throughput_req_s", 0.0)
            current_throughput = current.get("request_throughput_req_s", 0.0)
            previous_p90 = previous.get("latency_p90_ms", 0.0)
            current_p90 = current.get("latency_p90_ms", 0.0)

            throughput_declined = current_throughput < previous_throughput
            throughput_gain = (
                (current_throughput - previous_throughput) / previous_throughput
                if previous_throughput
                else 0.0
            )
            p90_growth = (
                (current_p90 - previous_p90) / previous_p90
                if previous_p90
                else 0.0
            )
            latency_inflection = throughput_gain < 0.30 and p90_growth > 0.50
            overloaded = overloaded or throughput_declined or latency_inflection

        if overloaded:
            overload_starts_at = current_concurrency
        elif current_success_rate == 1.0:
            stable_concurrency = current_concurrency

    return {
        "stable_concurrency": stable_concurrency,
        "peak_throughput_concurrency": peak.get("concurrency"),
        "overload_starts_at": overload_starts_at,
    }
