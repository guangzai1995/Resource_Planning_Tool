---
name: automated-performance-testing
description: Use when running automated performance benchmarks for text, ASR, or generic HTTP model APIs, including concurrency sweeps, throughput analysis, latency analysis, and bottleneck detection.
---

# Automated Performance Testing

Use this skill when the user asks for load testing, performance benchmarking, concurrency sweeps, throughput, latency, bottleneck analysis, threshold checks, or performance regressions for a model API.

Do not use automated load testing as the first interface check. If the request shape, authentication, ASR upload, or endpoint path is still uncertain, switch to the manual interface performance testing skill first.

## Confirm Before Running

Confirm these inputs before sending load:

1. Target service: host, port, endpoint path, model name, credentials, and whether the service is already running.
2. Protocol: `openai_chat`, `openai_completion`, `openai_asr`, or `generic_http`.
3. Dataset: text JSONL prompts or audio manifest JSONL, plus any override path.
4. Load shape: `--concurrency` values and either `--epochs` or `--duration-seconds`.
5. Thresholds: optional `--max-error-rate`, `--max-p90-latency-ms`, and whether to use `--fail-fast`.
6. Output location: `--output-dir`, especially when comparing multiple runs.

`--fail-fast` stops only after a tier with zero completed requests, 100% request failure, or a `--max-error-rate` violation. The p90 threshold does not trigger fail-fast; a `--max-p90-latency-ms` violation is recorded and makes the command exit non-zero after the sweep completes.

## Required Workflow

1. Run a manual smoke check first, or confirm one has already succeeded.
2. Run automated dry-run to verify the config, dataset path, output directory, concurrency tiers, and request counts. In duration mode, dry-run marks the plan as duration-based because the final request count depends on response timing.
3. Run a small-concurrency smoke tier, typically `--concurrency 1 --epochs 1`.
4. Run the formal sweep with the agreed concurrency, epochs or duration, thresholds, and report directory.
5. Read the generated reports before making any bottleneck claim.

## Preferred script usage

Prefer the package wrapper over ad-hoc request loops:

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --dry-run --concurrency 1,2 --epochs 2
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1 --epochs 1 --output-dir reports/smoke
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2,4,8 --epochs 5 --max-error-rate 0.05 --fail-fast --output-dir reports/sweep
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2,4,8 --epochs 5 --max-p90-latency-ms 2000 --output-dir reports/latency-check
```

Direct Python entry points are also valid:

```bash
python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run
python3 scripts/perf_auto.py --config configs/openai_asr.json --concurrency 1,2 --epochs 3 --output-dir reports/asr-sweep
python3 scripts/perf_auto.py --config configs/generic_http.json --concurrency 1,2,4 --duration-seconds 30 --fail-fast --output-dir reports/generic-http
```

When using `--duration-seconds`, the runner continuously submits replacement requests until the tier timer expires. Treat `metrics.json`, `metrics.csv`, and `summary.md` as the source of truth for total request counts.

## Reading Reports

Automated runs write reports under `--output-dir`:

- `summary.md`: human-readable conclusion, stable concurrency, peak throughput tier, overload tier, and metric table.
- `metrics.json`: aggregate metrics per concurrency tier.
- `metrics.csv`: aggregate metrics in spreadsheet-friendly form.
- `requests.jsonl`: one normalized request result per line.
- `errors.jsonl`: failed request rows only.

Read reports in this order:

1. Start with `summary.md` to identify stable concurrency, peak throughput, overload start, and any all-failed tier.
2. Check `metrics.json` or `metrics.csv` for success rate, error rate, p50/p90/p99 latency, throughput, and request counts.
3. Inspect `errors.jsonl` for repeated `auth_error`, `bad_request`, `not_found`, `file_not_found`, timeout, or connection patterns.
4. Use `requests.jsonl` to inspect per-request latency and response summaries when a tier behaves unexpectedly.

## Bottleneck Rules

- A useful bottleneck signal needs successful requests plus a measurable degradation pattern: falling throughput, rising latency percentiles, growing error rate, or threshold violations as concurrency increases.
- A tier with 100% request failure is not a bottleneck. Treat 100% request failure as an interface, authentication, service availability, dataset, or request-shape problem.
- If failures start at low concurrency, run the manual interface performance testing skill before increasing load.
- Compare only runs with the same model, dataset, request body, target service configuration, hardware, duration or epochs, and concurrency plan.
- Do not extrapolate beyond the tested conditions. Report the exact command, config, dataset, thresholds, output directory, and time of the run.

## Safety Rules

- Do not treat authentication, authorization, route, bad request, missing file, or 100% request failure as a performance bottleneck.
- Fix interface and credentials first; do not tune concurrency against a broken request.
- Do not assume this package manages the target service lifecycle. The user or surrounding automation must start, stop, warm, and scale the service.
- Avoid unapproved high-concurrency runs against shared, paid, production, or third-party services.
