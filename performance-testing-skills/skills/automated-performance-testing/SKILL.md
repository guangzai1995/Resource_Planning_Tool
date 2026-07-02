---
name: automated-performance-testing
description: Use when running automated performance benchmarks for text, ASR, or generic HTTP model APIs, including concurrency sweeps, throughput analysis, latency analysis, and bottleneck detection.
---

# Automated Performance Testing

Use this skill when the user asks for load testing, performance benchmarking, concurrency sweeps, throughput, latency, bottleneck analysis, or regression performance testing.

## Required Workflow

1. Identify the protocol: `openai_chat`, `openai_completion`, `openai_asr`, or `generic_http`.
2. Confirm the benchmark scale: concurrency list plus either epochs or duration.
3. Confirm the dataset: text JSONL, audio manifest JSONL, or a user-provided path.
4. Run a smoke request before the benchmark.
5. Prefer the package scripts over ad-hoc request loops:

```bash
python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run
```

6. After execution, analyze success rate, throughput, latency percentiles, error types, stable concurrency, and bottleneck range.

## Safety Rules

- Do not treat 100% request failure as a performance bottleneck.
- If authentication, URL, or request-shape errors appear, switch to the manual interface testing skill first.
- Do not assume the target service lifecycle is managed by this package.
