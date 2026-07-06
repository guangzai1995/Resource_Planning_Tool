---
name: generic-performance-testing
description: Use when Codex needs to create, run, or troubleshoot standalone HTTP/OpenAI-compatible performance tests, benchmark configs, latency/throughput summaries, or CSV/JSON/Markdown reports without depending on a project-specific benchmark framework.
---

# Generic Performance Testing

This skill packages a standalone performance testing workflow for HTTP APIs, OpenAI-compatible chat/completion endpoints, and lightweight ASR request planning. It does not import or require the parent benchmark project.

## Workflow

1. Pick a config from `configs/` or copy one close to the target API.
2. Put prompt or audio manifest samples in `datasets/`.
3. Run `scripts/run_auto.sh --config configs/openai_chat.json --dry-run` to validate the generated requests before sending traffic.
4. Run `scripts/run_auto.sh --config <config>` for a benchmark run.
5. Use `scripts/run_manual.sh --url <url> --body '{"prompt":"hello"}'` for a single request smoke test.
6. Inspect `reports/results.json`, `reports/results.csv`, and `reports/summary.md`, or pass `--output-dir` to write reports elsewhere.

## Shell Usage

Automatic benchmark dry run:

```bash
scripts/run_auto.sh --config configs/openai_chat.json --dry-run --limit 3
```

Automatic benchmark execution:

```bash
scripts/run_auto.sh --config configs/generic_http.json --output-dir reports/generic-smoke
```

Manual one-shot request:

```bash
scripts/run_manual.sh \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --header "Authorization: Bearer token" \
  --body '{"model":"demo","messages":[{"role":"user","content":"hello"}]}'
```

Set `PYTHON_BIN=/path/to/python` when the wrapper should use a specific Python interpreter.

## Config Files

Each config has three top-level sections:

- `run`: benchmark name, request count, concurrency, warmup count, timeout, and output directory.
- `target`: endpoint type, base URL, path, model name, headers, and optional generic body template.
- `dataset`: JSONL path containing prompt or audio manifest samples.

Supported `target.type` values:

- `openai-chat`: builds `/v1/chat/completions` style JSON bodies.
- `openai-completion`: builds `/v1/completions` style JSON bodies.
- `openai-asr`: builds JSON request plans for audio transcription endpoints; use dry-run first to verify file paths and payload shape.
- `generic-http`: renders `body_template` values with sample fields such as `{prompt}`.

Relative dataset paths are resolved from the package root, so the skill can be moved as a self-contained folder.

## Reports

`scripts/perf_auto.py` writes:

- `results.json`: complete summary and per-request records.
- `results.csv`: per-request rows for spreadsheets.
- `summary.md`: compact human-readable latency and throughput summary.

The implementation uses only Python standard library modules.
