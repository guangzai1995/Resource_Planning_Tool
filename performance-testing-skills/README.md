# Performance Testing Skills

`performance-testing-skills` is an independent portable package for manual and automated performance testing of model APIs. It does not depend on this repository: copy this directory to any machine or project, keep its internal paths together, and run the scripts from the package root.

The package does not start, stop, deploy, or manage the target service. Bring up the API yourself, then use these scripts to validate request shape, run smoke checks, and measure performance under the exact test conditions you choose.

## Directory Layout

```text
performance-testing-skills/
  skills/      Codex or Claude skill instructions for manual and automated testing
  configs/     Example JSON configs for each supported protocol
  datasets/    Example text JSONL prompts and audio manifest JSONL rows
  scripts/     CLI entry points, shell wrappers, and shared Python helpers
  reports/     Default report output location, usually reports/latest
  tests/       Package structure, dry-run, client, metrics, and local HTTP tests
```

## Supported Protocols

- `openai_chat`: POST to `<base_url>/chat/completions`
- `openai_completion`: POST to `<base_url>/completions`
- `openai_asr`: multipart POST to `<base_url>/audio/transcriptions`
- `generic_http`: direct HTTP request using `method`, `url`, `headers`, and `body_template`

## Quick Start

Run these commands from the package root after copying the package or checking out the repository.

Manual dry-run builds the request and prints curl without network traffic:

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run --input "hello dry run"
```

Manual curl mode prints only the equivalent curl command:

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --mode curl --input "hello curl"
```

Automated dry-run prints the benchmark plan without sending traffic:

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --dry-run --concurrency 1,2 --epochs 2
```

Automated run sends requests and writes reports:

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2 --epochs 2 --output-dir reports/latest
```

The same CLIs can be called directly if you prefer Python entry points:

```bash
python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run
python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run
```

## Configuration

Each config is a JSON object. Relative paths are resolved from the package root.

- `protocol`: one of `openai_chat`, `openai_completion`, `openai_asr`, or `generic_http`.
- `base_url`: base API URL for OpenAI-compatible protocols. The scripts append the endpoint path.
- `model`: model name sent in OpenAI-compatible request bodies and ASR multipart fields.
- `headers`: HTTP headers. Values such as `"Bearer ${API_KEY}"` are expanded from environment variables.
- `request`: optional protocol-specific fields merged into the generated OpenAI-compatible body or multipart form.
- `dataset`: object with `type` and `path`, for example `text_prompts` or `audio_manifest`.
- `body_template`: `generic_http` JSON body template. It is not environment-expanded. Its `${prompt}` and `${max_tokens}` values are sample placeholders filled from the dataset row, not shell environment variables.

Example environment variable usage:

```bash
export API_KEY="..."
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run
```

For `generic_http`, a template like this:

```json
{
  "prompt": "${prompt}",
  "max_tokens": "${max_tokens}"
}
```

uses `prompt` and `expected_output_len` from each dataset sample.

## Datasets

Text datasets are JSONL files. Each line is one sample:

```jsonl
{"prompt": "Write a short introduction to API benchmarking.", "expected_output_len": 128, "metadata": {"case": "intro"}}
```

Common fields:

- `prompt`: text input for chat, completion, or generic HTTP templates.
- `expected_output_len`: used as the default `${max_tokens}` value for `generic_http`.
- `metadata`: optional caller-defined details retained for traceability.

Audio manifest datasets are JSONL files for ASR:

```jsonl
{"audio": "datasets/audio/sample.wav", "prompt": "Transcribe the audio.", "reference": "example transcript", "duration_s": 12.3}
```

Common fields:

- `audio`: audio file path used as the multipart `file` upload.
- `prompt`: optional ASR prompt.
- `reference`: optional expected transcript for manual comparison.
- `duration_s`: optional audio duration recorded into request metrics.

## Manual Testing

Use manual testing before automated load testing to validate endpoint shape, authentication, ASR uploads, and curl reproduction.

Parameters:

- `--config`: config JSON path. Defaults to `configs/openai_chat.json`.
- `--mode request|curl`: `request` sends or dry-runs a request; `curl` prints only curl. Defaults to `request`.
- `--input`: text prompt override for the first dataset sample.
- `--audio-file`: ASR audio path override for the first manifest sample.
- `--request-count`: number of requests to send in request mode. Defaults to `1`.
- `--print-curl`: print curl before sending requests.
- `--save-response`: write response JSON to a file.
- `--dry-run`: construct and print the request without network traffic.

Recommended commands for Codex or Claude:

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run --input "smoke test"
./scripts/run_manual.sh --config configs/openai_chat.json --mode curl --input "smoke test"
./scripts/run_manual.sh --config configs/openai_asr.json --audio-file path/to/audio.wav --print-curl --save-response reports/manual-asr.json
```

## Automated Testing

Use automated testing after a manual smoke succeeds.

Parameters:

- `--config`: config JSON path. Defaults to `configs/openai_chat.json`.
- `--dataset`: dataset path override for the config.
- `--concurrency`: comma or space separated concurrency values, for example `1,2,4` or `1 2 4`.
- `--epochs`: requests per worker for each concurrency value when `--duration-seconds` is not set.
- `--duration-seconds`: optional time-based load duration per concurrency tier.
- `--max-error-rate`: optional threshold from `0` to `1`.
- `--max-p90-latency-ms`: optional p90 latency threshold in milliseconds.
- `--fail-fast`: stop after the first failed concurrency tier.
- `--output-dir`: report directory. Defaults to `reports/latest`.
- `--dry-run`: print the plan without network traffic.

Recommended commands for Codex or Claude:

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --dry-run --concurrency 1,2 --epochs 2
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2 --epochs 2 --max-error-rate 0.05 --max-p90-latency-ms 2000 --fail-fast --output-dir reports/latest
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2,4 --duration-seconds 30 --output-dir reports/sweep
```

Suggested flow:

1. Run manual dry-run and curl checks.
2. Run automated dry-run to verify the benchmark plan.
3. Run a small smoke tier such as `--concurrency 1 --epochs 1`.
4. Run the formal sweep with the planned concurrency, epochs or duration, thresholds, and output directory.

## Reports

Automated runs write these files under `--output-dir`:

- `summary.md`: human-readable conclusion, stable concurrency, peak throughput tier, overload tier, and metric table.
- `metrics.json`: aggregate metrics per concurrency tier.
- `metrics.csv`: the same aggregate metrics in CSV form.
- `requests.jsonl`: one normalized request result per line.
- `errors.jsonl`: only failed request rows.

Read reports in this order:

1. Open `summary.md` for the conclusion and stable/peak/overload tiers.
2. Check `metrics.json` or `metrics.csv` for success rate, error rate, latency percentiles, throughput, and request counts.
3. Inspect `errors.jsonl` for repeated `auth_error`, `bad_request`, `not_found`, `file_not_found`, or timeout patterns.
4. Use `requests.jsonl` when you need per-request latency or response summaries.

A 100% request failure result is not a performance bottleneck. If every request fails, fix the interface first: check `base_url`, endpoint path, credentials, model name, request body, ASR file paths, and target service availability. Do not interpret all-failed throughput or latency as service capacity.

## Notes

- Do not treat authentication, authorization, bad request, missing file, or route errors as performance bottlenecks.
- Always run a manual smoke check before automated load testing.
- The package measures the target you point it at; it does not manage target service startup, shutdown, warmup, deployment, or scaling.
- Do not extrapolate beyond the tested model, hardware, dataset, concurrency, duration, request body, and service configuration.
