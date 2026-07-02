---
name: manual-interface-performance-testing
description: Use when manually validating text, ASR, or generic HTTP model API requests, including smoke tests, small batches, curl generation, and request-shape debugging.
---

# Manual Interface Performance Testing

Use this skill when the user needs to validate request shape, generate curl, test one or a few requests, verify ASR multipart upload, inspect errors, or debug a target API before running automated benchmarks.

Manual checks answer "can this request work?" They do not establish a performance bottleneck or capacity limit.

## Current CLI Modes

The current manual CLI supports only these modes:

- `--mode request`: construct and send a request, or construct and print it with `--dry-run`. This is the default.
- `--mode curl`: print only the equivalent curl command and do not send network traffic.

Do not use older examples such as a smoke mode. For smoke behavior, use `--mode request` with `--request-count 1`, optionally with `--print-curl`.

## Required Workflow

1. Identify the protocol and config: `openai_chat`, `openai_completion`, `openai_asr`, or `generic_http`.
2. Confirm the target service is running and that credentials, `base_url` or `url`, model, and dataset path are intentional.
3. Run `--dry-run` first to inspect method, URL, protocol, and curl without network traffic.
4. Use `--mode curl` when the user wants a reproducible command for another terminal or issue report.
5. Send one request with `--mode request --request-count 1`, adding `--print-curl` if useful.
6. For ASR, pass `--audio-file` when the manifest sample is not the file you want to upload.
7. Save responses with `--save-response` when debugging status codes, response bodies, or error classification.
8. Diagnose errors before running automated load.

## Preferred Commands

Use the wrapper from the package root:

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run --input "smoke test"
./scripts/run_manual.sh --config configs/openai_chat.json --mode curl --input "smoke test"
./scripts/run_manual.sh --config configs/openai_chat.json --mode request --request-count 1 --print-curl --save-response reports/manual-chat.json
./scripts/run_manual.sh --config configs/openai_asr.json --mode request --audio-file path/to/audio.wav --print-curl --save-response reports/manual-asr.json
```

Direct Python entry points are also valid:

```bash
python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run --input "hello"
python3 scripts/perf_manual.py --config configs/openai_chat.json --mode curl --input "hello"
python3 scripts/perf_manual.py --config configs/generic_http.json --mode request --request-count 1 --print-curl
python3 scripts/perf_manual.py --config configs/openai_asr.json --audio-file path/to/audio.wav --save-response reports/manual-asr.json
```

## Output Interpretation

Dry-run output includes:

- `protocol`: selected client behavior.
- `method`: HTTP method.
- `url`: resolved endpoint.
- `curl`: equivalent command for reproduction.

Request output prints one line per request with:

- `request_id`: manual request number.
- `success`: whether the HTTP response was 2xx.
- `status_code`: HTTP status if a response was received.
- `latency_ms`: elapsed request time.
- `response_summary`: extracted text or short response body summary.
- `error_type` and `error_message`: present when the request failed.

When `--save-response PATH` is used, the CLI writes a JSON object with a `requests` array. In `--dry-run` or `--mode curl`, that array is empty because no request was sent.

## Error Diagnosis

- `auth_error`: check headers, tokens, scopes, and environment variable expansion.
- `bad_request`: inspect body shape, model name, request parameters, and ASR multipart fields.
- `not_found`: verify `base_url`, `url`, and endpoint path.
- `file_not_found` or `file_read_error`: verify the ASR `audio` manifest path or `--audio-file`.
- `timeout`, `connection_reset`, or `disconnect`: confirm target service availability before increasing request count.

Do not call a manual failure a performance bottleneck. Fix the interface or service availability first.

## When To Switch To Automated Testing

Switch to the automated performance testing skill only after:

- Dry-run shows the expected request shape.
- Curl mode produces a command the user accepts.
- One real request succeeds, or the user explicitly chooses to benchmark a known failing condition for diagnostics.
- The dataset, concurrency, epochs or duration, thresholds, and output directory are agreed.

If the manual result is 401, 403, 400, 404, missing ASR file, or 100% request failure in a tiny batch, stay in manual diagnosis.
