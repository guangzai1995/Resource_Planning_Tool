# vLLM Standalone Bench Token Stats Design

Date: 2026-06-29

## Context

This design covers only `vllm_standalone_bench`. The existing
`benchmark_tools` implementation is used as a reference for token accounting
ideas, but it is not part of this change.

The current standalone benchmark already improves on upstream vLLM 0.23.0 in
some places by reading streaming `usage`, tracking `finish_reason`, and keeping
CSV aggregation from falling back to requested lengths when all requests fail.
There are still gaps that can make benchmark results misleading:

- Prefix-cache runs can show `avg_input_tokens` as the suffix length even when
  the actual prompt includes a shared prefix.
- Random prompt generation decodes random token ids and then re-tokenizes the
  text, but it does not actively correct drift toward the target token length.
- Result rows do not preserve enough source information to distinguish target
  lengths, local tokenizer measurements, and server-reported usage.
- When `usage` is missing, fallback token counts need to be marked as fallback
  data instead of being treated as equally authoritative.

## Goals

- Make actual input and output token statistics trustworthy in
  `run_bench_multi.py` CSV/XLSX output.
- Preserve the requested benchmark configuration separately from measured token
  lengths.
- Use the local tokenizer to construct and validate request lengths.
- Use API `usage` as the primary source for final measured statistics when it
  is available.
- Add regression tests that lock the intended accounting behavior.

## Non-Goals

- Do not modify `benchmark_tools`.
- Do not import the full vLLM package or add a hard dependency on torch.
- Do not implement every dataset type from upstream vLLM.
- Do not change benchmark scheduling semantics beyond token accounting fixes.

## Token Source Policy

Token accounting uses two different priorities depending on the phase.

During request construction, the local tokenizer is the preferred source. It is
used to generate random prompts, validate decoded prompt lengths, and estimate
prefix plus suffix lengths before requests are sent.

During result aggregation, server API `usage` is the preferred source. For
OpenAI-compatible chat endpoints, server-side chat templates, role tokens,
special tokens, and reasoning tokens can make local re-tokenization differ from
the actual server accounting. Therefore:

1. Use `usage.prompt_tokens` and `usage.completion_tokens` for final measured
   input and output totals when all successful requests report them.
2. Use the local tokenizer as fallback when usage is missing.
3. Use client-side prompt lengths as a last-resort input estimate only when no
   server prompt usage is available.
4. Never use requested lengths as measured averages.

Each result row must expose the source used for measured token fields so users
can identify whether the row is authoritative or fallback-based.

## Data Model

The CSV/XLSX result row should distinguish these concepts:

- `input_len`: requested unique suffix input length, kept for compatibility.
- `prefix_tokens`: requested shared prefix length.
- `total_input_len`: requested total input target, `input_len + prefix_tokens`.
- `avg_input_tokens`: measured average input tokens among successful requests.
- `avg_output_tokens`: measured average output tokens among successful requests.
- `input_compliance`: measured input average divided by `total_input_len`.
- `output_compliance`: measured output average divided by requested
  `output_len`.
- `token_source`: source of final measured token statistics, such as `usage`,
  `tokenizer_fallback`, `client_estimate`, or `none`.
- `finish_reason_length_pct`: percentage of successful requests whose
  `finish_reason` is `length`.

For `completed == 0`, measured averages and compliance fields are `0.0`, and
`token_source` is `none`.

## Request Generation

The random dataset shim in `run_bench_serve.py` should be tightened without
copying the entire upstream dataset framework.

When a tokenizer is available:

- Generate shared prefix token ids once per run.
- Generate per-request suffix token ids independently.
- Decode the combined ids to text.
- Re-encode the decoded text with `add_special_tokens=False`.
- If the length differs from the target, truncate or append allowed random
  token ids and retry a bounded number of times.
- Store the final re-encoded length in `SampleRequest.prompt_len`.

For prefix-cache tests, the target input length is `prefix_tokens + input_len`.
The requested suffix length remains visible as `input_len`, but measured input
statistics must use the full prompt length.

When no tokenizer is available, preserve the current approximate string
generation behavior and mark result rows as client-estimated rather than
usage-authoritative unless the server reports `usage`.

## Result Aggregation

`vllm_bench/serve.py` should continue to parse streaming usage and
`finish_reason`. `run_bench_multi.py` should aggregate totals from
`main_async()` result dictionaries rather than per-request detailed arrays,
because detailed arrays can be removed when `save_detailed` is false.

Aggregation rules:

- `avg_input_tokens = total_input_tokens / completed` when `completed > 0`.
- `avg_output_tokens = total_output_tokens / completed` when `completed > 0`.
- Compliance calculations use unrounded averages.
- `output_compliance` should be used for skip decisions.
- `input_compliance` should be reported as a diagnostic, not as a default skip
  decision.
- Rows with missing usage but available tokenizer fallback remain usable, but
  their source must indicate fallback.

## Error Handling

- If all requests fail, do not fabricate requested token averages.
- If usage is partially reported for successful requests, mark the source as
  fallback or partial rather than `usage`.
- If local tokenizer loading fails, continue only when the CLI explicitly allows
  tokenizer-free operation, and mark rows as estimated unless API usage is
  complete.
- If generated prompt length cannot be corrected exactly after bounded retries,
  keep the actual re-encoded length and expose it through measured input
  statistics.

## Tests

Add or extend tests under `vllm_standalone_bench/tests`:

- Random prefix generation records full prompt length, not suffix-only length.
- `_extract_row` computes averages from totals divided by completed requests.
- `_extract_row` reports `input_compliance` and `output_compliance` from
  unrounded measured averages.
- `completed == 0` produces zero measured averages and `token_source=none`.
- Usage-present rows choose `token_source=usage`.
- Usage-missing rows with tokenizer choose `token_source=tokenizer_fallback`.
- Usage-missing rows without tokenizer choose `client_estimate` or `none`.
- CSV and XLSX headers include any new fields in both English and Chinese
  header rows.
- Existing SSE parser tests for usage, `finish_reason`, empty chat frames, and
  reasoning deltas remain passing.

## Success Criteria

- Existing `vllm_standalone_bench/tests` pass.
- New tests fail before the implementation and pass after it.
- Prefix-cache benchmark rows report `avg_input_tokens` near
  `input_len + prefix_tokens` when no server prompt usage changes that count.
- Output length compliance reflects actual measured output length, not requested
  length.
- Result files make it clear whether measured token counts came from server
  usage, local tokenizer fallback, or client estimates.
