# vLLM Standalone Bench Prefix Ratio Total Input Design

## Status

This specification is ready for user review.

## Context

`vllm_standalone_bench/` supports random benchmark workloads with optional shared prefixes for prefix cache testing.

Current behavior treats `prefix_ratio` as extra input added on top of `input_len`:

```text
input_len=512, prefix_ratio=0.5
old prompt ~= 256 shared prefix + 512 unique suffix = 768 input tokens
```

That makes benchmark rows with different `prefix_ratio` values incomparable because total prompt length changes with the ratio. The intended benchmark question is different: for a fixed total input length, how much of that input is shared and therefore eligible for prefix cache reuse?

The requested design is to make `input_len` the total prompt token budget and make `prefix_ratio` the portion of that budget used by the shared prefix.

## Goals

- Keep `--input-lens` as the requested total input length.
- Define `prefix_ratio` as the shared-prefix proportion of that total input length.
- Keep each request in a benchmark group close to the same total `input_len`, regardless of prefix ratio.
- Preserve the existing batch runner model: each `(input_len, output_len, parallel_num, prefix_ratio)` configuration generates its own request set and result row.
- Keep CSV/XLSX output clear enough that old "prefix as extra input" semantics cannot be mistaken for the new behavior.
- Update tests so future changes cannot reintroduce extra-prefix semantics.

## Non-Goals

- Do not change vLLM server configuration, prefix cache enablement flags, or container orchestration.
- Do not add a new compatibility mode for the old semantics.
- Do not migrate historical CSV/XLSX files.
- Do not add prefix cache hit-rate scraping from server metrics in this change.

## Current Code Points

| File | Current role |
|---|---|
| `vllm_standalone_bench/run_bench_multi.py` | Parses `--prefix-ratio`, builds each benchmark config, extracts CSV/XLSX rows. |
| `vllm_standalone_bench/run_bench_serve.py` | Generates random requests through `_generate_random_requests()`. |
| `vllm_standalone_bench/vllm_bench/serve.py` | Maps generic `args.input_len` to `args.random_input_len` before calling `get_samples()`. |
| `vllm_standalone_bench/tests/test_random_dataset.py` | Currently locks old `prefix + input` prompt length behavior. |
| `vllm_standalone_bench/tests/test_extract_row.py` | Currently locks old `total_input_len = input_len + prefix_tokens` report behavior. |

## Chosen Approach

Use the direct semantic correction:

```text
total_input_len = input_len
prefix_tokens = int(input_len * prefix_ratio)
suffix_tokens = input_len - prefix_tokens
prompt_len ~= prefix_tokens + suffix_tokens ~= input_len
```

This is preferred over adding a mode flag because it keeps the CLI simple and aligns with how benchmark users normally compare prefix cache performance: same total prompt size, different shared-prefix ratios.

## Parameter Semantics

`--input-lens` means total prompt length in requested tokens.

`--prefix-ratio` must be in `[0.0, 1.0]`.

`prefix_tokens` is computed with the existing integer style:

```text
prefix_tokens = int(input_len * prefix_ratio)
suffix_tokens = input_len - prefix_tokens
```

Examples:

| input_len | prefix_ratio | prefix_tokens | suffix_tokens | total prompt |
|---:|---:|---:|---:|---:|
| 512 | 0.0 | 0 | 512 | 512 |
| 512 | 0.5 | 256 | 256 | 512 |
| 512 | 0.8 | 409 | 103 | 512 |
| 512 | 1.0 | 512 | 0 | 512 |

`prefix_ratio=1.0` is valid. It intentionally creates identical prompts within a group and can be used as the extreme prefix-cache case.

## Data Flow

### Batch runner

For each benchmark group, `run_bench_multi.py` should derive prefix and suffix lengths from the total input length:

```text
(input_len, output_len, parallel_num, prefix_ratio)
  -> prefix_tokens = int(input_len * prefix_ratio)
  -> suffix_tokens = input_len - prefix_tokens
  -> cfg.input_len = input_len
  -> cfg.random_prefix_len = prefix_tokens
```

The group still uses the existing request count rule:

```text
num_prompts = parallel_num * epochs
```

`parallel_num` affects only maximum in-flight requests. It does not change prompt composition.

`epochs` affects only the number of requests generated for that configuration. It does not change prompt composition.

Each group should generate one shared prefix and `num_prompts` request-specific suffixes. Different groups should generate their own shared prefix so results do not rely on cache state from a previous group.

### Random request generation

`run_bench_serve.py::_generate_random_requests()` should interpret:

```text
random_input_len = total requested input length
random_prefix_len = shared-prefix length within that total
suffix_len = random_input_len - random_prefix_len
```

Generated prompt:

```text
prompt = shared_prefix + unique_suffix
prompt_len ~= random_input_len
```

With a tokenizer, keep the current decode/encode correction path but target `random_input_len`, not `random_prefix_len + random_input_len`.

Without a tokenizer, the generated whitespace-separated pseudo-token text should contain exactly `random_input_len` tokens.

## Report Semantics

For new results:

```text
input_len = requested total input length
total_input_len = requested total input length
prefix_tokens = shared-prefix tokens inside total_input_len
prefix_ratio = configured shared-prefix ratio
```

`input_compliance` should compare measured average prompt tokens against `input_len`, not against `input_len + prefix_tokens`:

```text
input_compliance = avg_input_tokens / input_len * 100
```

The existing `total_input_len` column may remain for compatibility with existing sheets, but in the new semantics it equals `input_len`. This is intentional and should be documented in README/help text.

The implementation may add a `suffix_tokens` CSV column if useful, but it is not required. If it is not added, logs should still print suffix length clearly for each group.

## Logging And Help Text

CLI help for `--prefix-ratio` should state that the ratio is part of `input_len`, not extra input.

Current text like this is wrong and must be removed:

```text
actual prompt_len ~= input_len * (1 + prefix_ratio)
```

Recommended log wording for a group:

```text
input=512, output=128, parallel=8, prefix=256, suffix=256, total_input=512
```

`run_bench.sh` comments should be updated similarly. For example:

```text
PREFIX_RATIO=0.5 -> input_len=512 means prefix=256 + suffix=256
```

## Error Handling And Boundaries

Validate `prefix_ratio` before running the matrix:

- `< 0.0` is invalid.
- `> 1.0` is invalid.
- invalid values should fail fast with a clear parser or runtime error.

Length boundaries:

- `prefix_ratio=0.0`: no shared prefix, suffix length is the full `input_len`.
- `prefix_ratio=1.0`: suffix length is zero, requests in the same group may be identical.
- small `input_len` values use `int(input_len * prefix_ratio)` consistently, even when that rounds the prefix down to zero.

Tokenizer correction:

- If decode/encode cannot hit the exact target length, return the actual encoded length as today.
- The existing `input_compliance` metric is responsible for surfacing any mismatch.
- Do not silently report compliance against the old longer target.

## Testing Plan

Use TDD in the implementation step.

Update `tests/test_random_dataset.py`:

- Replace the old expectation that `random_input_len=8` and `random_prefix_len=3` yields prompt length `11`.
- New expectation: prompt length is `8`.
- Confirm the two requests still differ when suffix length is nonzero.
- Add a `prefix_ratio=1.0` equivalent case through direct generator arguments: `random_input_len=8`, `random_prefix_len=8`, expected prompt length `8`; identical prompts are acceptable.

Update `tests/test_extract_row.py`:

- Replace the old expectation that `total_input_len == input_len + prefix_tokens`.
- New expectation: `total_input_len == input_len`.
- For `input_len=128`, `prefix_tokens=102`, `total_in=384`, `completed=3`, expected `avg_input_tokens=128.0` and `input_compliance=100.0`.

Add or update `run_bench_multi.py` tests:

- `input_len=128`, `prefix_ratio=0.8` derives `prefix_tokens=102` and `suffix_tokens=26`.
- `prefix_ratio=0.0` derives `prefix_tokens=0`, `suffix_tokens=input_len`.
- `prefix_ratio=1.0` derives `prefix_tokens=input_len`, `suffix_tokens=0`.
- invalid ratios `-0.1` and `1.1` fail before benchmark execution.

Regression test scope:

- Run the existing `vllm_standalone_bench/tests` suite after implementation.
- Ensure endpoint parsing, metrics, integration, CSV headers, and random dataset tests still pass.

## Acceptance Criteria

1. `--input-lens 512 --prefix-ratio 0.5` generates prompts whose measured input length is about `512`, not about `768`.
2. The batch runner keeps prompt length fixed for every `prefix_ratio` at the same `input_len`.
3. CSV/XLSX rows show `input_len == total_input_len` under the new semantics.
4. `prefix_tokens` still records the shared-prefix portion of the prompt.
5. `input_compliance` is calculated against total input length.
6. Help text, shell wrapper comments, and logs no longer describe prefix tokens as extra input.
7. Tests that previously locked the old extra-prefix behavior are updated to lock the new fixed-total behavior.
