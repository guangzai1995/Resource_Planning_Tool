# vLLM Batch Benchmark Per-Config Seed Design

## Problem

`vllm_standalone_bench/run_bench_multi.py` runs a matrix of benchmark configurations by calling the local `vllm_bench/serve.py::main_async()` once per `(input_len, output_len, parallel_num)` group.

`serve.py::main_async()` resets Python and NumPy RNG state from `args.seed` at the start of every invocation. Because the batch runner currently leaves `cfg.seed` unchanged across groups, higher concurrency groups reuse the same initial random prompt sequence as lower concurrency groups.

Example with `epochs=3`:

```text
parallel=1  -> num_prompts=3   -> prompts 0..2
parallel=4  -> num_prompts=12  -> prompts 0..11
parallel=8  -> num_prompts=24  -> prompts 0..23
```

When vLLM prefix caching is enabled and the same server remains running across groups, this makes higher concurrency rows partly reuse prompts or shared prefixes warmed by earlier rows. TTFT then reflects both concurrency pressure and cross-row cache state, which makes mean TTFT harder to interpret.

## Goal

Each benchmark configuration should use a different random seed by default, while keeping runs reproducible for the same configuration.

The design should:

- Prevent higher parallelism rows from reusing lower parallelism rows' prompt sequence.
- Keep repeated runs of the same benchmark matrix reproducible.
- Make the actual seed visible in logs and CSV/XLSX output.
- Preserve a compatibility mode for reproducing old results with one fixed seed across all groups.

## Non-Goals

- Do not change prefix length semantics in this work. The separate fixed-total-input prefix design covers that change.
- Do not restart or clear vLLM server-side prefix cache between rows.
- Do not change TTFT, TPOT, or E2EL timing formulas.
- Do not change request scheduling, concurrency, or `epochs` semantics.

## User-Facing Behavior

Add two batch runner options:

```text
--seed 0
--no-vary-seed-by-config
```

Default behavior:

```text
seed = 0
vary_seed_by_config = true
```

With default behavior, every benchmark group gets a derived seed. With `--no-vary-seed-by-config`, every group uses the base `--seed` value exactly, matching the old behavior.

`run_bench.sh` should expose matching shell variables:

```bash
SEED=0
VARY_SEED_BY_CONFIG=true
```

When `VARY_SEED_BY_CONFIG=false`, the shell wrapper appends `--no-vary-seed-by-config`.

## Seed Derivation

The batch runner should compute the effective seed before calling `_serve.main_async(cfg)`.

The seed input key is:

```text
base_seed
input_len
output_len
parallel_num
prefix_ratio
config_index
```

Use a stable hash, not Python's built-in `hash()`, because Python hash randomization would make values process-dependent.

Recommended derivation:

```text
key = f"{base_seed}:{input_len}:{output_len}:{parallel_num}:{prefix_ratio:.12g}:{config_index}"
derived_seed = int(sha256(key).hexdigest()[:8], 16)
```

The resulting seed is a 32-bit unsigned integer suitable for both `random.seed()` and `np.random.seed()`.

If compatibility mode is enabled:

```text
effective_seed = base_seed
```

Otherwise:

```text
effective_seed = derived_seed
```

Then set:

```text
cfg.seed = effective_seed
```

before invoking:

```text
_serve.main_async(cfg)
```

## Reporting

Add a `seed` column to CSV/XLSX output. This column records the effective seed used for that result row, not merely the base seed.

The row schema should keep existing metric columns unchanged and insert `seed` near the benchmark configuration columns:

```text
model, backend, input_len, output_len, total_input_len, prefix_ratio,
prefix_tokens, parallel_num, epochs, num_prompts, seed, ...
```

The per-group log line should include the effective seed:

```text
input=4096, output=1024, parallel=8, seed=123456789
```

This makes abnormal rows reproducible by running the same configuration with compatibility mode and the recorded seed.

## Compatibility And Reproduction

Old behavior can be reproduced with:

```bash
python3 run_bench_multi.py ... --seed 0 --no-vary-seed-by-config
```

To reproduce a single abnormal row from a CSV:

```bash
python3 run_bench_multi.py \
  --input-lens <row_input_len> \
  --output-lens <row_output_len> \
  --parallel-nums <row_parallel_num> \
  --epochs <row_epochs> \
  --seed <row_seed> \
  --no-vary-seed-by-config \
  ...
```

This produces the same prompt sequence for that single configuration, assuming tokenizer, model, prefix settings, and other benchmark arguments are unchanged.

## Error Handling

Validate `--seed` as a non-negative integer within NumPy's accepted seed range:

```text
0 <= seed < 2**32
```

If an invalid seed is supplied, fail before starting the benchmark matrix with a clear error message.

`prefix_ratio` should be normalized consistently when building the hash key. Formatting with `:.12g` avoids differences from incidental string representation while preserving enough precision for the configured ratio.

## Tests

Add focused tests around the batch runner, using the existing `main_async` monkeypatch pattern:

1. Different parallel values receive different `cfg.seed` values by default.
2. Repeating the same matrix produces the same derived seeds.
3. `--no-vary-seed-by-config` makes every row use the base seed.
4. CSV output includes the effective `seed` column.
5. Existing token accounting and output compliance tests continue to pass.

No live vLLM server is required for unit tests.

## Acceptance Criteria

- Default batch runs no longer reuse the same prompt sequence across different benchmark groups.
- The effective seed for each result row is visible in logs and persisted in CSV/XLSX.
- A user can reproduce old fixed-seed behavior explicitly.
- A user can reproduce a single result row by passing that row's recorded seed with `--no-vary-seed-by-config`.
- The existing test suite passes after implementation.
