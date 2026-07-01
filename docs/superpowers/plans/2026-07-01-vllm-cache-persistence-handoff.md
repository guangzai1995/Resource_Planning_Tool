# vLLM Cache Persistence Handoff

## Snapshot

- Date: 2026-07-01
- Worktree: `/Resource_Planning_Tool/.worktrees/vllm-cache-persistence`
- Branch: `feat/vllm-cache-persistence`
- Latest implementation commit: `2c80bfdbb355` (`fix(bench): harden vllm cache persistence`)
- Design spec: `docs/superpowers/specs/2026-07-01-vllm-cache-persistence-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-01-vllm-cache-persistence.md`

## Objective

Persist vLLM compile/JIT cache across `auto_bench.py` service restarts so GLM5.2 benchmark automation can pay the cold compile cost once and reuse cache on later compatible runs.

## Completed Work

Tasks 1-8 are complete and passed the earlier review cycle:

1. Config schema and parsing for `run.vllm_cache` and `serve_profiles[].cache_key`.
2. Invalid config coverage for cache config and key safety.
3. Cache key, cache dir, and env helpers.
4. Cache directory creation during local path validation.
5. vLLM docker run cache mount/env injection while leaving SGLang unchanged.
6. `vllm_cache.json` artifact writing.
7. README documentation and local `.cache/` ignore coverage.
8. Targeted verification, dry-run coverage, whitespace check, and baseline note.

The final read-only review found four required hardening items, all handled in this repair:

- Default cache key fingerprint now uses canonical JSON rather than only `model.name + serve_profile.name + image_ref_hash`.
- `enabled=true` with missing `root` now defaults to `<config_dir>/.cache/vllm_auto_bench`; explicit non-string `root` such as `null` still raises `ConfigError`.
- `container_path` now rejects `/`, `/models`, and `/models/...` in addition to relative paths and `..`.
- `vllm_cache.json` now includes `cache_key_source` and `cache_key_inputs` for default and explicit keys.

## Verification Status

- Red test before implementation: `PYTHONPATH=vllm_standalone_bench pytest -q vllm_standalone_bench/tests/test_auto_bench.py` -> `11 failed, 172 passed`.
- Green targeted test after implementation: `PYTHONPATH=vllm_standalone_bench pytest -q vllm_standalone_bench/tests/test_auto_bench.py` -> `183 passed`.
- Fresh coordinator verification after final review:
  - `PYTHONPATH=vllm_standalone_bench pytest -q vllm_standalone_bench/tests/test_auto_bench.py` -> `183 passed in 0.32s`.
  - `git diff --check` -> exit 0.
  - `git check-ignore -q .cache` -> exit 0.
  - Dry-run with `run.vllm_cache={"enabled": true}` and omitted `root` -> exit 0; vLLM command included `/tmp/.cache/vllm_auto_bench/qwen15b-bf16-default-cache-dry-run:/vllm-cache:rw` plus `VLLM_CACHE_ROOT`, `DG_JIT_CACHE_DIR`, and `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`.
  - Full `pytest -q` -> `11 failed, 306 passed, 1 skipped, 6 warnings`; all failures were in `tests/test_inference_token_factory_report.py` due missing `outputs/context_analysis_20260609_034248/01_overview.json`.
- Final read-only review of `ea9480e58ba6..2c80bfdbb355` found no Critical, Important, or Minor issues and assessed the branch as mergeable.
- The full-suite failure is the known unrelated baseline issue, not introduced by this branch.

## Notes For Future Work

- Default key fingerprints several model/profile inputs, but the image dimension is still only the configured image ref string. It does not inspect the image id or digest behind mutable tags such as `latest` or `offline`.
- Formal GLM5.2 benchmark profiles should still prefer explicit, audited `cache_key` values that include hardware and benchmark policy context.
- `cache_key_inputs` is written even for explicit keys so reviewers can detect accidental reuse across incompatible model/profile inputs.
