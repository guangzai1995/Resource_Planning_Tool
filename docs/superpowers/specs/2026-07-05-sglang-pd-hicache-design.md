# SGLang PD + HiCache Support Design

## Status

Reviewed in conversation and ready for implementation planning after user review.

## Context

The benchmark tool already supports remote Docker PD topologies through
`topology_profiles`. For SGLang, the current command renderer can start
prefill workers, decode workers, and `sglang_router.launch_router` with
`--pd-disaggregation`. Recent work added explicit RDMA device support for PD
and added `mount_infiniband` for vLLM NIXL workers.

The next goal is first-class SGLang PD + HiCache support. The purpose is not
to invent a new serving architecture. The implementation should render and
validate SGLang-supported options based on the local source tree under
`sglang-main`, then provide runnable remote benchmark configs that follow the
same topology style as the existing vLLM PD configs.

## Source Findings

The design is based on local source, not inferred from external examples:

- `sglang-main/python/sglang/srt/server_args.py` defines the HiCache CLI
  arguments: `--enable-hierarchical-cache`, `--hicache-ratio`,
  `--hicache-size`, `--hicache-write-policy`, `--hicache-io-backend`,
  `--hicache-mem-layout`, `--hicache-storage-backend`,
  `--hicache-storage-prefetch-policy`, and
  `--hicache-storage-backend-extra-config`.
- The same file defines PD flags:
  `--disaggregation-transfer-backend`, `--disaggregation-bootstrap-port`,
  `--disaggregation-ib-device`, and
  `--disaggregation-decode-enable-offload-kvcache`.
- `--disaggregation-decode-enable-offload-kvcache` is only valid on decode
  workers and requires `--hicache-storage-backend`.
- SGLang normalizes incompatible HiCache layout and I/O combinations. In
  particular, Mooncake storage does not support `layer_first`; for Mooncake
  with direct I/O, the effective layout should be `page_first_direct`.
- `sglang-main/python/sglang/srt/environ.py` defines Mooncake and HiCache
  environment variables such as `MOONCAKE_MASTER`,
  `MOONCAKE_TE_META_DATA_SERVER`, `MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`,
  `MOONCAKE_GLOBAL_SEGMENT_SIZE`, `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH`, and
  `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE`.
- `sglang-main/python/sglang/srt/mem_cache/storage/mooncake_store/README.md`
  describes Mooncake master, metadata, and store services as deployment
  prerequisites. The benchmark tool should connect to them through explicit
  configuration; it should not start or manage the Mooncake control plane.
- SGLang exposes cache hit information through OpenAI usage when
  `--enable-cache-report` is enabled, and through Prometheus counter
  `sglang:cached_tokens_total` with `cache_source` labels.
- `sglang_router.launch_router` accepts PD prefill entries in the form
  `--prefill URL [BOOTSTRAP_PORT]`, and the existing benchmark renderer
  already emits bootstrap ports for SGLang prefill workers.

## Goals

- Add a structured `sglang_hicache` configuration object for SGLang PD
  topology profiles.
- Render SGLang HiCache flags from this structured object instead of forcing
  users to hand-maintain long `args` lists.
- Support two deployment modes that SGLang source and docs align with:
  prefill-only HiCache and full async decode KV offload.
- Reuse existing `transfer_backend`, `disaggregation_ib_device`,
  `mount_infiniband`, `env`, `volumes`, and node-level `args` patterns.
- Extend `mount_infiniband` to SGLang worker containers, matching the vLLM
  NIXL behavior: mount `/dev/infiniband`, add `IPC_LOCK`, and set unlimited
  memlock. The router does not need IB devices.
- Add validation that catches invalid SGLang HiCache combinations before
  remote containers are started.
- Preserve the current escape hatch: users can still append arbitrary SGLang
  flags through node `args`.
- Add example configs for SGLang PD + HiCache + Mooncake that follow the
  existing remote topology style.
- Improve observability for cache reuse by preserving existing
  `avg_cached_tokens` and `cache_hit_rate` and adding optional SGLang
  Prometheus cache-source deltas.

## Non-Goals

- No automatic RDMA NIC, IB device, Mooncake master, metadata server, store
  service, or storage-size discovery.
- No orchestration of Mooncake master, metadata server, or store service.
- No Kubernetes, Helm, service discovery, or dynamic worker discovery.
- No attempt to abstract every SGLang storage backend into a complete storage
  platform. The implementation validates known SGLang choices and provides a
  Mooncake example, while keeping `storage_backend_extra_config` and `env`
  explicit.
- No cross-host distributed worker group management beyond the existing
  topology model. A `TopologyNode` remains one Docker container.
- No replacement for `node.args`; structured config covers the common and
  benchmark-critical HiCache settings only.

## Configuration Design

`topology_profiles[]` gains an optional `sglang_hicache` object. It is valid
only when `engine == "sglang"` and `mode == "pd"`.

Example:

```json
{
  "name": "sglang_pd_hicache_mooncake_2p2d",
  "engine": "sglang",
  "mode": "pd",
  "provider": "ssh_docker",
  "transfer_backend": "mooncake",
  "disaggregation_ib_device": "mlx5_0",
  "mount_infiniband": true,
  "sglang_hicache": {
    "mode": "full_async_offload",
    "page_size": 64,
    "ratio": 2.0,
    "size": 0,
    "write_policy": "write_through",
    "io_backend": "direct",
    "mem_layout": "page_first_direct",
    "storage_backend": "mooncake",
    "storage_prefetch_policy": "timeout",
    "storage_backend_extra_config": {"tp_lcm_size": 4},
    "enable_metrics": true,
    "enable_cache_report": true
  },
  "env": {
    "MOONCAKE_MASTER": "10.200.1.10:50051",
    "MOONCAKE_TE_META_DATA_SERVER": "http://10.200.1.10:8080/metadata",
    "MOONCAKE_PROTOCOL": "rdma",
    "MOONCAKE_DEVICE": "mlx5_0",
    "MOONCAKE_GLOBAL_SEGMENT_SIZE": "64gb"
  }
}
```

### Supported Fields

- `mode`: required string. Allowed values:
  - `prefill_only`: render HiCache flags only on prefill workers.
  - `full_async_offload`: render HiCache/storage flags on prefill workers and
    decode workers; add `--disaggregation-decode-enable-offload-kvcache` only
    on decode workers.
- `page_size`: optional positive integer. Renders `--page-size`.
- `ratio`: optional positive float. Renders `--hicache-ratio`.
- `size`: optional non-negative integer. Renders `--hicache-size`.
- `write_policy`: optional string. Allowed SGLang choices are `write_back`,
  `write_through`, and `write_through_selective`.
- `io_backend`: optional string. Allowed SGLang choices are `direct`,
  `kernel`, and `kernel_ascend`.
- `mem_layout`: optional string. Allowed SGLang choices are `layer_first`,
  `page_first`, `page_first_direct`, `page_first_kv_split`, and `page_head`.
- `storage_backend`: optional string. Allowed SGLang choices are `file`,
  `mooncake`, `hf3fs`, `nixl`, `aibrix`, `dynamic`, `eic`, and `simm`.
- `storage_prefetch_policy`: optional string. Allowed SGLang choices are
  `best_effort`, `wait_complete`, and `timeout`.
- `storage_backend_extra_config`: optional object or string. Objects are JSON
  encoded when rendering `--hicache-storage-backend-extra-config`. Strings are
  passed through, including SGLang-supported `@path` values.
- `enable_metrics`: optional boolean. If true, render `--enable-metrics`.
- `enable_cache_report`: optional boolean. If true, render
  `--enable-cache-report`.

Defaults are chosen to be explicit for Mooncake benchmarking:

```text
page_size = 64
ratio = 2.0
size = 0
write_policy = write_through
io_backend = direct
mem_layout = page_first_direct
storage_prefetch_policy = timeout
enable_metrics = true
enable_cache_report = true
```

`storage_backend` has no implicit default. A profile that uses
`full_async_offload` must set it. A `prefill_only` profile may omit it if the
intent is host-memory-only HiCache, but Mooncake examples should set
`storage_backend: "mooncake"`.

## Command Rendering

SGLang worker rendering stays in `remote_topology.py`:

1. Build the existing Docker command base.
2. Add GPU, network, model mount, explicit env, and volumes as today.
3. If `mount_infiniband` is true and the role is a worker, append:
   - `--device /dev/infiniband`
   - `--cap-add IPC_LOCK`
   - `--ulimit memlock=-1:-1`
4. Render the existing SGLang launch flags:
   - `python3 -m sglang.launch_server`
   - `--model-path`
   - `--served-model-name`
   - `--host 0.0.0.0`
   - `--port`
   - `--disaggregation-mode prefill|decode`
   - `--disaggregation-transfer-backend` when configured
   - `--disaggregation-bootstrap-port` when configured
   - `--disaggregation-ib-device` when configured
5. Append structured HiCache flags before node `args`, so node-specific
   `args` can intentionally override or extend behavior.

For `prefill_only`:

- Prefill workers get `--enable-hierarchical-cache` and the configured
  `--hicache-*` flags.
- Decode workers do not get HiCache/offload flags from `sglang_hicache`.

For `full_async_offload`:

- Prefill workers get `--enable-hierarchical-cache` and the configured
  `--hicache-*` flags.
- Decode workers get the same storage-related `--hicache-*` flags and
  `--disaggregation-decode-enable-offload-kvcache`.
- Decode workers do not get `--enable-hierarchical-cache` unless SGLang source
  requires it later. Current source normalizes HiCache settings when either
  hierarchical cache or decode offload is active, and decode offload is guarded
  by `disaggregation_decode_enable_offload_kvcache`.

The router command remains the current SGLang router path. Any router policy
settings such as `--prefill-policy cache_aware` and
`--decode-policy power_of_two` stay in `frontend.args`, because those are
router-level choices rather than HiCache worker flags.

## Validation Rules

The config parser should fail early for:

- `sglang_hicache` used when `engine` is not `sglang`.
- `sglang_hicache` used when `mode` is not `pd`.
- Unknown keys inside `sglang_hicache`.
- Invalid enum values for write policy, I/O backend, memory layout, storage
  backend, or storage prefetch policy.
- `full_async_offload` without `storage_backend`.
- `full_async_offload` when no decode workers exist.
- `storage_backend_extra_config` that is neither a mapping nor a string.
- Non-positive `page_size` or `ratio`.
- Negative `size`.

The parser should not reject explicit node `args` that contain the same flags.
The intended precedence is: structured flags first, node `args` last.

## Observability

The existing benchmark result extraction already reports:

- `avg_cached_tokens`
- `cache_hit_rate`
- `avg_gpu_kv_cache_usage`
- `peak_gpu_kv_cache_usage`

For SGLang HiCache, add optional Prometheus cache-source deltas when `/metrics`
is available:

- `cache_hit_rate_metrics`: total `sglang:cached_tokens_total` delta divided
  by `sglang:prompt_tokens_total` delta.
- `cache_hit_tokens_device`
- `cache_hit_tokens_host`
- `cache_hit_tokens_storage`
- `cache_hit_tokens_storage_mooncake` or a generic source-derived column if
  the label is `storage_<backend>`.

Implementation detail: this can extend the existing runtime metrics parser to
sum `sglang:cached_tokens_total{cache_source=...}` and
`sglang:prompt_tokens_total`. If metrics are unavailable, these fields should
remain empty or zero using the existing result-row conventions.

The example configs should enable both `--enable-metrics` and
`--enable-cache-report` by default so response usage and Prometheus metrics can
both confirm whether HiCache is doing useful work.

## Example Configs

Add at least one dry-run-safe example:

- `vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote.example.json`

Add one cluster-specific template mirroring the existing Minimax host layout,
but keep model path, Mooncake control-plane addresses, and RDMA devices
explicit:

- `vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote_minimax.json`

The cluster template should prefer the topology that made sense in previous
PD analysis:

- 2P2D as the primary profile.
- Avoid presenting 1P3D as a recommended profile because a single prefill pool
  is a bottleneck for long prompts.
- Include a prefill-only profile only if it is useful for isolating shared
  prefix reuse from decode offload.

Benchmark profiles should use higher prefix reuse than the earlier PD transfer
test when validating HiCache, for example `prefix_ratio` around `0.6` or
`0.8`. A low-prefix profile may still be included as a negative control.

## Tests

Add or update tests in `vllm_standalone_bench/tests/test_remote_topology.py`:

- SGLang HiCache `prefill_only` renders HiCache flags only on prefill workers.
- SGLang HiCache `full_async_offload` renders decode offload only on decode
  workers and renders `storage_backend`.
- `mount_infiniband` adds IB Docker flags for SGLang workers and not router.
- Invalid enum values raise `ConfigError`.
- `full_async_offload` without `storage_backend` raises `ConfigError`.
- Existing SGLang PD and vLLM PD tests continue to pass.

Add or update result extraction tests if Prometheus cache-source deltas are
implemented:

- Parser sums `sglang:cached_tokens_total` by `cache_source`.
- Metrics-derived cache hit rate uses before/after deltas.
- Missing metrics remain backward compatible.

Add config-load tests:

- The new example config loads.
- Dry-run command rendering includes expected HiCache and IB flags.

## Rollout

1. Implement the structured config parser and command renderer.
2. Add topology tests and example config load tests.
3. Extend SGLang worker `mount_infiniband`.
4. Add cache-source metrics extraction while keeping existing result rows
   backward compatible when metrics are unavailable.
5. Add README notes that Mooncake master/store services are prerequisites and
   are not managed by `auto_bench`.
6. Run focused tests first, then the broader benchmark test subset used by this
   package.

## Implementation Choices

- The first implementation includes metrics-derived cache-source columns.
  Missing `/metrics` data must not fail the benchmark or break old result
  parsing.
- The Minimax example includes one recommended 2P2D full-async-offload profile
  and one prefill-only isolation profile. It does not include 1P3D as a
  recommended profile.
