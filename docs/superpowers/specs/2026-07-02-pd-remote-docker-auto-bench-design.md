# PD Remote Docker Auto Bench Design

## Status

Ready for user review.

## Context

`vllm_standalone_bench/auto_bench.py` currently orchestrates one serving
container on the local Docker host, runs one local bench-runner container
against that serving container, records results under `results/<run_id>/`,
and then cleans up containers and Docker network resources owned by the run.
The existing implementation already supports vLLM and SGLang single-service
profiles through `serve_profiles[].engine`.

The next capability is automated prefill/decode (PD) disaggregated benchmark
for cross-host deployments. The user-approved target is remote Docker
orchestration:

- The controller still runs on the local control host.
- The bench-runner still runs locally on the control host.
- Remote hosts are reached through SSH.
- The SSH user can run `docker` directly without `sudo`.
- SSH key authentication is supported, and password authentication is allowed
  through a safe configuration path.
- All service IPs, ports, model mounts, GPU selections, and transfer settings
  are configured explicitly. The tool does not auto-detect NICs or IPs.
- The first implementation supports static multi-prefill and multi-decode
  topology lists. It does not orchestrate a single prefill or decode role that
  is itself split across multiple distributed nodes.

The benchmark client can continue to treat the target as a single
OpenAI-compatible endpoint. For PD deployments, that endpoint is provided by a
frontend role:

- SGLang: `sglang_router.launch_router --pd-disaggregation`.
- vLLM: a configured disaggregated proxy, usually supplied as a proxy image or
  command by the user.

## Source Findings

SGLang has first-class PD launch flags in
`sglang-main/python/sglang/srt/server_args.py`:

- `--disaggregation-mode null|prefill|decode`
- `--disaggregation-transfer-backend`
- `--disaggregation-bootstrap-port`
- `--disaggregation-ib-device`

SGLang documentation shows the intended pattern:

```text
prefill workers:
  python -m sglang.launch_server --disaggregation-mode prefill ...

decode workers:
  python -m sglang.launch_server --disaggregation-mode decode ...

frontend:
  python -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://<prefill-host>:<port> \
    --decode http://<decode-host>:<port>
```

vLLM exposes PD through KV transfer configuration. `KVTransferConfig` supports:

- `kv_connector`
- `kv_role=kv_producer|kv_consumer|kv_both`
- `kv_rank`
- `kv_parallel_size`
- `kv_ip`
- `kv_port`
- `kv_connector_extra_config`

vLLM examples use two vLLM instances plus a proxy. The prefill instance is a
KV producer, the decode instance is a KV consumer, and a proxy exposes the
OpenAI-compatible endpoint used by clients.

## Goals

- Add a remote Docker topology mode for PD benchmark cases.
- Keep all existing single-host, single-service configs working unchanged.
- Make SGLang PD a first-class supported topology in the first version.
- Support static multi-P and multi-D SGLang topologies.
- Support vLLM PD command generation and execution through explicit proxy and
  KV transfer configuration, without trying to infer every connector-specific
  setting.
- Run the bench-runner on the local control host against the remote frontend
  endpoint.
- Store all PD role commands, logs, inspect output, resource summaries, and
  benchmark results under the existing `results/<run_id>/` tree.
- Preserve current `run`, `status`, `logs`, `stop`, and `resume` semantics.
- Ensure dry-run output and resolved config do not leak SSH passwords.

## Non-Goals

- No Kubernetes, LWS, Helm, or service-discovery orchestration in the first
  version.
- No automatic IP, NIC, RDMA device, or port discovery.
- No `sudo docker` or privilege escalation support.
- No automatic image transfer, model synchronization, or dependency
  installation on remote hosts.
- No autoscaling or dynamic worker discovery.
- No distributed orchestration for one role spanning multiple nodes through
  `nnodes` and `node-rank`. Users may still pass those engine arguments
  manually later, but first-version lifecycle management treats each configured
  node as one Docker container.
- No attempt to normalize vLLM and SGLang internal PD metrics. The benchmark
  result endpoint remains OpenAI-compatible request timing plus existing
  result columns.

## Configuration Design

Existing config remains valid. If `topology_profiles` is absent, the current
`serve_profiles` flow is used.

When `topology_profiles` is present, the benchmark matrix becomes:

```text
models x topology_profiles x bench_profiles
```

Example SGLang PD profile:

```json
{
  "run": {
    "name": "sglang_pd_remote_bench",
    "results_dir": "vllm_standalone_bench/results",
    "bench_image": "vllm-bench-runner:offline",
    "api_key": "local-bench-key",
    "ready_timeout_sec": 1800,
    "resource_monitor": {
      "enabled": true,
      "backend": "nvidia-smi",
      "interval_sec": 1.0
    }
  },
  "mounts": {
    "models": "/models",
    "datasets": "/datasets"
  },
  "models": [
    {
      "name": "glm52",
      "model_path": "/models/GLM-5.2-FP8",
      "tokenizer_path": "/models/GLM-5.2-FP8",
      "served_model_name": "GLM-5.2"
    }
  ],
  "topology_profiles": [
    {
      "name": "sglang_pd_2p2d",
      "engine": "sglang",
      "mode": "pd",
      "provider": "ssh_docker",
      "transfer_backend": "mooncake",
      "network": "host",
      "image": "lmsysorg/sglang:latest",
      "router_image": "sglang-router:offline",
      "hosts": {
        "p1": {
          "address": "10.0.0.11",
          "ssh_user": "root",
          "auth": {"type": "key"}
        },
        "p2": {
          "address": "10.0.0.12",
          "ssh_user": "root",
          "auth": {"type": "password_env", "env": "P2_SSH_PASSWORD"}
        },
        "d1": {
          "address": "10.0.0.21",
          "ssh_user": "root",
          "auth": {"type": "key"}
        },
        "d2": {
          "address": "10.0.0.22",
          "ssh_user": "root",
          "auth": {"type": "key"}
        },
        "router": {
          "address": "10.0.0.30",
          "ssh_user": "root",
          "auth": {"type": "key"}
        }
      },
      "prefill": [
        {
          "name": "p1",
          "host": "p1",
          "port": 30000,
          "bootstrap_port": 12335,
          "gpus": "0,1,2,3",
          "args": ["--tp-size", "4", "--trust-remote-code"]
        },
        {
          "name": "p2",
          "host": "p2",
          "port": 30000,
          "bootstrap_port": 12335,
          "gpus": "0,1,2,3",
          "args": ["--tp-size", "4", "--trust-remote-code"]
        }
      ],
      "decode": [
        {
          "name": "d1",
          "host": "d1",
          "port": 30001,
          "bootstrap_port": 12335,
          "gpus": "0,1,2,3",
          "args": ["--tp-size", "4", "--trust-remote-code"]
        },
        {
          "name": "d2",
          "host": "d2",
          "port": 30001,
          "bootstrap_port": 12335,
          "gpus": "0,1,2,3",
          "args": ["--tp-size", "4", "--trust-remote-code"]
        }
      ],
      "frontend": {
        "kind": "sglang_router",
        "host": "router",
        "port": 8000,
        "args": [
          "--prefill-policy", "cache_aware",
          "--decode-policy", "power_of_two"
        ]
      }
    }
  ],
  "bench_profiles": [
    {
      "name": "latency_matrix",
      "backend": "openai-chat",
      "input_lens": [512, 2048],
      "output_lens": [256],
      "parallel_nums": [1, 4, 16],
      "epochs": 3
    }
  ]
}
```

### SSH Authentication

Supported `auth.type` values:

- `key`: use SSH key authentication through the user's SSH configuration or an
  optional explicit key path.
- `password_env`: read the password from a local environment variable named by
  `auth.env`.
- `password`: supported for temporary internal testing only. Any resolved
  config, command log, dry-run output, or error message must render the value
  as `***`.

`password_env` is preferred over inline `password`. Resolved config stores the
environment variable name, not the secret value.

## Runtime Architecture

Add a topology orchestration layer under `auto_bench.py`, with helper modules
if the implementation would otherwise make the file too large:

```text
AutoBenchConfig
  run
  mounts
  models
  serve_profiles        # existing local single-service path
  topology_profiles     # new remote PD path
  bench_profiles

TopologyProfile
  engine: sglang|vllm
  mode: pd
  provider: ssh_docker
  hosts
  prefill
  decode
  frontend

RemoteDockerRunner
  execute(host, command)
  capture(host, command)
  docker_run(host, args)
  docker_logs(host, container)
  docker_inspect(host, container)
  docker_stop_rm(host, container)
```

The existing local Docker path stays intact. The new path is selected only
when running a `BenchmarkCase` derived from a `TopologyProfile`.

## Execution Flow

For a PD topology case:

1. Parse config and validate topology references.
2. Validate `password_env` variables are present when needed.
3. Build a remote runner for each configured host.
4. Start all prefill containers through SSH.
5. Start all decode containers through SSH.
6. Wait for prefill and decode readiness.
   - SGLang: prefer `/health`.
   - vLLM: prefer `/v1/models`.
7. Start the frontend container.
8. Wait for frontend readiness.
9. Build the local bench-runner command with:

   ```text
   --base-url http://<frontend.address>:<frontend.port>/v1
   ```

10. During the bench-runner lifecycle, start remote resource monitoring for
    each involved host.
11. Stop remote resource monitoring in `finally` and write summaries.
12. Save remote role logs, inspect output, and generated commands.
13. Stop and remove remote frontend, decode, and prefill containers in reverse
    dependency order.
14. Record case status in `manifest.json` and `status.json`.

Cleanup order:

```text
local bench-runner container
remote frontend
remote decode nodes
remote prefill nodes
```

## SGLang Adapter

SGLang prefill and decode roles are generated from the same command template:

```text
docker run -d --name <container> \
  --label vllm_auto_bench.managed=true \
  --label vllm_auto_bench.run_id=<run_id> \
  --label vllm_auto_bench.run_dir=<run_dir> \
  --label vllm_auto_bench.model=<model> \
  --label vllm_auto_bench.topology_profile=<profile> \
  --label vllm_auto_bench.role=<prefill|decode> \
  --label vllm_auto_bench.role_name=<node_name> \
  --gpus '"device=<gpus>"' \
  --network host \
  -v <remote_model_root>:/models:ro \
  --entrypoint python3 <image> \
  -m sglang.launch_server \
  --model-path <model_path> \
  --served-model-name <served_model_name> \
  --host 0.0.0.0 \
  --port <node.port> \
  --disaggregation-mode <prefill|decode> \
  --disaggregation-transfer-backend <transfer_backend> \
  --disaggregation-bootstrap-port <node.bootstrap_port> \
  [--disaggregation-ib-device <value>] \
  <node.args...>
```

SGLang router command:

```text
docker run -d --name <container> \
  --label ... \
  --network host \
  --entrypoint python3 <router_image_or_image> \
  -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<p1.address>:<p1.port> [<p1.bootstrap_port>] \
  --prefill http://<p2.address>:<p2.port> [<p2.bootstrap_port>] \
  --decode http://<d1.address>:<d1.port> \
  --decode http://<d2.address>:<d2.port> \
  --host 0.0.0.0 \
  --port <frontend.port> \
  <frontend.args...>
```

The adapter must allow topology-level defaults and node-level overrides for:

- `transfer_backend`
- `disaggregation_ib_device`
- environment variables
- Docker volumes
- Docker `--gpus`
- additional engine args

## vLLM Adapter

vLLM PD is supported through explicit KV transfer configuration. The first
version should not attempt to infer connector-specific settings.

Worker command:

```text
docker run -d --name <container> \
  --label ... \
  --network host \
  -v <remote_model_root>:/models:ro \
  --entrypoint vllm <image> \
  serve <model_path> \
  --served-model-name <served_model_name> \
  --host 0.0.0.0 \
  --port <node.port> \
  --kv-transfer-config '<rendered_json>' \
  <node.args...>
```

Supported first-version frontend modes:

- `external`: run a configured proxy image and command. This is the recommended
  vLLM production path because users can package their known-good proxy.
- `vllm_example_proxy`: run the vLLM example proxy command when the image
  contains the required example file and dependencies.

The vLLM adapter must support `kv_transfer_config_template` with variable
substitution for values such as node address, node port, run id, rank, and
parallel size.

## Remote Resource Monitoring

The existing `resource_monitor.py` samples local `/proc` and `nvidia-smi`.
For remote PD topologies, reuse the same parser and summary logic but obtain
raw samples through SSH.

Each relevant host gets its own resource output directory:

```text
resources/<host_name>/resource_samples.csv
resources/<host_name>/resource_summary.json
```

The case-level result merge appends host-prefixed summary columns, such as:

```text
p1_gpu_util_avg_pct
p1_gpu_mem_used_max_mb
d1_gpu_util_avg_pct
router_cpu_util_avg_pct
```

If a remote monitor fails, benchmark success is unaffected. The resource
summary records unavailable state for that host.

## Result Layout

PD case directories use `topology_profile` instead of `serve_profile`:

```text
results/<run_id>/<model>/<topology_profile>/<bench_profile>/
  bench.log
  result.csv
  result.xlsx
  status.json
  topology.resolved.json
  commands/
    p1.txt
    p2.txt
    d1.txt
    d2.txt
    router.txt
    bench.txt
  logs/
    p1.log
    p2.log
    d1.log
    d2.log
    router.log
  inspect/
    p1.json
    p2.json
    d1.json
    d2.json
    router.json
  resources/
    p1/resource_samples.csv
    p1/resource_summary.json
    d1/resource_samples.csv
    d1/resource_summary.json
    router/resource_samples.csv
    router/resource_summary.json
```

Manifest rows add `topology_profile` while preserving existing fields:

```json
{
  "model": "glm52",
  "serve_profile": null,
  "topology_profile": "sglang_pd_2p2d",
  "bench_profile": "latency_matrix",
  "status": "passed",
  "endpoint": "http://10.0.0.30:8000/v1"
}
```

The compare aggregation must treat either `serve_profile` or
`topology_profile` as the serving dimension.

## Error Handling

- Config validation errors fail before any remote command runs.
- SSH connection failure marks the topology case `skipped`.
- Remote `docker run` failure stops already-started roles and marks the case
  `failed`.
- Ready timeout saves role logs and inspect output where possible, then marks
  the case `failed`.
- Bench-runner failure marks the case `failed` and still cleans up the remote
  topology.
- Stop signals mark active and pending topology cases `interrupted`.
- Cleanup failures are warnings. The controller must only attempt to remove
  containers with matching ownership labels.

Remote containers must carry these labels:

```text
vllm_auto_bench.managed=true
vllm_auto_bench.run_id=<run_id>
vllm_auto_bench.run_dir=<run_dir>
vllm_auto_bench.model=<model>
vllm_auto_bench.topology_profile=<profile>
vllm_auto_bench.role=<prefill|decode|frontend>
vllm_auto_bench.role_name=<node_name>
```

## Status, Logs, Stop, and Resume

- `status` continues to read `state.json` and `manifest.json`.
- During a PD case, `state.json` includes `topology_profile` and the frontend
  endpoint.
- `logs` initially continues to show controller or current bench logs. The
  command output should mention where role logs live for PD cases.
- `stop` still signals the controller. The controller performs remote cleanup
  from its in-memory topology state and ownership labels.
- `resume` uses the original `config.resolved.json`. Passed cases are skipped.
  Pending keys are:

  ```text
  (model, serve_profile, bench_profile)
  ```

  for legacy cases, and:

  ```text
  (model, topology_profile, bench_profile)
  ```

  for PD topology cases.

## Security

- Password values are never printed.
- Inline `auth.password` is allowed only with masking. It is written as `***`
  in resolved config and command logs.
- `password_env` is preferred. The resolved config stores only the environment
  variable name.
- SSH host key policy should default to the user's known-hosts behavior. An
  optional `allow_unknown_host_key` may be added later, but it is not required
  for the first implementation.

## Tests

Unit tests:

- Existing configs without `topology_profiles` parse and run as before.
- SGLang PD config validates hosts, auth, prefill, decode, frontend, and port
  references.
- `password_env` succeeds when the variable exists and fails clearly when it
  does not.
- Inline password and env-derived password are masked in dry-run and resolved
  config.
- SGLang prefill, decode, and router commands match the documented flags.
- PD bench command uses the frontend `base_url`.
- Remote Docker cleanup checks ownership labels before removing containers.
- Resume case keys support both legacy serve profiles and topology profiles.

Fake integration tests:

- 2P2D SGLang PD success path starts prefill/decode before frontend, then runs
  local bench-runner, saves artifacts, and cleans remote containers.
- Prefill startup failure cleans any already-started prefill containers.
- Frontend ready timeout cleans prefill, decode, and frontend containers.
- Stop during benchmark marks the case interrupted and cleans remote roles.
- Remote resource monitoring writes per-host summaries and merges prefixed
  columns into result files.

Verification commands:

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
bash -n vllm_standalone_bench/run_auto_bench.sh
git diff --check
```

## Acceptance Criteria

- Existing smoke and SGLang comparison configs work unchanged.
- A SGLang `topology_profiles` dry-run prints remote Docker commands for all
  prefill, decode, and router roles, with passwords masked.
- The generated SGLang router command registers every configured prefill and
  decode endpoint using configured IPs and ports.
- The local bench-runner command targets the router endpoint through
  `--base-url`.
- On a fake 2P2D run, artifacts are written under
  `results/<run_id>/<model>/<topology_profile>/<bench_profile>/`.
- `stop` and `resume` preserve current semantics for both legacy and PD cases.
- Resource summary files are present per remote host when monitoring is enabled.

## Risks

- SGLang router image packaging may vary by environment. The design allows a
  separate `router_image` so users can provide an offline image containing
  `sglang_router`.
- vLLM PD proxy behavior is less standardized than SGLang router behavior. The
  first implementation treats vLLM proxy as explicit user configuration rather
  than inferring it.
- Remote resource monitoring through SSH can add sampling overhead. It is
  best-effort and should not change benchmark pass/fail status.
- Cross-host networking, firewall, RDMA, and model path correctness are outside
  the tool's automatic repair scope. The tool validates configuration shape
  and reports readiness failures with role logs.
