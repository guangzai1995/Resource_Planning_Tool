# vLLM Offline Auto Bench Design

## Status

This specification is ready for user review.

## Context

The existing `vllm_standalone_bench` project already provides a lightweight benchmark client:

- `run_bench_multi.py` runs a matrix of input lengths, output lengths, concurrency levels, epochs, prefix ratios, and exports CSV/XLSX.
- `run_bench_serve.py` provides local shims so the benchmark client does not need `vllm` or `torch` installed.
- `run_bench.sh` is a fixed-parameter shell wrapper for an already running OpenAI-compatible vLLM service.

The missing capability is service lifecycle orchestration:

1. Start a vLLM container from an offline image.
2. Wait for the model service to become ready.
3. Run benchmark profiles in a separate benchmark runner container.
4. Collect logs, CSV, XLSX, and metadata.
5. Stop the vLLM container.
6. Clean up Docker network resources created by the run.
7. Continue through a full matrix of models, serving profiles, and benchmark profiles.

The target test host may have no network access and only a basic host environment. Python benchmark dependencies such as `aiohttp`, `numpy`, `tqdm`, `transformers`, and `openpyxl` must therefore be provided by an offline container image instead of installed on the host.

The current host can run a real smoke test. The vLLM image `009e4cb46541` is present locally and resolves to `xemegpzeib7tis.xuanyuan.run/vllm/vllm-openai:nightly`. A Qwen 1.5B-class model should be prepared from ModelScope before smoke validation, using `Qwen/Qwen2.5-1.5B-Instruct` from https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct.

## Goals

- Support a full matrix: multiple models, multiple vLLM serving profiles, and multiple benchmark profiles.
- Use two images: one vLLM serving image and one benchmark runner image.
- Run orchestration on the host through a Python standard-library script.
- Avoid `--network host` by default and avoid fixed host port conflicts.
- Support foreground and detached background execution.
- Preserve partial results when a scenario fails or the user interrupts the run.
- Save enough metadata to reproduce each benchmark case.
- Clean up vLLM containers and Docker networks created by the orchestration run.
- Allow implementation in a git worktree, then merge back to `main` before final validation.

## Non-Goals

- No web UI in the first version.
- No Docker Compose dependency in the first version.
- No parallel execution of multiple vLLM serving containers in the first version.
- No automatic idle GPU selection in the first version.
- No automatic host port selection in the first version.
- No complex resume engine in the first version. Completed artifacts are preserved, but reruns start a new `run_id`.

## Architecture

The system has four boundaries:

1. **Host orchestrator**
   - File: `vllm_standalone_bench/auto_bench.py`
   - Runs on the host.
   - Depends only on Python standard library and Docker CLI.
   - Reads JSON config, validates local resources, starts/stops containers, waits for readiness, writes state and manifest files.

2. **vLLM serving container**
   - Uses the configured offline vLLM image.
   - Mounts model files read-only.
   - Runs `vllm serve <model_path>` with the selected serving profile.
   - Listens on `0.0.0.0:8000` inside the Docker network.

3. **Benchmark runner container**
   - Uses a separate offline image built in a networked environment.
   - Contains `vllm_standalone_bench` source files and Python benchmark dependencies.
   - Runs `run_bench_multi.py`.
   - Mounts models read-only for tokenizer access and mounts results read-write.

4. **JSON test config**
   - Describes models, serving profiles, benchmark profiles, Docker images, mounts, networking, timeouts, and output settings.
   - Uses JSON instead of YAML so the host orchestrator does not require `PyYAML`.

The execution matrix is:

```text
for model in models:
  for serve_profile in serve_profiles:
    start vLLM container
    wait until /v1/models is ready
    for bench_profile in bench_profiles:
      run bench-runner container
    stop vLLM container
    collect vLLM logs and metadata
```

## Network Model

The default network mode is a user-defined Docker bridge network.

```text
Docker network: configured name, default vllm-bench-net
vLLM container listen address: 0.0.0.0:8000
bench-runner access URL: http://<vllm_container_name>:8000/v1
host port publishing: disabled by default
```

This avoids conflicts with ports already used on the host.

Optional host port publishing is supported only for manual debugging:

```json
{
  "publish_host_port": true,
  "host_port": 18000
}
```

When publishing is enabled, the orchestrator binds `127.0.0.1:<host_port>` to container port `8000`. It checks whether the host port is already occupied before starting the scenario. If the port is occupied, the scenario fails cleanly instead of replacing or killing another process.

## Docker Network Lifecycle

Network cleanup is part of the orchestrator lifecycle.

Configuration:

```json
{
  "network": "vllm-bench-net",
  "create_network": true,
  "cleanup_network": true
}
```

Rules:

- If `create_network=true` and the network does not exist, the orchestrator creates it and marks it as owned by this run.
- If the network already exists, the orchestrator reuses it and marks it as not owned by this run.
- If `cleanup_network=true`, the orchestrator removes only networks created by the current run.
- Before removal, the orchestrator checks connected containers.
- If unexpected external containers are still attached or removal fails, the orchestrator logs a warning and leaves the network in place.
- On `SIGINT` or `SIGTERM`, the orchestrator stops the current vLLM container and then applies the same network cleanup rules.

## Configuration Format

Example:

```json
{
  "run": {
    "name": "h20_full_bench",
    "results_dir": "vllm_standalone_bench/results",
    "vllm_image": "009e4cb46541",
    "bench_image": "vllm-bench-runner:offline",
    "network": "vllm-bench-net",
    "create_network": true,
    "cleanup_network": true,
    "container_port": 8000,
    "publish_host_port": false,
    "host_port": 18000,
    "api_key": "local-bench-key",
    "ready_timeout_sec": 1800,
    "cooldown_sec": 20
  },
  "mounts": {
    "models": "/Resource_Planning_Tool/model"
  },
  "models": [
    {
      "name": "qwen2_5_1_5b",
      "model_path": "/models/Qwen2.5-1.5B-Instruct",
      "tokenizer_path": "/models/Qwen2.5-1.5B-Instruct",
      "served_model_name": "qwen2_5_1_5b"
    }
  ],
  "serve_profiles": [
    {
      "name": "bf16_default",
      "gpus": "all",
      "args": [
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.90"
      ]
    },
    {
      "name": "bf16_prefix_on",
      "gpus": "all",
      "args": [
        "--dtype", "bfloat16",
        "--enable-prefix-caching",
        "--gpu-memory-utilization", "0.90"
      ]
    }
  ],
  "bench_profiles": [
    {
      "name": "smoke",
      "backend": "openai-chat",
      "input_lens": [64],
      "output_lens": [32],
      "parallel_nums": [1],
      "epochs": 1,
      "prefix_ratio": 0.0,
      "warmup_requests": 0
    },
    {
      "name": "latency_matrix",
      "backend": "openai-chat",
      "input_lens": [128, 512, 1024],
      "output_lens": [1024],
      "parallel_nums": [1, 4, 8],
      "epochs": 3,
      "prefix_ratio": 0.8,
      "warmup_requests": 1,
      "max_ttft_ms": 15000,
      "min_throughput_tok_s": 5
    }
  ]
}
```

Model path semantics:

- `model_path` is the path inside the vLLM container.
- `tokenizer_path` is the path inside the benchmark runner container.
- Both paths are usually under `/models` because the same host model root is mounted into both containers.
- API requests use `served_model_name` as the model name. They do not use the filesystem model path.

## Commands

Foreground run:

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.example.json
```

Detached background run:

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.example.json \
  --detach
```

Status:

```bash
python3 vllm_standalone_bench/auto_bench.py status --run-id <run_id>
```

Logs:

```bash
python3 vllm_standalone_bench/auto_bench.py logs --run-id <run_id>
python3 vllm_standalone_bench/auto_bench.py logs --run-id <run_id> --follow
```

Stop:

```bash
python3 vllm_standalone_bench/auto_bench.py stop --run-id <run_id>
```

Dry run:

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.example.json \
  --dry-run
```

## Background Execution

Detached mode creates a run directory first, then starts a child controller process.

Artifacts:

```text
results/<run_id>/
  controller.pid
  controller.log
  state.json
  manifest.json
```

Behavior:

- The parent process writes initial state and returns the `run_id`.
- The child process continues the full matrix execution.
- Child stdout and stderr are redirected to `controller.log`.
- `status` reads `state.json`, `controller.pid`, and `manifest.json`.
- `logs --follow` tails `controller.log`.
- `stop` sends `SIGTERM` to the controller process.
- The controller handles termination by stopping the current vLLM container, writing an interrupted state, and cleaning up owned Docker networks.

## Bench Runner Image

File:

```text
vllm_standalone_bench/Dockerfile.bench-runner
```

Contents:

```text
/opt/vllm_standalone_bench/
  run_bench_multi.py
  run_bench_serve.py
  requirements.txt
  vllm_bench/
```

Dependencies installed in the image:

```text
aiohttp
numpy
tqdm
transformers
openpyxl
pytest
modelscope
```

Rationale:

- `openpyxl` is included so XLSX output works offline.
- `transformers` is included for local tokenizer-based prompt generation.
- `pytest` is included for build-time or offline smoke checks.
- `modelscope` is included for the optional model preparation command in networked environments.
- `vllm` and `torch` are not required in this image because `run_bench_serve.py` provides shims.

Networked build:

```bash
docker build \
  -f vllm_standalone_bench/Dockerfile.bench-runner \
  -t vllm-bench-runner:offline \
  vllm_standalone_bench
docker save vllm-bench-runner:offline -o vllm-bench-runner-offline.tar
```

Offline load:

```bash
docker load -i vllm-bench-runner-offline.tar
docker load -i vllm-offline-image.tar
```

## Model Preparation

Model download is a networked preparation step, not part of the offline benchmark matrix.

Command:

```bash
python3 vllm_standalone_bench/auto_bench.py prepare-model \
  --modelscope-id Qwen/Qwen2.5-1.5B-Instruct \
  --target /Resource_Planning_Tool/model/Qwen2.5-1.5B-Instruct \
  --bench-image vllm-bench-runner:offline
```

Rules:

- Download into `<target>.download-tmp`.
- Validate that tokenizer files, config files, and at least one complete `*.safetensors` weight file exist.
- Move into `<target>` only after validation.
- If `<target>` already exists but lacks complete weight files, fail with a clear error unless `--force` is passed.
- `--force` moves the existing target aside into a timestamped backup before replacing it.

The Qwen smoke model is `Qwen/Qwen2.5-1.5B-Instruct` from ModelScope:

```text
https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct
```

## Container Commands

vLLM container example:

```bash
docker run -d \
  --name bench-vllm-qwen2_5_1_5b-bf16_default-<run_id> \
  --gpus all \
  --network vllm-bench-net \
  -v /Resource_Planning_Tool/model:/models:ro \
  009e4cb46541 \
  vllm serve /models/Qwen2.5-1.5B-Instruct \
    --served-model-name qwen2_5_1_5b \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key local-bench-key \
    --dtype bfloat16
```

Benchmark runner example:

```bash
docker run --rm \
  --network vllm-bench-net \
  -v /Resource_Planning_Tool/model:/models:ro \
  -v /Resource_Planning_Tool/vllm_standalone_bench/results/<run_id>/qwen2_5_1_5b/bf16_default/smoke:/results \
  vllm-bench-runner:offline \
  python /opt/vllm_standalone_bench/run_bench_multi.py \
    --base-url http://bench-vllm-qwen2_5_1_5b-bf16_default-<run_id>:8000/v1 \
    --model qwen2_5_1_5b \
    --served-model-name qwen2_5_1_5b \
    --backend openai-chat \
    --api-key local-bench-key \
    --tokenizer /models/Qwen2.5-1.5B-Instruct \
    --input-lens 64 \
    --output-lens 32 \
    --parallel-nums 1 \
    --epochs 1 \
    --warmup-requests 0 \
    --output-csv /results/result.csv \
    --output-xlsx /results/result.xlsx
```

## Validation and Failure Handling

Preflight validation:

- Docker CLI is available.
- vLLM image exists locally.
- benchmark runner image exists locally.
- model host root exists.
- each configured model path exists after mount translation.
- tokenizer path exists when configured.
- result root is writable.
- profile and model names contain only safe filename characters: letters, digits, dot, underscore, and hyphen.
- `serve_profile.args` is a string array.
- `input_lens`, `output_lens`, and `parallel_nums` are positive integer arrays.
- `output_lens` length is either `1` or equal to `input_lens` length, unless `cross_product=true`.

Runtime handling:

- Remove a stale container only when its name matches the deterministic name generated for the current run.
- Save `docker logs` and `docker inspect` for every vLLM scenario.
- If vLLM readiness fails, mark all benchmark profiles under that serving scenario as skipped.
- If one benchmark profile fails, mark it failed and continue with the next benchmark profile.
- If the user interrupts the controller, stop the current vLLM container, write interrupted state, and clean up owned Docker networks.

## Results Layout

```text
vllm_standalone_bench/results/<run_id>/
  config.resolved.json
  controller.log
  controller.pid
  manifest.json
  state.json
  <model>/
    <serve_profile>/
      docker.inspect.json
      serve_command.txt
      vllm.log
      <bench_profile>/
        bench.log
        result.csv
        result.xlsx
        status.json
```

`manifest.json` summary:

```json
{
  "run_id": "h20_full_bench_20260629_153000",
  "status": "completed_with_failures",
  "cases": [
    {
      "model": "qwen2_5_1_5b",
      "serve_profile": "bf16_default",
      "bench_profile": "smoke",
      "status": "passed",
      "csv": "qwen2_5_1_5b/bf16_default/smoke/result.csv",
      "xlsx": "qwen2_5_1_5b/bf16_default/smoke/result.xlsx"
    }
  ]
}
```

`state.json` tracks current progress for `status`:

```json
{
  "run_id": "h20_full_bench_20260629_153000",
  "status": "running",
  "current": {
    "model": "qwen2_5_1_5b",
    "serve_profile": "bf16_default",
    "bench_profile": "smoke"
  },
  "counts": {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "running": 1,
    "total": 1
  }
}
```

## Development Workflow

Implementation should happen in an isolated git worktree.

Workflow:

1. Create a feature worktree and branch for the implementation.
2. Implement `Dockerfile.bench-runner`, `auto_bench.py`, example configs, tests, and README updates in the worktree.
3. Run unit tests and dry-run verification in the worktree.
4. Merge the feature branch back to `main`.
5. Run final verification on `main`, including the real smoke path when Docker, GPU, the vLLM image, and the complete Qwen model are available.

This keeps the current `main` checkout stable while implementing the orchestration layer.

## Test Strategy

Unit tests:

- JSON config parsing and default values.
- Matrix expansion for model, serving profile, and benchmark profile.
- Safe name validation.
- Host path to container path validation.
- Docker command construction.
- Result path construction.
- Network ownership and cleanup decision logic.
- Background state file updates.
- Manifest updates for passed, failed, skipped, and interrupted cases.

Dry-run tests:

- `run --dry-run` prints all Docker commands without starting containers.
- Dry run writes a resolved config preview but does not create benchmark result files.

Mocked integration tests:

- Mock Docker command execution.
- Simulate vLLM ready success.
- Simulate vLLM ready timeout.
- Simulate benchmark runner failure.
- Simulate `SIGTERM` cleanup.

Real smoke verification on the current host:

1. Ensure the vLLM image `009e4cb46541` exists.
2. Prepare `Qwen/Qwen2.5-1.5B-Instruct` from ModelScope into `/Resource_Planning_Tool/model/Qwen2.5-1.5B-Instruct`.
3. Build or load `vllm-bench-runner:offline`.
4. Run `configs/auto_bench.qwen2_5_1_5b.smoke.json`.
5. Verify `result.csv`, `result.xlsx`, `vllm.log`, `bench.log`, `status.json`, and `manifest.json` are generated.
6. Verify the current vLLM container is stopped after completion.
7. Verify Docker networks created by the run are removed after completion.

## First Implementation Scope

Files:

```text
vllm_standalone_bench/
  Dockerfile.bench-runner
  auto_bench.py
  configs/
    auto_bench.example.json
    auto_bench.qwen2_5_1_5b.smoke.json
  tests/
    test_auto_bench.py
  README.md
```

Required first-version features:

- JSON config.
- Full matrix execution.
- vLLM container lifecycle.
- benchmark runner container execution.
- Docker bridge network creation, reuse, and owned cleanup.
- Optional host port publishing for debugging.
- Foreground and detached background execution.
- Status, logs, stop, and dry-run commands.
- Model preparation command using the benchmark runner image and ModelScope.
- Result directory, logs, status files, and manifest.
- Unit tests and mock integration tests.
- Real smoke config for Qwen 1.5B-class validation with vLLM image `009e4cb46541`.

Deferred features:

- Web UI.
- Docker Compose support.
- Parallel vLLM serving scenarios.
- Automatic GPU selection.
- Automatic host port allocation.
- In-place resume of a partially completed `run_id`.
