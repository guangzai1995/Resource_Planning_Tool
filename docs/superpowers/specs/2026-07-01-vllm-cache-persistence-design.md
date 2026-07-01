# vLLM 编译/JIT Cache 持久化设计

## Status

Ready for implementation planning.

## Context

GLM5.2 启动时间很长，尤其在 `vllm_standalone_bench/auto_bench.py` 自动化框架反复启停 vLLM 服务时，单轮启动和编译可能接近半小时。当前用户选择的优化目标是 **A：优化自动化 benchmark 多轮运行总耗时，允许第一次慢、后续快**。

源码与框架梳理结论：

1. `auto_bench.py` 启动 vLLM serving 容器时只挂载模型目录：`-v <models>:/models:ro`。vLLM 默认编译/JIT 缓存位于容器内，例如 `/root/.cache/vllm`，容器删除后缓存随之丢失。
2. vLLM 的 `VLLM_CACHE_ROOT` 控制 vLLM cache 根目录；torch.compile、AOT compile、TorchInductor/Triton cache 会落在该根目录派生路径下。
3. vLLM 的 DeepGEMM 工具会把 `DG_JIT_CACHE_DIR` 默认设置到 `VLLM_CACHE_ROOT/deep_gemm`，因此持久化 `VLLM_CACHE_ROOT` 能覆盖 DeepGEMM JIT 缓存。
4. FlashInfer autotune 有单独的 `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` 可覆盖目录。GLM5.2 的 DSA/MoE 路径会大量依赖这类内核编译/调优。
5. vLLM 默认 `-O2` 会启用 `VLLM_COMPILE` 和较完整的 CUDA graph capture；GLM5.2 这类 DSA/MoE 模型首次启动成本高是合理的。
6. `auto_bench.py` 已经在同一个 `model + serve_profile` 组内复用一个服务，但不同 run 和不同 serve profile 仍会重启容器。当前缺口是重启后不能复用编译/JIT 产物。

## Goals

- 为 vLLM serving 容器提供可配置的持久化 cache mount。
- 默认行为完全兼容：未配置 cache 时，docker 命令与当前行为一致。
- 支持每个 serve profile 使用独立 cache key，避免不同模型、镜像、TP、dtype、优化参数混用同一 cache。
- 自动设置 vLLM 常用 cache env，减少配置方需要理解的 vLLM 内部路径。
- 将 resolved config 和 dry-run 命令完整展示 cache mount/env，便于审计。
- 将本地 cache 目录加入 `.gitignore`，避免大体积编译产物误提交。
- 保持 SGLang 对比能力不受影响；第一版只对 `engine=vllm` 生效。

## Non-Goals

- 不改变 vLLM 的优化等级、CUDA graph capture、DeepGEMM warmup 策略。
- 不实现长驻服务、`reuse_existing` 或 `keep_alive` 生命周期模式。
- 不自动清理 cache 目录。cache 是跨 run 复用资产，不属于单次 run artifact。
- 不保证不同 GPU 架构、不同 vLLM 镜像、不同模型参数之间的 cache 可共享。
- 不在第一版解析 vLLM 日志输出启动阶段耗时；该能力可作为后续观测增强。

## Design

### 配置结构

在 `run` 下新增 `vllm_cache` 对象：

```json
{
  "run": {
    "vllm_cache": {
      "enabled": true,
      "root": "/Resource_Planning_Tool/.cache/vllm_auto_bench",
      "container_path": "/vllm-cache",
      "set_default_env": true
    }
  }
}
```

字段语义：

- `enabled`：布尔值，默认 `false`。关闭时不挂载 cache、不设置 cache env。
- `root`：宿主机 cache 根目录。必填于 `enabled=true`。相对路径按配置文件所在目录解析。
- `container_path`：容器内 cache 根目录，默认 `/vllm-cache`。必须是绝对 POSIX 路径，不能包含 `..`。
- `set_default_env`：默认 `true`。启用时自动注入：
  - `VLLM_CACHE_ROOT=<container_path>`
  - `DG_JIT_CACHE_DIR=<container_path>/deep_gemm`
  - `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=<container_path>/flashinfer_autotune`

在 `serve_profiles[]` 下新增可选字段：

```json
{
  "name": "glm52_fp8_tp8_o2",
  "engine": "vllm",
  "cache_key": "glm52-fp8-tp8-h20-o2",
  "gpus": "all",
  "args": ["--tensor-parallel-size", "8", "--kv-cache-dtype", "fp8"]
}
```

- `cache_key`：安全文件名，默认自动生成。
- 默认 key 按 `BenchmarkCase` 生成：`<model.name>__<serve_profile.name>__<image_hash>`。
- 推荐 GLM5.2 正式配置显式写 `cache_key`，把模型、GPU 架构、TP、dtype、优化口径写进名字，例如 `glm52-fp8-tp8-h20-o2`。
- 如果同一个 serve profile 同时用于多个模型，不建议手写同一个 `cache_key`；应使用默认 key，或为不同模型拆分 serve profile。

### Cache 目录布局

启用后，每个 vLLM serving 容器挂载：

```text
<root>/<cache_key>:/vllm-cache:rw
```

实际目录示例：

```text
/Resource_Planning_Tool/.cache/vllm_auto_bench/
  glm52-fp8-tp8-h20-o2/
    torch_compile_cache/
    deep_gemm/
    flashinfer_autotune/
```

如果 `cache_key` 自动生成，使用安全字符串直接作为 `<root>/<cache_key>`；如果用户显式配置 `cache_key`，同样直接作为 `<root>/<cache_key>`。

### Docker 命令行为

未启用 cache 时，现有命令保持不变：

```text
docker run -d ... -v <models>:/models:ro --entrypoint vllm <image> serve ...
```

启用 cache 且 `engine=vllm` 时，命令增加：

```text
-v <cache_dir>:/vllm-cache:rw
-e VLLM_CACHE_ROOT=/vllm-cache
-e DG_JIT_CACHE_DIR=/vllm-cache/deep_gemm
-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/vllm-cache/flashinfer_autotune
```

`engine=sglang` 的 serve profile 不挂载 vLLM cache，也不注入这些 env。

### Validation

配置解析新增校验：

- `run.vllm_cache` 必须是对象。
- `enabled`、`set_default_env` 必须是布尔值。
- `enabled=true` 时 `root` 必须存在或可创建；实现应在启动前 `mkdir -p`。
- `container_path` 必须是绝对 POSIX 路径，不能包含 `..`。
- `cache_key` 必须满足现有 `SAFE_NAME_RE`，不能是 `.` 或 `..`。
- 自动 key 使用模型名、serve profile 名和镜像字符串 hash，避免 Docker image tag 中的 `/`、`:` 污染路径。

本地路径验证阶段：

- `validate_local_paths()` 不要求 cache root 预先存在，但会尝试创建 `<root>` 和 `<root>/<cache_key>`。
- 创建失败时报 `ConfigError`，不要等到 `docker run` 才失败。

### Resolved Config

`config.resolved.json` 应包含：

- `run.vllm_cache` 的解析后绝对 `root`。
- `serve_profiles[].cache_key` 的用户配置值或 `null`。

实际 per-case cache key 由 `BenchmarkCase` 计算。为便于复现，每个 serve 目录新增 `vllm_cache.json`：

```json
{
  "enabled": true,
  "cache_key": "glm52-fp8-tp8-h20-o2",
  "host_dir": "/Resource_Planning_Tool/.cache/vllm_auto_bench/glm52-fp8-tp8-h20-o2",
  "container_path": "/vllm-cache",
  "env": {
    "VLLM_CACHE_ROOT": "/vllm-cache",
    "DG_JIT_CACHE_DIR": "/vllm-cache/deep_gemm",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/vllm-cache/flashinfer_autotune"
  }
}
```

这样后续复现实验时能明确知道实际使用了哪个 cache 目录。

### Example Config

`configs/auto_bench.qwen2_5_1_5b.smoke.json` 默认不启用 cache，避免改变现有 smoke 行为。

新增一个 GLM5.2 或通用示例片段放入 README，而不是强行改所有样例：

```json
{
  "run": {
    "vllm_cache": {
      "enabled": true,
      "root": "/Resource_Planning_Tool/.cache/vllm_auto_bench",
      "container_path": "/vllm-cache",
      "set_default_env": true
    }
  },
  "serve_profiles": [
    {
      "name": "glm52_fp8_tp8_o2",
      "engine": "vllm",
      "cache_key": "glm52-fp8-tp8-h20-o2",
      "gpus": "all",
      "args": [
        "--tensor-parallel-size", "8",
        "--kv-cache-dtype", "fp8"
      ]
    }
  ]
}
```

### Git Ignore

新增：

```gitignore
.cache/
```

原因：vLLM 编译/JIT cache 体积可能很大，而且是宿主机和 GPU 环境相关产物，不应提交。

## Expected Impact

第一次运行某个 cache key 时，GLM5.2 仍然需要完整编译、JIT 和 autotune。

后续相同条件的运行应复用：

- vLLM torch compile cache
- AOT compiled function
- TorchInductor/Triton cache
- DeepGEMM JIT cache
- FlashInfer autotune cache

预期结果是自动化 benchmark 的第二次及后续启动显著缩短。实际缩短幅度取决于 vLLM 版本、PyTorch 版本、GPU 架构、serve args、TP 配置和模型结构。

## Risks / Trade-offs

- **Cache key 过宽**：不同 TP、dtype、优化参数共用 cache key，可能导致 cache 污染或难以解释结果。通过显式 `cache_key` 和文档约束缓解。
- **Cache key 过窄**：每次 profile 名变化都生成新 cache，复用率下降。通过 README 建议稳定命名。
- **磁盘增长**：cache 不随 run 清理。需要用户定期删除 `.cache/vllm_auto_bench/<key>`。
- **正式 benchmark 口径**：持久化 cache 会改变“冷启动时间”指标，但不应改变服务启动后的请求性能口径。结果报告应说明这是 warm-cache benchmark harness。
- **跨 GPU 复用**：不同 GPU 架构或驱动环境不应强行共享同一个 key。GLM5.2 配置建议把硬件代号写入 `cache_key`。

## Test Strategy

- `tests/test_auto_bench.py`
  - 默认配置不启用 cache，`build_vllm_run_command()` 不包含 `/vllm-cache`、`VLLM_CACHE_ROOT` 或额外 `-v`。
  - 启用 `run.vllm_cache` 后，解析得到绝对 `root`、默认 `container_path` 和 `set_default_env=true`。
  - 启用 cache 的 vLLM 命令包含 cache mount 和三个 env。
  - `engine=sglang` 时不注入 vLLM cache mount/env。
  - 非法 `container_path`、非法 `cache_key`、非对象 `vllm_cache` 抛 `ConfigError`。
  - `resolve_vllm_cache_dir()` 对默认 key 和显式 key 都返回正确目录。
  - 启用 cache 后，run 的 serve 目录写入 `vllm_cache.json`。
  - `validate_local_paths()` 会创建 cache root/key 目录。

- Dry-run
  - `python3 vllm_standalone_bench/auto_bench.py run --dry-run ...` 输出包含 cache mount/env。
  - `config.resolved.json` 含绝对 cache root；dry-run 命令含实际 per-case cache 目录。

- Targeted pytest
  - `PYTHONPATH=vllm_standalone_bench pytest -q vllm_standalone_bench/tests/test_auto_bench.py`

- Baseline note
  - 当前 worktree 完整 `pytest -q` 已知存在与本任务无关的既有失败：`tests/test_inference_token_factory_report.py` 依赖缺失的 `outputs/context_analysis_20260609_034248/01_overview.json`。

## Implementation Scope

1. 添加 `VllmCacheConfig` dataclass，并在 `RunConfig` 或 `AutoBenchConfig` 中挂接。
2. 扩展 `_parse_run()` / `_parse_serve_profiles()` 解析 `run.vllm_cache` 和 `serve_profiles[].cache_key`。
3. 新增 cache key 生成与 cache 目录解析函数。
4. 扩展 `validate_local_paths()` 创建 cache 目录。
5. 扩展 `build_vllm_run_command()` 注入 cache mount/env。
6. 在每个 serve 目录写入 `vllm_cache.json` 记录实际 cache 信息。
7. 扩展 `config_to_dict()` resolved 输出绝对 cache root。
8. 更新 README 增加 GLM5.2 warm-cache 配置说明。
9. 更新 `.gitignore` 忽略 `.cache/`。
