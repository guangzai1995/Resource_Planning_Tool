# vLLM / SGLang 同台对比基准测试 — 设计文档

- 日期：2026-06-30
- 分支：`feat/sglang-bench-compat`
- worktree：`.worktrees/sglang-bench-compat`
- 状态：待审查（brainstorming 产物，下一步进入 writing-plans）

## 1. 背景与目标

现有 `vllm_standalone_bench` 离线自动压测工程只能启动 **vLLM** 服务做基准。本仓库已在 `.magic/sglang` 拉取 SGLang 最新源码，目标是让同一套工程能在**同一次 run 内**分别启动 vLLM 与 SGLang 两个推理后端，对同一模型、同一压测矩阵产出可对照的性能数据。

用户三个核心诉求：

1. **同台对比** — vLLM 与 SGLang 在同一框架、同一次 run 内可分别启动，结果按引擎可对照。
2. **输出到指定长度** — 压测时强制每个请求输出到指定的 token 数（由 `output_lens` 控制）。
3. **自动测试脚本兼容** — 现有 `run_bench_multi.py` 自动压测链路对 SGLang 同样可用，无需另起一套。

## 2. 现状分析（兼容性矩阵）

工程是分层的。经源码核对，只有「服务启动」与「配置 schema」两层与 vLLM 耦合，其余天然兼容 SGLang：

| 层 | 现状 | 对 SGLang |
|---|---|---|
| 压测请求 + 指标采集（`run_bench_multi.py` + `vllm_bench/serve.py`） | 走标准 OpenAI streaming API；TTFT/TPOT/throughput 等全基于 token 到达时间，与服务端无关；仅依赖 aiohttp/numpy/tqdm | ✅ 天然兼容，**0 改动** |
| 输出到指定长度 | 客户端在请求体传 `ignore_eos=true` + `max_tokens=output_len`（`serve.py` 强制 `ignore_eos=True`） | ✅ 兼容。SGLang OpenAI 协议含 `ignore_eos`（`entrypoints/openai/protocol.py:356` 与 `:728`，completions/chat 均支持），底层 `sampling/sampling_params.py:106` 支持 |
| 就绪探测 / 容器编排 | `wait_for_container_ready` 走 `/v1`；labels/network/cleanup/manifest 与服务端无关 | ✅ 兼容。SGLang 暴露 `/v1/models`（`entrypoints/http_server.py:1717`）与 `/health`（`:570`） |
| **服务启动命令**（`build_vllm_run_command`） | 硬编码 `--entrypoint vllm`、`config.run.vllm_image`、`serve <model>` 这套 vLLM 语法 | ❌ **vLLM 专属，需扩展**。SGLang 启动为 `python -m sglang.launch_server --model-path <m> --host 0.0.0.0 --port <p>`（参数 `model_path/host/port/api_key/served_model_name` 已确认） |
| **配置 schema** | `RunConfig.vllm_image` 单值写死；`ServeProfile` 无引擎概念 | ❌ **需扩展**：区分引擎 + per-engine 镜像 |

**结论**：用户关心的「输出到指定长度」「自动脚本兼容」对 SGLang 直接可用；真正要改的是「如何把 SGLang 服务启起来」+ 对应配置 schema。

## 3. 范围

### In scope

- `ServeProfile` 引入 `engine` 字段；`RunConfig` 引入 `images` 映射（向后兼容 `vllm_image`）。
- 启动命令按 `engine` 分派（vLLM 走原逻辑，SGLang 生成 `sglang.launch_server` 命令）。
- 新增对比聚合层 `bench_compare.py`：产出 `compare.csv/xlsx` + 图表，**原始 per-profile 结果原样保留**。
- bench-runner 镜像新增 `matplotlib`；README 补 SGLang 镜像离线搬运与参数等价提示。
- 配置/命令/聚合的单测与集成测试。

### Out of scope（YAGNI）

- **不做** vLLM↔SGLang 启动参数自动翻译（`serve_profile.args` 原样透传各引擎）。
- **不做** 多模态、PD 分离、专家并行等 SGLang 高级特性。
- **不做** 方案 2 的 `ServingBackend` 抽象接口（当前仅两引擎，属过度设计；留作未来加第三个引擎时的演进方向）。
- **不做** 实时可视化/看板；仅离线生成静态 png。

## 4. 设计

### 4.1 配置 schema 变更（向后兼容）

`auto_bench.py`：

- `ServeProfile` 新增 `engine: str = "vllm"`（可选；老配置不填即 vLLM，**现有 smoke 配置零修改可继续跑**）。
- `RunConfig` 新增 `images: dict[str, str] | None = None`；保留 `vllm_image`。
  - 回落规则：`images = images or {"vllm": vllm_image}`。
- 校验（在 `load_config` / 解析阶段，`ConfigError` 早失败）：
  - `engine ∈ {"vllm", "sglang"}`，否则报错。
  - `images` 必须覆盖配置中出现过的所有 `engine`，否则报错并指明缺失的引擎。
  - `vllm_image` 与 `images` 至少存在其一。

样例（同一次 run 内两引擎并存）：

```jsonc
{
  "run": {
    "name": "qwen2_5_1_5b_sglang_compare",
    "results_dir": "vllm_standalone_bench/results",
    "images": { "vllm": "009e4cb46541", "sglang": "lmsysorg/sglang:latest" },
    "bench_image": "vllm-bench-runner:offline",
    "network": "vllm-bench-net",
    "container_port": 8888,
    "api_key": "local-bench-key",
    "ready_timeout_sec": 1800,
    "cooldown_sec": 5
  },
  "serve_profiles": [
    { "name": "vllm_bf16",   "engine": "vllm",  "gpus": "all", "args": ["--dtype","bfloat16","--gpu-memory-utilization","0.70"] },
    { "name": "sglang_bf16", "engine": "sglang", "gpus": "all", "args": ["--dtype","bfloat16","--mem-fraction-static","0.70"] }
  ]
}
```

### 4.2 启动命令分派（`auto_bench.py`）

将 `build_vllm_run_command` 重构为 `build_serve_run_command(config, case, run_dir)`，按 `case.serve_profile.engine`（默认 `vllm`）分派。容器名、labels、`--network`、`-v models:/models:ro`、`--gpus` 等公共部分保持一致；`serve_profile.args` **原样追加**（不翻译）。

- **vLLM 分支**（与现有逻辑等价）：
  - `--entrypoint vllm`，镜像 `images["vllm"]`
  - `serve <model_path> --served-model-name <api_model_name> --host 0.0.0.0 --port <container_port> [--api-key ...] <args...>`

- **SGLang 分支**（容器内 model_path 与 vLLM 一致，复用 `/models` 挂载）：
  ```
  docker run -d --name <container_name>
    --label ...（同 vLLM）
    --gpus <gpus> --network <network> -v <models>:/models:ro
    --entrypoint python3 <images["sglang"]>
    -m sglang.launch_server
    --model-path <model_path>            # 容器内路径，如 /models/Qwen2.5-1.5B-Instruct
    --host 0.0.0.0 --port <container_port>
    --served-model-name <api_model_name>
    [--api-key <api_key>]
    <serve_profile.args...>
  ```
  - 实现时需验证所选 SGLang 官方镜像的 entrypoint/workdir，必要时用 `--entrypoint python3` 兜底（确保 `python -m sglang.launch_server` 可用）。
  - bench 容器仍通过 `http://<case.container_name>:<container_port>/v1` 发请求，与引擎无关。

### 4.3 就绪探测

复用现有 `wait_for_container_ready`（探测 `/v1`，SGLang 有 `/v1/models`，兼容）。**不改**。如需更快反馈，可作为可选增强改探 `/health`，但非必需。

### 4.4 对比聚合层（新文件 `vllm_standalone_bench/bench_compare.py`）

触发时机：`run_controller` 所有 serve group 跑完后调用一次（仅当出现 ≥2 个不同 `engine`，或 serve_profile 数 > 1 时）。

输入：`results/<run_id>/<serve_profile>/result.csv`（每引擎/profile 一份，由现有 `run_bench_multi.py` 产出）。

对齐键：`(bench_profile, input_len, output_len, parallel_num)`。

输出（全部写到 `results/<run_id>/`，与各 profile 子目录平级）：

- `compare.csv` / `compare.xlsx`：每个对齐键一行，关键指标（`ttft_mean/p50/p90/p99_ms`、`throughput_tok_s`、`input_throughput_tok_s`、`prefill_effective_tok_s`、`decode_effective_tok_s`）按引擎并排列（列名带引擎前缀，如 `sglang__ttft_p50_ms`）。
- `plots/<bench_profile>__<input_len>x<output_len>__ttft_vs_parallel.png`
- `plots/<bench_profile>__<input_len>x<output_len>__throughput_vs_parallel.png`
  - 横轴 `parallel_num`，每引擎一条线；TTFT 图含 p50/p90/p99 可选子图，throughput 图用 `throughput_tok_s`。

**铁律（强约束）**：`bench_compare.py` 只读各 `result.csv`，**永不修改或删除**任何原始 per-profile 的 `result.csv` / `result.xlsx`。

### 4.5 数据流

```
config(含 engine+images)
  → expand_cases          每个 case 带 engine
  → _group_cases_by_serve 每个 serve_profile = 一次启停（vllm 或 sglang）
     → build_serve_run_command(按 engine 分派)
     → 启动容器（复用 docker 编排/labels/network/cleanup）
     → wait_for_container_ready（/v1，两引擎都兼容）
     → run_bench_multi（压测层 0 改动；ignore_eos+max_tokens 两引擎都生效）
     → 停容器（复用 cleanup + cooldown）
  → bench_compare.aggregate（全部完成后：读各 result.csv → compare + 图表，原数据保留）
```

## 5. 错误处理

- 非法 `engine` / `images` 缺失引擎 → `ConfigError`，在解析阶段早失败。
- 某引擎容器启动失败或就绪超时 → 复用现有 `_record_skipped_group`，该引擎该组标记 `skipped`，**不阻塞另一引擎**的 group。
- `bench_compare` 聚合时某引擎缺 `result.csv` → 该引擎对应列填 `N/A`，不抛异常。
- 图表生成失败 → best-effort：catch + 在日志/manifest 记录，**不影响** `compare.csv/xlsx` 已生成的结果。
- 多引擎聚合的产物路径与各 profile 子目录隔离，避免覆盖。

## 6. 测试策略

- **单元测试**（`tests/`）：
  1. config 解析：`engine` 默认为 `vllm`；`images` 回落到 `{"vllm": vllm_image}`；非法 `engine`、`images` 缺引擎 → 报错。
  2. `build_serve_run_command`：对 `engine=vllm`/`sglang` 分别 dry-run，断言生成的命令字符串（sglang 分支含 `-m sglang.launch_server`、`--model-path`、`--host 0.0.0.0`、`--port`，且镜像取自 `images["sglang"]`）。
  3. `bench_compare.aggregate`：用两份 mock `result.csv`（同对齐键、不同引擎列）断言 `compare.csv` 行列对齐、缺失引擎列填 `N/A`，且**原始 mock 文件未被修改**（内容哈希前后一致）。
- **集成测试**：现有 `tests/test_integration.py` 增加一个 SGLang `--dry-run` 用例，断言 resolved 命令含 `sglang.launch_server`。
- **手动验收（需 GPU + 镜像，离线）**：用样例配置跑真实 smoke，确认 vLLM/SGLang 各自起服务、压测落盘，`compare.*` 与 `plots/*.png` 生成，原始 `result.csv` 保留。

## 7. 依赖与前置准备

- `vllm_standalone_bench/Dockerfile.bench-runner`：新增 `matplotlib`（pip 安装）；相应更新 `requirements`。
- README 新增章节：
  - SGLang 官方镜像离线搬运（`docker pull` → `docker save` → 离线机 `docker load`，与现有 vLLM 镜像同套路）。
  - vLLM↔SGLang 常用启动参数等价提示（**仅文档，代码不翻译**）：`--gpu-memory-utilization` ↔ `--mem-fraction-static`；`--tensor-parallel-size` ↔ `--tp-size`；`--max-model-len` ↔ `--context-length`。
- 新增样例配置：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json`（两引擎并存的对照配置）。

## 8. 验收标准

1. 现有 `auto_bench.qwen2_5_1_5b.smoke.json`（无 `engine`/`images`）`--dry-run` 行为不变（向后兼容）。
2. 新增 SGLang 样例配置 `--dry-run` 生成的 docker 启动命令正确（含 `sglang.launch_server`、正确镜像/端口/模型路径）。
3. `run_bench_multi.py` 对 SGLang 服务发请求，`ignore_eos=true` + `max_tokens` 生效，输出到达指定长度。
4. 同台 run 完成后产出 `compare.csv`/`compare.xlsx`/`plots/*.png`，且各引擎原始 `result.csv` 完整保留。
5. 单元 + 集成测试全部通过；`pytest -q` 绿。

## 9. 风险

- **SGLang 官方镜像 entrypoint 未知**：缓解——`--entrypoint python3` 兜底；实现时实测确认。
- **SGLang `ignore_eos` 在 chat/completions 的字段透传**：源码已确认协议层支持（`protocol.py:356/728`），但需在真实 smoke 中验证端到端输出长度。
- **指标口径差异**：vLLM 与 SGLang 的 `usage` 字段统计口径可能略有差异（如是否含首 token）。本工程沿用现有口径，差异在对比报告中如实呈现，不做归一化。
- **离线 matplotlib 依赖**：需确保 bench-runner 镜像构建时联网装好，再搬运；绘图失败不阻断主流程。

## 10. 涉及文件清单

- 改：`vllm_standalone_bench/auto_bench.py`（schema、`build_serve_run_command` 分派、`run_controller` 末尾接入聚合）
- 新：`vllm_standalone_bench/bench_compare.py`（对比聚合 + 绘图）
- 改：`vllm_standalone_bench/Dockerfile.bench-runner`（+ matplotlib）
- 新：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json`
- 改：`vllm_standalone_bench/README.md`（SGLang 镜像搬运、参数等价、对比用法）
- 改/新：`vllm_standalone_bench/tests/`（config、命令分派、聚合的单测 + 集成 dry-run 用例）
