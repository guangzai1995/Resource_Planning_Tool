# vLLM 性能基准测试工具

从 `vllm-main/vllm/benchmarks/serve.py` 提取核心指标计算逻辑，**无需安装 vllm 包**即可运行。

## 🚀 快速开始

```bash
# 单次 serve benchmark 帮助
python3 vllm_standalone_bench/run_bench_serve.py --help

# 批量矩阵 benchmark 帮助
python3 vllm_standalone_bench/run_bench_multi.py --help

# 离线自动压测控制器帮助
python3 vllm_standalone_bench/auto_bench.py --help

# 检查 smoke 自动压测将执行的 Docker 命令
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json \
  --run-id docs_dry_run \
  --dry-run
```

## 与 vllm bench serve 的对齐程度

| 能力 | vllm bench serve | 本工具 |
|---|---|---|
| TTFT/TPOT/ITL/E2EL 分位数统计 | ✅ | ✅ |
| Goodput (SLO 约束) | ✅ | ✅ |
| Peak 吞吐 / 并发 | ✅ | ✅ |
| OpenAI Completions API | ✅ | ✅ |
| OpenAI Chat Completions API | ✅ | ✅ |
| Poisson 请求调度 | ✅ | ✅ |
| 并发限制 | ✅ | ✅ |
| 预热 | ✅ | ✅ |
| 端点就绪检查 | ✅ | ✅ |
| Random 数据集 | ✅ | ✅ |
| ShareGPT 数据集 | ✅ | ✅（基础） |
| Built-in MTP Chat 数据集 | ❌ | ✅ |
| Ramp-up 策略 | ✅ | ❌ |
| Burstiness (Gamma) | ✅ | ❌ |
| Speculative Decoding 指标 | ✅ | ✅（/metrics 差分） |
| 多模态 | ✅ | ❌ |
| Timeline Plot | ✅ | ❌ |

## 安装

```bash
pip install -r requirements.txt
```

**不需要安装 vllm！** 仅依赖 `aiohttp`、`numpy`、`tqdm`。

## 离线双镜像自动化压测

`auto_bench.py` 是离线自动化压测入口。主机侧只需要 Docker CLI、可用的 GPU container runtime 和 Python 3；benchmark 运行依赖封装在 `vllm-bench-runner:offline` 镜像中，不需要在主机安装 `openpyxl`、ModelScope 或 benchmark Python 依赖。

### 构建和搬运 bench-runner

在联网环境构建 benchmark runner 镜像：

```bash
docker build \
  -f vllm_standalone_bench/Dockerfile.bench-runner \
  -t vllm-bench-runner:offline \
  vllm_standalone_bench

docker save vllm-bench-runner:offline -o vllm-bench-runner.offline.tar
```

在离线环境导入 bench-runner 和 vLLM 镜像：

```bash
docker load -i vllm-bench-runner.offline.tar
docker load -i vllm.offline.tar
```

示例配置默认使用 vLLM 镜像 ID `009e4cb46541`，并使用 `vllm-bench-runner:offline` 发起压测。

### 准备 Qwen smoke 模型

联网或可访问 ModelScope 缓存的环境中，使用 bench-runner 将模型准备到计划路径：

```bash
python3 vllm_standalone_bench/auto_bench.py prepare-model \
  --modelscope-id Qwen/Qwen2.5-1.5B-Instruct \
  --target /Resource_Planning_Tool/model/Qwen2.5-1.5B-Instruct \
  --bench-image vllm-bench-runner:offline
```

配置中的容器内模型路径保持为 `/models/Qwen2.5-1.5B-Instruct`。

### 前台运行

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json
```

### 后台运行和控制

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json \
  --detach

python3 vllm_standalone_bench/auto_bench.py status \
  --results-dir vllm_standalone_bench/results \
  --run-id <run_id>

python3 vllm_standalone_bench/auto_bench.py logs \
  --results-dir vllm_standalone_bench/results \
  --run-id <run_id> \
  --follow

python3 vllm_standalone_bench/auto_bench.py stop \
  --results-dir vllm_standalone_bench/results \
  --run-id <run_id>
```

默认使用 Docker bridge network `vllm-bench-net`，不使用 `--network host`，也不暴露主机端口。控制器只会清理本次自动创建并带有本次运行标签或元数据的资源，包括 vLLM 容器和 Docker network；`stop` 会请求后台控制器优雅退出并执行这些清理。

### 当前主机 smoke 验证

先用 dry-run 检查将要执行的 Docker 命令，不启动容器：

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json \
  --run-id docs_dry_run \
  --dry-run
```

dry-run 会写入 `vllm_standalone_bench/results/docs_dry_run/config.resolved.json` 作为 resolved config preview，但不会写 manifest、state 或 benchmark 结果文件。

准备好 vLLM 镜像、bench-runner 镜像和完整模型后，运行真实 smoke：

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json
```

## vLLM / SGLang 同台对比

通过 `serve_profiles` 的 `engine` 字段与 `run.images` 映射，可在同一次 run 内分别启动 vLLM 与 SGLang 做对比。样例：`configs/auto_bench.qwen2_5_1_5b.sglang_compare.json`。

### SGLang 镜像离线搬运

```bash
# 联网机
docker pull lmsysorg/sglang:latest
docker save lmsysorg/sglang:latest -o sglang.offline.tar
# 离线机
docker load -i sglang.offline.tar
```

### 常用启动参数等价（仅文档参考，配置里 `args` 原样透传，不自动翻译）

| 用途 | vLLM | SGLang |
|---|---|---|
| 显存占用比例 | `--gpu-memory-utilization` | `--mem-fraction-static` |
| 张量并行 | `--tensor-parallel-size` | `--tp-size` |
| 最大上下文 | `--max-model-len` | `--context-length` |

run 结束后在 `results/<run_id>/` 产出 `compare.csv`、`compare.xlsx` 与 `plots/*.png`，各引擎原始 `result.csv` 保留在 `<model>/<serve_profile>/<bench_profile>/` 子目录。

### 固定并发预热（消除小并发档 TTFT 首批尖峰）

`bench_profile` 可选字段：
- `warmup_concurrency`：warmup 固定并发数（默认 `null`=跟随该档并发）
- `warmup_output_len`：warmup 请求输出长度（默认 `null`=跟随该档输出）

整个测试仅在首个配置前预热一次：用固定并发 × 首个配置输入 × 指定输出长度，
凑齐一波满并发，热起 vLLM 多请求调度路径。`smoke` / `sglang_compare` 配置
默认 `warmup_concurrency=4`、`warmup_output_len=128`。CLI 直跑可用
`--warmup-concurrency 4 --warmup-output-len 128`。

### MTP 真实风格内置数据集

`bench_profiles[].dataset.name = "builtin_mtp_chat"` 会启用离线内置的 chat-style
MTP prompt 数据集。该数据集用真实任务风格的中文/英文技术场景构造 prompt，适合替代随机
token prompt 来观察 MTP/Spec Decode 接受率。

```json
"dataset": {
  "name": "builtin_mtp_chat",
  "length_policy": "bucket",
  "input_len_tolerance": 0.2,
  "on_bucket_shortage": "error",
  "sampling": "shuffle"
}
```

启用该数据集后，`input_lens` 不会失效，而是作为 prompt token bucket 目标。例如
`input_lens: [4096]` 且 `input_len_tolerance: 0.2` 表示选择约 3276 到 4915 token
的真实风格 prompt。`output_lens`、`parallel_nums`、`epochs`、`cross_product` 保持
现有矩阵语义。

注意事项：
- `builtin_mtp_chat` 需要模型 tokenizer，配置中必须提供 `models[].tokenizer_path`。
- 数据集只负责 prompt；MTP 本身仍通过 `serve_profiles[].args` 配置，例如
  `--speculative-config.method mtp` 和 `--speculative-config.num_speculative_tokens 1`。
- 自动化配置会拒绝漏写前导 `--` 的 `speculative-config.*` 参数。
- 结果表会导出 `spec_decode_acceptance_rate`、`spec_decode_system_efficiency`、
  `spec_decode_num_drafts`、`spec_decode_num_accepted_tokens`、
  `spec_decode_num_draft_tokens` 和 `spec_decode_per_position_acceptance_rates`。

示例配置见
`vllm_standalone_bench/configs/auto_bench.mtp_builtin_dataset.example.json`。

### vLLM 编译/JIT cache 持久化

GLM5.2 这类 DSA/MoE 模型首次启动会触发 torch.compile、AOT、Triton/Inductor、
DeepGEMM JIT 和 FlashInfer autotune。默认情况下这些缓存位于 vLLM 容器内，
auto_bench 停止并删除容器后会丢失。需要反复运行同一套 benchmark 时，可以启用
`run.vllm_cache`：

```json
"run": {
  "vllm_cache": {
    "enabled": true,
    "container_path": "/vllm-cache",
    "set_default_env": true
  }
}
```

`root` 可省略；启用 cache 且省略 `root` 时，默认使用配置文件所在目录下的
`.cache/vllm_auto_bench`。也可以显式配置绝对路径或相对路径，相对路径按配置文件所在
目录解析。

启用后，vLLM serving 容器会挂载 `<root>/<cache_key>:/vllm-cache:rw`，并自动设置：

- `VLLM_CACHE_ROOT=/vllm-cache`
- `DG_JIT_CACHE_DIR=/vllm-cache/deep_gemm`
- `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/vllm-cache/flashinfer_autotune`

建议为 GLM5.2 正式 profile 显式配置稳定 `cache_key`：

```json
"serve_profiles": [{
  "name": "glm52_fp8_tp8_o2",
  "engine": "vllm",
  "cache_key": "glm52-fp8-tp8-h20-o2",
  "gpus": "all",
  "args": ["--tensor-parallel-size", "8", "--kv-cache-dtype", "fp8"]
}]
```

第一次运行仍会完整编译和 JIT；后续相同镜像、模型、GPU 架构、TP、dtype 和 serve args
应复用 cache。不要让不同硬件或不同 serve 参数共享同一个 `cache_key`。cache 目录不会随
run 清理，可手动删除 `.cache/vllm_auto_bench/<cache_key>` 释放磁盘空间。

默认 cache key 会随 `run.images.vllm` 镜像引用字符串、`model.name`、`model_path`、
`tokenizer_path`、`served_model_name`、serve profile 名称、`gpus` 和 `serve args`
变化。正式 GLM5.2 benchmark 仍建议显式配置稳定 `cache_key`，把模型、GPU 架构、TP、
dtype、优化级别和关键环境口径写进名字，便于人工审计和跨机器协作。

注意可变镜像 tag 的风险：默认 cache key 只看配置里的镜像引用字符串，不会自动知道 tag
背后的 image id 或 digest；`latest`、`offline` 等 tag 可能在底层镜像变更后仍保持相同
引用。显式 `cache_key` 也会绕过默认 fingerprint。使用可变 tag 时，建议把
`run.images.vllm` 改成稳定的 image id 或 digest，或把 image id、digest、构建号纳入
`cache_key`。升级 vLLM 镜像、GPU 架构、TP、dtype 或关键 serve args 后，应更换
`cache_key` 或清理对应 cache 目录，避免复用不兼容的编译/JIT cache。

## 使用方法

### 基本用法（Random 数据集）

```bash
python3 vllm_standalone_bench/run_bench_serve.py \
    --backend openai \
    --host 127.0.0.1 \
    --port 8000 \
    --model your-model-name \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate 10
```

### Chat Completions 端点

```bash
python3 vllm_standalone_bench/run_bench_serve.py \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --host 127.0.0.1 \
    --port 8000 \
    --model your-model-name \
    --dataset-name random \
    --num-prompts 50 \
    --request-rate 5
```

### 带 Goodput SLO 约束

```bash
python3 vllm_standalone_bench/run_bench_serve.py \
    --backend openai \
    --host 127.0.0.1 --port 8000 \
    --model your-model \
    --num-prompts 200 \
    --request-rate 20 \
    --goodput ttft:500 tpot:100 e2el:5000
```

### 保存结果

```bash
python3 vllm_standalone_bench/run_bench_serve.py \
    --backend openai \
    --host 127.0.0.1 --port 8000 \
    --model your-model \
    --num-prompts 100 \
    --request-rate 10 \
    --save-result \
    --result-dir ./results
```

## 指标说明

| 指标 | 含义 | 计算方式 |
|---|---|---|
| **TTFT** | 首 Token 时延 | `首个 token 到达时间 - 请求发出时间` |
| **TPOT** | 平均输出 Token 时延 | `(E2E - TTFT) / (output_tokens - 1)` |
| **ITL** | Token 间延迟 | `token[i+1] 时刻 - token[i] 时刻` |
| **E2EL** | 端到端延迟 | `最后一个 token 时刻 - 请求发出时刻` |
| **Goodput** | 有效吞吐 | 满足所有 SLO 约束的请求数 / 总时间 |
| **Peak tok/s** | 峰值输出吞吐 | 按秒桶统计的最大每秒输出 token 数 |
| **throughput_tok_s** | 输出 Token 系统吞吐 | `total_output_tokens / benchmark_duration`，保留 vLLM 官方 `output_throughput` 口径 |
| **input_throughput_tok_s** | 输入 Token 系统吞吐 | `total_input_tokens / benchmark_duration` |
| **prefill_effective_tok_s** | Prefill 有效速率 | `avg_input_tokens / mean_TTFT_s`，TTFT 包含排队、调度和首 token |
| **decode_effective_tok_s** | Decode 有效速率 | `1 / mean_TPOT_s`，基于 TPOT 的 next-token decode 近似速率 |
| **cache_hit_rate** | 缓存命中率 | `avg_cached_tokens / avg_input_tokens`，统计输入 tokens 中从缓存命中的比例 |
| **avg_cached_tokens** | 平均缓存 Token 数 | 每个请求平均从 prefix cache 命中的 token 数 |

> **缓存命中率（`cache_hit_rate`）**：统计结果中的 `cache_hit_rate` / `avg_cached_tokens`
> 取自响应 `usage.cached_tokens`。该值非零需要服务端开启前缀缓存
> （vLLM `--enable-prefix-caching`；SGLang 对应缓存开关）；未开启时命中率为 0。

## 与 benchmark_tools 的关键差异

1. **TPOT 分母**：本工具和当前 benchmark_tools 都按 `(E2E - TTFT) / (output_tokens - 1)` 计算，排除首 token
2. **ITL**：本工具完整计算，benchmark_tools 缺失
3. **统计维度**：本工具提供 mean/median/std + 完整分位数，benchmark_tools 仅提供均值
4. **吞吐展示**：本工具同时展示系统吞吐和阶段有效速率，benchmark_tools 主要展示输出 Token 系统吞吐与延迟均值
