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
| Ramp-up 策略 | ✅ | ❌ |
| Burstiness (Gamma) | ✅ | ❌ |
| Speculative Decoding 指标 | ✅ | ❌ |
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

默认使用 Docker bridge network `vllm-bench-net`，不使用 `--network host`，也不暴露主机端口。控制器只会清理本次自动创建并带有本次运行标签的 Docker network。

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

## 与 benchmark_tools 的关键差异

1. **TPOT 分母**：本工具用 `output_len - 1`（排除首 token），benchmark_tools 用 `output_len`
2. **ITL**：本工具完整计算，benchmark_tools 缺失
3. **统计维度**：本工具提供 mean/median/std + 完整分位数，benchmark_tools 仅提供均值
4. **吞吐含义**：本工具计算全局系统吞吐，benchmark_tools 计算单请求平均速率
