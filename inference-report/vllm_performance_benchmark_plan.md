# vLLM 推理框架性能测试方案

> **模型**: Qwen3-35B-A3B（MoE 架构，35B 总参数 / 3B 激活参数）
> **测试 GPU**: NVIDIA A40 (48GB) / NVIDIA H2x (144GB)
> **推理框架**: vLLM (vllm-main)
> **测试工具**: vllm_standalone_bench
> **文档版本**: v1.1 | 2026-06-08

---

## 目录

1. [测试目标与范围](#1-测试目标与范围)
2. [测试环境](#2-测试环境)
3. [测试场景设计](#3-测试场景设计)
4. [测试矩阵](#4-测试矩阵)
5. [各专项测试方案](#5-各专项测试方案)
6. [指标体系](#6-指标体系)
7. [执行流程](#7-执行流程)
8. [预期产出](#8-预期产出)

---

## 1. 测试目标与范围

### 1.1 测试目标

针对 **Qwen3-35B-A3B** MoE 模型，在 A40 和 H2x 两种 GPU 上，系统性评估 vLLM 推理框架在以下六大核心特性下的性能表现：

| 编号 | 特性 | 核心关注点 |
|------|------|-----------|
| F1 | **连续批处理 (Continuous Batching)** | 吞吐量 vs 延迟的权衡，调度策略对并发能力的影响 |
| F2 | **KV Cache / Prefix Caching** | 缓存命中率对 TTFT 的加速效果，内存利用率优化 |
| F3 | **量化 (AWQ / FP8)** | 精度-性能-显存三角权衡，MoE 模型量化兼容性 |
| F4 | **投机解码 (Speculative Decoding)** | MTP 等投机方法对单请求延迟的降低效果 |
| F5 | **P/D 分离 (Disaggregated Prefill/Decode)** | 长上下文场景下 prefill 与 decode 解耦的收益 |
| F6 | **MOE 专家并行优化** | 专家负载均衡、All2All 通信效率、EP vs TP 对比 |

### 1.2 测试范围

- **输入场景覆盖**: Agent 长上下文 (8K–196K tokens)
- **并发梯度**: 低并发 (1–8)、中并发 (16)、高并发 (32)、极限并发 (64/128)
- **性能指标**: TTFT、TPOT、E2EL、吞吐量 (tok/s)

---

## 2. 测试环境

### 2.1 硬件配置

| 项目 | A40 配置 | H2x 配置 |
|------|----------|----------|
| **GPU 型号** | NVIDIA A40 | NVIDIA H2x |
| **显存** | 48 GB | 144 GB |
| **GPU 数量** | 2 / 4 (P/D 分离) | 1 / 2 (P/D 分离) |
| **CPU** | Intel Xeon Gold 6326 | Intel Xeon Gold 6326 |
| **内存** | 1024 GB DDR4 | 1024 GB DDR4 |
| **存储** | NVMe SSD | NVMe SSD |
| **网络** | 25GbE (单机测试) | 25GbE / NVLink (P/D 分离) |
| **CUDA 版本** | 12.4 | 12.4 |
| **驱动版本** | 550.x | 550.x |

### 2.2 软件环境

| 组件 | 版本/说明 |
|------|----------|
| **操作系统** | Ubuntu 22.04.5 LTS |
| **Python** | 3.10+ |
| **vLLM** | 最新 main 分支 (从 vllm-main 构建) |
| **PyTorch** | 2.5+ (CUDA 12.4) |
| **测试工具** | vllm_standalone_bench (独立版，无需安装 vllm) |
| **依赖** | aiohttp, numpy, tqdm, openpyxl |

### 2.3 模型信息

| 属性 | 值 |
|------|-----|
| **模型名称** | Qwen3-35B-A3B |
| **架构** | Mixture-of-Experts (MoE) |
| **总参数量** | 35B |
| **激活参数量** | 3B |
| **专家数量** | 64 experts, top-8 激活 |
| **上下文长度** | 262,144 tokens |
| **默认精度** | BF16 |

---

## 3. 测试场景设计

### 3.1 Agent 长上下文场景定义

Agent 场景的典型特征是**长系统提示 + 长工具调用历史 + 较短生成**，与传统对话场景有显著差异。测试覆盖从 8K 到 196K 的完整上下文梯度。

| 场景名称 | 输入长度 (tokens) | 输出长度 (tokens) | 说明 |
|----------|-------------------|-------------------|------|
| **Agent-Long** | 8,192 | 1,024 | 长对话历史 + 复杂工具链 |
| **Agent-XLong** | 16,384 | 1,024 | 长文档 + 多次工具交互 |
| **Agent-UltraLong** | 32,768 | 1,024 | 极长上下文 |
| **Code-Refactor** | 64,000 | 1,024 | 大规模代码重构场景 |
| **Doc-Analysis** | 96,000 | 1,024 | 长文档深度分析 |
| **Multi-Agent** | 128,000 | 1,024 | 多 Agent 协作长上下文 |
| **Extreme-160K** | 160,000 | 1,024 | 极端长上下文压力测试 |
| **Extreme-196K** | 196,000 | 1,024 | 接近模型上下文极限 |

### 3.2 并发梯度设计

| 梯度 | 并发数 | 适用场景 |
|------|--------|----------|
| **单用户** | 1 | 单请求延迟基准 |
| **低并发** | 4 | 小规模 Agent 部署 |
| **低-中并发** | 8 | 中小规模 Agent 服务 |
| **中并发** | 16 | 中等负载 Agent 服务 |
| **高并发** | 32 | 生产级 Agent 服务 |
| **极限并发** | 64 / 128 | 压力测试，寻找性能拐点 |

### 3.3 前缀缓存共享比例

| 比例 | 说明 | 适用场景 |
|------|------|----------|
| 0.0 | 无共享前缀 (基准) | 每个请求完全独立 |
| 0.5 | 50% 前缀共享 | Agent 共享系统提示 |
| 0.8 | 80% 前缀共享 | Agent 共享系统提示 + 工具定义 |

---

## 4. 测试矩阵

### 4.1 总体测试矩阵

```
特性 (6) × GPU (2) × 场景 (8) × 并发 (6) × 重复轮次 (3) = 1,728 组测试
每组请求数 = 并发数 × EPOCHS (3) = 并发数 × 3
```

### 4.2 分特性测试规模

| 特性 | 核心变化参数 | 测试组合数 | 预估耗时 |
|------|-------------|-----------|----------|
| F1 连续批处理 | 调度策略、max_num_seqs、chunked_prefill | ~180 | 2h |
| F2 KV Cache/Prefix | prefix_ratio、block_size、kv_cache_dtype | ~120 | 1.5h |
| F3 量化 | AWQ、FP8、BF16 基准 | ~120 | 2h |
| F4 投机解码 | MTP、ngram、spec_tokens 数量 | ~100 | 1.5h |
| F5 P/D 分离 | 1P1D、2P1D、1P2D 拓扑 | ~90 | 1.5h |
| F6 MOE 专家并行 | EP vs TP、EPLB、All2All 后端 | ~120 | 2h |
| **总计** | | **~730** | **~11h** |

---

## 5. 各专项测试方案

---

### 5.1 F1: 连续批处理 (Continuous Batching)

#### 5.1.1 测试原理

vLLM 的连续批处理（Iteration-level Batching）允许在每个解码迭代中动态插入新请求、移除已完成请求，避免传统静态批处理的资源浪费。关键参数：

- `max_num_batched_tokens`: 单次迭代最大 token 数（默认 2048）
- `max_num_seqs`: 单次迭代最大序列数（默认 128）
- `enable_chunked_prefill`: 将长 prefill 分块处理，避免阻塞 decode
- `scheduling_policy`: 调度策略 (fcfs / priority)

#### 5.1.2 测试用例

| 用例编号 | 变量 | 测试值 | 固定参数 |
|----------|------|--------|----------|
| F1-01 | max_num_seqs | 16, 32, 64, 128 | chunked_prefill=True |
| F1-02 | max_num_batched_tokens | 2048, 4096, 8192, 16384 | max_num_seqs=128 |
| F1-03 | Chunked Prefill 开关 | True / False | max_num_seqs=128 |
| F1-04 | max_num_partial_prefills | 1, 2, 4 | chunked_prefill=True |
| F1-05 | 长上下文调度 | Agent-Long (8K) + 高并发 | 对比 chunked vs non-chunked |

#### 5.1.3 vLLM 启动命令示例

```bash
# F1-01: max_num_seqs=64
vllm serve /path/to/Qwen3-35B-A3B \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --port 8000

# F1-03: 关闭 Chunked Prefill (对比)
vllm serve /path/to/Qwen3-35B-A3B \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 \
    --port 8001
```

#### 5.1.4 测试脚本配置

```bash
# run_bench.sh 配置 (F1 测试)
INPUT_LENS="512 2048 8192"
OUTPUT_LENS="128 256 512"
PARALLEL_NUMS="1 4 16 32 64"
EPOCHS=3
PREFIX_RATIO=0.0        # 不使用前缀缓存，纯测试批处理效果
SLEEP_BETWEEN=3.0
MAX_TTFT_MS=10000       # 长上下文允许更长 TTFT
MIN_THROUGHPUT_TOK_S=5
```

#### 5.1.5 分析重点

1. **吞吐量-延迟曲线**: 绘制不同 `max_num_seqs` 下的吞吐量 (tok/s) vs P99 TTFT 曲线
2. **Chunked Prefill 收益**: 对比 Agent-Long 场景下开启/关闭 chunked prefill 的 TTFT 和 TPOT 差异
3. **GPU 利用率**: 通过 `nvidia-smi` 监控不同配置下的 GPU 计算利用率

---

### 5.2 F2: KV Cache / Prefix Caching

#### 5.2.1 测试原理

Prefix Caching 利用请求间的共享前缀（如系统提示、工具定义）复用 KV Cache，显著降低重复 prefill 的计算开销。关键参数：

- `enable_prefix_caching`: 启用前缀缓存（默认 True）
- `prefix_caching_hash_algo`: 哈希算法 (sha256 / xxhash)
- `block_size`: KV Cache 块大小（默认 16 tokens）
- `gpu_memory_utilization`: GPU 显存分配比例（默认 0.92）
- `kv_cache_dtype`: KV Cache 数据类型 (auto / fp8_e5m2 / fp8_e4m3)

#### 5.2.2 测试用例

| 用例编号 | 变量 | 测试值 | 说明 |
|----------|------|--------|------|
| F2-01 | Prefix Caching 开关 | 开 / 关 | Agent-Long 场景，前缀共享比例 0.8 |
| F2-02 | 前缀共享比例 | 0.0, 0.5, 0.8 | 固定 input_len=8192 |
| F2-03 | KV Cache dtype | auto (BF16), fp8_e5m2, fp8_e4m3 | 节省显存但可能损失精度 |
| F2-04 | hash 算法 | sha256, xxhash | 性能差异（xxhash 更快） |

#### 5.2.3 vLLM 启动命令示例

```bash
# F2-01: 启用 Prefix Caching
vllm serve /path/to/Qwen3-35B-A3B \
    --enable-prefix-caching \
    --prefix-caching-hash-algo xxhash \
    --block-size 16 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype auto \
    --port 8000

# F2-03: FP8 KV Cache (节省显存)
vllm serve /path/to/Qwen3-35B-A3B \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e5m2 \
    --gpu-memory-utilization 0.92 \
    --port 8001
```

#### 5.2.4 测试脚本配置

```bash
# run_bench.sh 配置 (F2 测试)
INPUT_LENS="2048 8192 16384"
OUTPUT_LENS="256 512"
PARALLEL_NUMS="1 4 16 32"
EPOCHS=3
PREFIX_RATIO=0.8        # 80% 前缀共享
TOKENIZER="/path/to/Qwen3-35B-A3B"
SLEEP_BETWEEN=5.0
```

#### 5.2.5 分析重点

1. **缓存命中加速比**: 对比 prefix_ratio=0.0 vs 0.8 时的 TTFT 降低比例
2. **前缀长度 vs 加速比**: 不同 input_len 下，前缀缓存的边际收益递减曲线
3. **KV Cache 显存效率**: FP8 vs BF16 KV Cache 的显存节省比例 vs 精度损失
4. **高并发下的缓存争用**: 32+ 并发时 prefix caching 的实际命中率和收益

---

### 5.3 F3: 量化 (AWQ / FP8)

#### 5.3.1 测试原理

量化通过降低权重和/或激活值的精度来减少显存占用和提升计算吞吐量。对于 Qwen3-35B-A3B MoE 模型：

- **AWQ (4-bit)**: 权重量化到 INT4，大幅减少显存，适合显存受限场景
- **FP8 (8-bit)**: 权重/激活值量化到 FP8，H2x 原生支持 FP8 加速

#### 5.3.2 测试用例

| 用例编号 | 量化方案 | 说明 | 适用 GPU |
|----------|----------|------|----------|
| F3-01 | **BF16 基准** (无量化) | 全精度基准线 | A40, H2x |
| F3-02 | **AWQ (W4A16)** | 4-bit 权重 + BF16 激活 | A40, H2x |
| F3-03 | **FP8 per-tensor (W8A8)** | 全层 FP8 静态量化 | H2x (原生 FP8) |
| F3-04 | **KV Cache FP8 + 权重 FP8** | 全链路 FP8 | H2x |

#### 5.3.3 模型准备

```bash
# AWQ 模型需要预先量化（若无预量化版本）
python -m awq.quantize \
    --model_path /path/to/Qwen3-35B-A3B \
    --quant_path /path/to/Qwen3-35B-A3B-AWQ \
    --w_bit 4 --q_group_size 128 \
    --calib_data pileval --calib_size 128

# FP8 模型可直接加载（vLLM 运行时动态量化）
```

#### 5.3.4 vLLM 启动命令示例

```bash
# F3-01: BF16 基准
vllm serve /path/to/Qwen3-35B-A3B \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --port 8000

# F3-02: AWQ 量化
vllm serve /path/to/Qwen3-35B-A3B-AWQ \
    --quantization awq \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --port 8001

# F3-03: FP8 per-tensor
vllm serve /path/to/Qwen3-35B-A3B \
    --quantization fp8 \
    --kv-cache-dtype fp8_e5m2 \
    --gpu-memory-utilization 0.92 \
    --port 8002
```

#### 5.3.5 测试脚本配置

```bash
# run_bench.sh 配置 (F3 测试)
INPUT_LENS="512 2048 8192"
OUTPUT_LENS="128 256 512"
PARALLEL_NUMS="1 4 16 32 64"
EPOCHS=3
PREFIX_RATIO=0.0        # 不使用前缀缓存，专注量化效果
SLEEP_BETWEEN=3.0
```

#### 5.3.6 分析重点

1. **精度-性能权衡**: 量化方案 vs BF16 的吞吐量提升比例和延迟变化
2. **显存节省**: AWQ (4-bit) vs FP8 vs BF16 的显存占用对比
3. **A40 vs H2x 差异**: FP8 在 A40 上无硬件加速 vs H2x 上原生支持的性能差距
4. **长上下文稳定性**: 量化后 Agent-UltraLong (32K+) 场景的输出质量是否退化

---

### 5.4 F4: 投机解码 (Speculative Decoding)

#### 5.4.1 测试原理

投机解码通过模型自带的多 token 预测能力或轻量方法快速生成多个候选 token，再由大模型一次性验证，降低单请求延迟。本测试重点关注：

- **MTP (Multi-Token Prediction)**: 模型自带的多 token 预测能力，无额外模型开销
- **ngram**: 基于 n-gram 匹配的投机，零额外模型开销

#### 5.4.2 测试用例

| 用例编号 | 投机方法 | 配置 | 说明 |
|----------|----------|------|------|
| F4-01 | **无投机** (基准) | — | 对比基准线 |
| F4-02 | **MTP** | num_speculative_tokens=3 | 模型自带多 token 预测 |
| F4-03 | **MTP 不同步数** | num_speculative_tokens=2, 3, 4, 5 | 寻找最优投机步数 |
| F4-04 | **ngram** | prompt_lookup_max=5, num_speculative_tokens=5 | 轻量级投机 |
| F4-05 | **MTP + 长上下文** | Agent-Long (8K) / Agent-UltraLong (32K) | 长上下文下 MTP 效果 |

#### 5.4.3 vLLM 启动命令示例

```bash
# F4-02: MTP 投机解码
vllm serve /path/to/Qwen3-35B-A3B \
    --num-speculative-tokens 3 \
    --gpu-memory-utilization 0.90 \
    --port 8000

# F4-04: ngram 投机解码
vllm serve /path/to/Qwen3-35B-A3B \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5}' \
    --gpu-memory-utilization 0.90 \
    --port 8001
```

#### 5.4.4 测试脚本配置

```bash
# run_bench.sh 配置 (F4 测试 - 重点关注低并发延迟)
INPUT_LENS="512 2048 8192"
OUTPUT_LENS="128 256 512"
PARALLEL_NUMS="1 4 8 16"     # 投机解码在低并发下效果最佳
EPOCHS=5                      # 多轮测试以稳定接受率统计
PREFIX_RATIO=0.0
SLEEP_BETWEEN=3.0
```

#### 5.4.5 分析重点

1. **TPOT 加速比**: MTP vs 基准的 TPOT 降低比例
2. **接受率 / 接受长度**: MTP 不同投机步数的实际 token 接受率和平均接受长度
3. **并发退化**: 随并发数增加，投机解码的收益是否递减
4. **长上下文效果**: Agent-Long (8K+) 场景下投机解码的有效性

---

### 5.5 F5: P/D 分离 (Disaggregated Prefill/Decode)

#### 5.5.1 测试原理

P/D 分离将推理过程拆分为 Prefill（计算密集型）和 Decode（访存密集型）两个阶段，分别在不同 GPU 上执行。优势：

- **消除 Prefill/Decode 干扰**: 长上下文 prefill 不会阻塞正在 decode 的请求
- **独立扩缩容**: Prefill 和 Decode 节点可按需独立扩缩
- **长上下文优化**: Prefill 节点可配置更大的 `max_num_batched_tokens`

关键配置 (`KVTransferConfig`):
- `kv_connector`: KV 缓存传输后端
- `kv_role`: `kv_producer` (Prefill) / `kv_consumer` (Decode)
- `kv_rank`: 实例排名

#### 5.5.2 测试用例

| 用例编号 | 拓扑 | Prefill 配置 | Decode 配置 | 说明 |
|----------|------|-------------|-------------|------|
| F5-01 | **单体** (基准) | — | — | 1 GPU 完成 P+D |
| F5-02 | **1P1D** | 1×H2x (Prefill) | 1×H2x (Decode) | 基础 P/D 分离 |
| F5-03 | **1P1D** | 1×A40 (Prefill) | 1×A40 (Decode) | A40 上的 P/D 分离 |
| F5-04 | **2P1D** | 2×H2x (Prefill) | 1×H2x (Decode) | Prefill 扩展 |
| F5-05 | **1P2D** | 1×H2x (Prefill) | 2×H2x (Decode) | Decode 扩展 |
| F5-06 | **1P1D + 长上下文** | 1×H2x | 1×H2x | Agent-UltraLong (32K) |

#### 5.5.3 vLLM 启动命令示例

```bash
# Prefill 节点 (Producer)
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3-35B-A3B \
    --kv-transfer-config '{"kv_connector":"TorchDistributedConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"192.168.1.100","kv_port":14579}' \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.90 \
    --port 8000

# Decode 节点 (Consumer)
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/Qwen3-35B-A3B \
    --kv-transfer-config '{"kv_connector":"TorchDistributedConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_ip":"192.168.1.100","kv_port":14579}' \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --port 8001
```

#### 5.5.4 测试脚本配置

```bash
# run_bench.sh 配置 (F5 测试 - 连接到 Decode 节点)
BASE_URL="http://<decode_node_ip>:8001"
INPUT_LENS="8192 32768 64000 128000"
OUTPUT_LENS="256 512 1024"
PARALLEL_NUMS="1 4 16 32"
EPOCHS=3
PREFIX_RATIO=0.8
SLEEP_BETWEEN=5.0
```

#### 5.5.5 分析重点

1. **Prefill 隔离效果**: P/D 分离后 Decode 阶段的 TPOT 稳定性（无 Prefill 干扰）
2. **长上下文 TTFT**: 32K+ 输入下，单体 vs P/D 分离的 TTFT 差异
3. **KV 传输开销**: Prefill → Decode 的 KV Cache 传输延迟占比
4. **拓扑扩展性**: 1P1D → 2P1D → 1P2D 的吞吐量线性扩展比
5. **A40 vs H2x**: 不同 GPU 上 P/D 分离的适用性差异

---

### 5.6 F6: MOE 专家并行优化

#### 5.6.1 测试原理

Qwen3-35B-A3B 是 MoE 架构，拥有 64 个专家、每次激活 top-8。vLLM 提供多种 MOE 优化策略：

- **张量并行 (TP)**: 传统方式，将每个专家的权重按列切分到多 GPU
- **专家并行 (EP)**: 将不同专家分配到不同 GPU，减少通信量
- **专家负载均衡 (EPLB)**: 动态调整专家分布，平衡各 GPU 负载
- **All2All 通信**: 专家并行的核心通信原语，多种后端可选

#### 5.6.2 测试用例

| 用例编号 | 并行策略 | 配置 | 说明 |
|----------|----------|------|------|
| F6-01 | **TP=1** (单卡基准) | tensor_parallel_size=1 | 单卡无并行 |
| F6-02 | **TP=2** | tensor_parallel_size=2 | 传统张量并行 |
| F6-03 | **EP=2** | enable_expert_parallel, tp=1, dp=2 | 专家并行 |
| F6-04 | **EP=2 + EPLB** | enable_eplb, window_size=1000 | 带负载均衡的 EP |
| F6-05 | **EP=2 + round_robin** | expert_placement_strategy=round_robin | 轮询分布 |
| F6-06 | **All2All: allgather** | all2all_backend=allgather_reducescatter | 默认后端 |
| F6-07 | **All2All: deepep** | all2all_backend=deepep_high_throughput | DeepEP 高吞吐 |
| F6-08 | **DP=2** | data_parallel_size=2 | 数据并行 |
| F6-09 | **EP=2 vs TP=2** | 对比两种并行策略 | 总 GPU 数相同 |

#### 5.6.3 vLLM 启动命令示例

```bash
# F6-01: 单卡基准
vllm serve /path/to/Qwen3-35B-A3B \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --port 8000

# F6-03: EP=2 (专家并行)
vllm serve /path/to/Qwen3-35B-A3B \
    --enable-expert-parallel \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --port 8000

# F6-04: EP=2 + EPLB (带负载均衡)
vllm serve /path/to/Qwen3-35B-A3B \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"window_size":1000,"step_interval":3000,"use_async":true}' \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

#### 5.6.4 测试脚本配置

```bash
# run_bench.sh 配置 (F6 测试)
INPUT_LENS="512 2048 8192"
OUTPUT_LENS="128 256 512"
PARALLEL_NUMS="1 4 16 32 64"
EPOCHS=3
PREFIX_RATIO=0.0
SLEEP_BETWEEN=5.0
```

#### 5.6.5 分析重点

1. **EP vs TP 吞吐量对比**: 相同 GPU 数下，EP 和 TP 的吞吐量差异
2. **EPLB 效果**: 开启 EPLB 前后的负载均衡度和吞吐量变化
3. **All2All 通信开销**: 不同 All2All 后端的通信延迟占比
4. **专家负载热力图**: 各专家被激活的频率分布，识别热点专家
5. **并发扩展性**: EP 模式下从 4 并发到 64 并发的吞吐量扩展曲线

---

## 6. 指标体系

### 6.1 核心性能指标

| 指标 | 全称 | 含义 | 目标值参考 |
|------|------|------|-----------|
| **TTFT** | Time To First Token | 首 token 延迟 | Agent 场景 < 2s, 长上下文 < 5s |
| **TPOT** | Time Per Output Token | 平均输出 token 延迟 | < 100ms (流畅体验) |
| **ITL** | Inter-Token Latency | Token 间延迟 | < 150ms |
| **E2EL** | End-to-End Latency | 端到端延迟 | 业务 SLA 定义 |
| **Throughput** | — | 输出 token/s | 越高越好 |
| **Goodput** | — | 满足 SLO 的有效吞吐 | > 80% 峰值吞吐 |

### 6.2 资源效率指标

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| **GPU Memory Usage** | 显存占用 (GB) | `nvidia-smi` 监控 |
| **GPU Utilization** | GPU 计算利用率 (%) | `nvidia-smi` 监控 |
| **KV Cache Usage** | KV Cache 占用比例 | vLLM Prometheus 指标 |
| **Expert Load Balance** | 专家负载均衡度 (标准差) | vLLM 日志 / Prometheus |

### 6.3 测试工具采集指标

vllm_standalone_bench 自动采集的指标：

```
┌─────────────────────────────────────────────────────────┐
│ 请求级指标                                               │
│   latency (E2EL), ttft, itl[], tpot, output_tokens      │
│   prompt_len, success, error, start_time                 │
├─────────────────────────────────────────────────────────┤
│ 聚合级指标                                               │
│   request_throughput (req/s)                             │
│   output_throughput (tok/s)                              │
│   total_token_throughput (tok/s)                         │
│   request_goodput (req/s, 满足 SLO)                     │
│   max_output_tokens_per_s (峰值 tok/s)                  │
│   TTFT: mean, p50, p90, p99, std                        │
│   TPOT: mean, p50, p90, p99, std                        │
│   ITL:  mean, p50, p90, p99, std                        │
│   E2EL: mean, p50, p90, p99, std                        │
└─────────────────────────────────────────────────────────┘
```

### 6.4 Goodput SLO 定义

```bash
# Agent 场景 SLO
--goodput ttft:2000 tpot:100 e2el:10000

# Agent 长上下文场景 SLO
--goodput ttft:5000 tpot:150 e2el:30000

# 高并发压力测试 SLO
--goodput ttft:3000 tpot:200 e2el:15000
```

---

## 7. 执行流程

### 7.1 测试执行总流程

```
┌──────────────────────────────────────────────────────────────┐
│                    测试执行总流程                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 环境准备                                                 │
│     ├── 安装依赖: pip install -r requirements.txt            │
│     ├── 下载模型: Qwen3-35B-A3B + 量化版本                   │
│     └── 验证 GPU: nvidia-smi 确认 GPU 状态                   │
│                                                              │
│  2. 基准测试 (Baseline)                                      │
│     ├── 单卡 BF16 基准 (F3-01)                               │
│     └── 无投机、无前缀缓存、单体模式                          │
│                                                              │
│  3. 逐特性测试                                               │
│     ├── F1: 连续批处理 → 配置参数变化                        │
│     ├── F2: KV Cache/Prefix → 前缀比例变化                   │
│     ├── F3: 量化 → 量化方案变化                              │
│     ├── F4: 投机解码 → MTP/ngram 对比                        │
│     ├── F5: P/D 分离 → 拓扑变化                              │
│     └── F6: MOE 并行 → 并行策略变化                          │
│                                                              │
│  4. 组合优化测试                                             │
│     ├── 最佳量化 + 前缀缓存                                  │
│     ├── EP + MTP 投机解码                                    │
│     └── P/D 分离 + FP8 KV Cache                             │
│                                                              │
│  5. 结果分析与报告                                           │
│     ├── 汇总 CSV/XLSX 数据                                  │
│     ├── 生成对比图表                                         │
│     └── 输出优化建议                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 单项测试执行步骤

```bash
# Step 1: 启动 vLLM 服务
vllm serve /path/to/Qwen3-35B-A3B \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e5m2 \
    --gpu-memory-utilization 0.92 \
    --port 8000

# Step 2: 等待服务就绪
curl -s http://localhost:8000/v1/models | jq .

# Step 3: 执行性能测试
cd /work/development-code/Resource_Planning_Tool/vllm_standalone_bench
./run_bench.sh

# Step 4: 检查结果
ls -la results/
cat results/bench_*.csv
```

### 7.3 自动化脚本模板

```bash
#!/bin/bash
# run_full_benchmark.sh — 全特性批量测试脚本

MODEL_PATH="/path/to/Qwen3-35B-A3B"
RESULTS_DIR="results/full_benchmark_$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

run_test() {
    local test_name=$1
    local vllm_args=$2
    local bench_config=$3

    echo "=========================================="
    echo "  测试: $test_name"
    echo "=========================================="

    # 启动 vLLM
    vllm serve "$MODEL_PATH" $vllm_args --port 8000 &
    VLLM_PID=$!
    sleep 30  # 等待模型加载

    # 执行基准测试
    cd /work/development-code/Resource_Planning_Tool/vllm_standalone_bench
    eval "$bench_config"
    ./run_bench.sh

    # 移动结果
    mv results/bench_*.csv "$RESULTS_DIR/${test_name}.csv"
    mv results/bench_*.xlsx "$RESULTS_DIR/${test_name}.xlsx" 2>/dev/null

    # 停止 vLLM
    kill $VLLM_PID
    sleep 10  # 等待清理
}

# ─── F1: 连续批处理 ───
run_test "F1_chunked_on" \
    "--enable-chunked-prefill --max-num-seqs 128" \
    "INPUT_LENS='512 2048 8192' OUTPUT_LENS='128 256' PARALLEL_NUMS='1 4 16 32 64' EPOCHS=3"

run_test "F1_chunked_off" \
    "--max-num-seqs 128" \
    "INPUT_LENS='512 2048 8192' OUTPUT_LENS='128 256' PARALLEL_NUMS='1 4 16 32 64' EPOCHS=3"

# ─── F2: Prefix Caching ───
run_test "F2_prefix_0.0" \
    "--enable-prefix-caching" \
    "INPUT_LENS='8192' OUTPUT_LENS='512' PARALLEL_NUMS='1 4 16 32' PREFIX_RATIO=0.0 EPOCHS=3"

run_test "F2_prefix_0.8" \
    "--enable-prefix-caching" \
    "INPUT_LENS='8192' OUTPUT_LENS='512' PARALLEL_NUMS='1 4 16 32' PREFIX_RATIO=0.8 EPOCHS=3"

# ─── F3: 量化 ───
run_test "F3_bf16" \
    "--dtype bfloat16 --gpu-memory-utilization 0.90" \
    "INPUT_LENS='512 2048 8192' OUTPUT_LENS='128 256' PARALLEL_NUMS='1 4 16 32 64' EPOCHS=3"

run_test "F3_fp8" \
    "--quantization fp8 --kv-cache-dtype fp8_e5m2 --gpu-memory-utilization 0.92" \
    "INPUT_LENS='512 2048 8192' OUTPUT_LENS='128 256' PARALLEL_NUMS='1 4 16 32 64' EPOCHS=3"

# ─── F4: 投机解码 ───
run_test "F4_mtp_3" \
    "--num-speculative-tokens 3" \
    "INPUT_LENS='512 2048 8192' OUTPUT_LENS='128 256 512' PARALLEL_NUMS='1 4 8 16' EPOCHS=5"

run_test "F4_ngram_5" \
    "--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":5}'" \
    "INPUT_LENS='512 2048 8192' OUTPUT_LENS='128 256 512' PARALLEL_NUMS='1 4 8 16' EPOCHS=5"

echo "所有测试完成！结果保存在: $RESULTS_DIR"
```

---

## 8. 预期产出

### 8.1 产出物清单

| 产出物 | 格式 | 说明 |
|--------|------|------|
| **测试数据** | CSV / XLSX | 每组测试的详细指标数据 |
| **对比图表** | PNG / HTML | 各特性的性能对比图 |
| **分析报告** | Markdown | 综合分析与优化建议 |
| **配置模板** | Shell / JSON | 最佳实践配置模板 |

### 8.2 报告结构

```
vllm_performance_benchmark_report.md
├── 1. 执行摘要
│   ├── 关键发现 (Top 5)
│   └── 优化建议优先级
├── 2. 测试环境详情
├── 3. 各特性详细分析
│   ├── 3.1 连续批处理分析
│   ├── 3.2 KV Cache/Prefix 分析
│   ├── 3.3 量化方案分析
│   ├── 3.4 投机解码分析 (MTP)
│   ├── 3.5 P/D 分离分析
│   └── 3.6 MOE 专家并行分析
├── 4. 组合优化方案
├── 5. GPU 选型建议 (A40 vs H2x)
├── 6. 生产部署推荐配置
└── 附录
    ├── A. 完整测试数据表
    ├── B. 测试脚本
    └── C. 已知问题与限制
```

### 8.3 关键分析图表

1. **吞吐量-延迟曲线图**: 每个特性下不同配置的吞吐量 vs P99 TTFT
2. **GPU 对比雷达图**: A40 vs H2x 在各维度的综合表现
3. **量化方案对比柱状图**: BF16 / AWQ / FP8 的性能、显存对比
4. **前缀缓存加速比热力图**: 不同 input_len × prefix_ratio 的 TTFT 加速比
5. **P/D 分离拓扑对比图**: 不同拓扑下的吞吐量和延迟
6. **MOE 并行策略对比**: EP vs TP 在不同并发下的吞吐量曲线

---

## 附录 A: 快速参考卡

### A.1 vLLM 关键参数速查

```bash
# 连续批处理
--max-num-batched-tokens <int>    # 单次迭代最大 token 数 (默认 2048)
--max-num-seqs <int>              # 单次迭代最大序列数 (默认 128)
--enable-chunked-prefill          # 启用分块 prefill

# KV Cache / Prefix
--enable-prefix-caching           # 启用前缀缓存
--block-size <int>                # KV Cache 块大小 (默认 16)
--gpu-memory-utilization <float>  # GPU 显存分配 (默认 0.92)
--kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3}
--prefix-caching-hash-algo {sha256,xxhash}

# 量化
-q {awq,fp8,...}                  # 量化方法

# 投机解码
--num-speculative-tokens <int>    # MTP 投机 token 数
--speculative-config <json>       # 完整投机配置 JSON

# P/D 分离
--kv-transfer-config <json>       # KV 传输配置

# MOE 专家并行
--enable-expert-parallel -ep      # 启用专家并行
--enable-eplb                     # 启用专家负载均衡
--expert-placement-strategy {linear,round_robin}
--all2all-backend {allgather_reducescatter,deepep_high_throughput,...}
--data-parallel-size -dp <int>    # 数据并行大小
```

### A.2 测试脚本参数速查

```bash
# run_bench.sh 关键配置
INPUT_LENS="8192 16384 32768 64000 128000"  # 输入 token 长度列表
OUTPUT_LENS="256 512 1024"                   # 输出 token 长度列表
PARALLEL_NUMS="1 4 8 16 32 64"               # 并发请求数列表
EPOCHS=3                                     # 每组重复轮数
PREFIX_RATIO=0.8                             # 前缀缓存共享比例
SLEEP_BETWEEN=3.0                            # 组间等待时间 (秒)
MAX_TTFT_MS=10000                            # 最大 TTFT 阈值 (ms)
MIN_THROUGHPUT_TOK_S=5                       # 最低吞吐量阈值 (tok/s)
TOKENIZER="/path/to/tokenizer"               # Tokenizer 路径
```

### A.3 模型显存估算

| 精度 | Qwen3-35B-A3B 估算显存 | 说明 |
|------|----------------------|------|
| BF16 | ~18 GB (权重) + KV Cache | 激活参数 3B，总参数 35B |
| AWQ (W4) | ~5 GB (权重) + KV Cache | 4-bit 量化 |
| FP8 | ~9 GB (权重) + KV Cache | 8-bit 量化 |
| KV Cache (BF16, 8K ctx, batch=32) | ~8 GB | 约 256K tokens |
| KV Cache (FP8, 8K ctx, batch=32) | ~4 GB | FP8 节省 50% |

---

> **文档维护**: 本文档随测试进展持续更新。
> **联系方式**: 如有疑问请联系推理优化团队。
