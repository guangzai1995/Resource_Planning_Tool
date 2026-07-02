# DeepSeek v4 flash 910B/910C 8 卡估算设计

- 日期：2026-06-30
- 范围：根目录 `资源规划工具.xlsx`、`data/` 下历史测试 CSV、现有资源规划/预测代码
- 方案：方案二 - P800 实测基线 + 模型差异 + 硬件 Roofline 比例
- 目标：估算 `DeepSeek v4 flash` 在 `910B 8卡` 与 `910C 8卡` 部署下的测试数据，输出公式、依据、假设和置信度

## 1. 背景与目标

根目录 `资源规划工具.xlsx` 已包含 P800、H200、H20 多个模型的实测 sheet，其中 `671B-P800-8测试数据` 与 DeepSeek MoE 量级最接近，可作为 P800 侧的实测基线。`data/910B1/` 与 `data/910B3/` 中存在 910B vLLM 部署历史 CSV，但用户确认这批数据算力利用率偏低，只允许作为参考，不用于反推校准系数。

本次设计目标不是给出“实测替代品”，而是形成可审计的估算方法：每个输出数据点都能追溯到 P800 基线行、硬件参数、模型参数、瓶颈判断、公式比例和置信度。后续如果拿到内部 910B/910C 规格或 DeepSeek v4 flash 精确模型配置，只需要替换假设表并重新计算。

## 2. 数据来源与外部依据

| 来源 | 用途 | 口径 |
| --- | --- | --- |
| `资源规划工具.xlsx` / `671B-P800-8测试数据` | 主基线 | 使用输入长度、输出长度、并发数、吞吐、TTFT、增量时延等实测列 |
| 用户确认 + `backend/app/seed_data.py` | P800 默认规格 | 本次测试使用的 P800 显存按用户确认的 96GB；带宽/算力沿用项目内 2000 GB/s、280 BF16 TFLOPS 口径 |
| `data/910B1/*`、`data/910B3/*` | 参考样本 | 只做 sanity check，不参与校准和比例修正 |
| DeepSeek-V3 Technical Report | DeepSeek MoE 默认模型代理 | 报告摘要说明 DeepSeek-V3 是 671B 总参数、每 token 激活 37B 参数的 MoE 模型：https://arxiv.org/abs/2412.19437 |
| DeepSeek V4-Flash Hugging Face 模型卡 | 目标模型参数 | 模型卡记录 V4-Flash 为 284B 总参数、每 token 激活 13B 参数：https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |
| Serving Large Language Models on Huawei CloudMatrix384 | 910C/CloudMatrix 推理公开参考 | 论文摘要说明 CloudMatrix384 集成 384 个 Ascend 910C NPU，并给出 DeepSeek-R1 推理吞吐结果：https://arxiv.org/abs/2506.12708 |
| Tom's Hardware CloudMatrix384 报道 | 910C 单卡算力公开参考 | 报道引用 384 卡约 300 PFLOPS BF16 与单 910C 约 780 BF16 TFLOPS，作为公开非厂商默认值：https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power |

公开硬件规格允许进入默认公式假设；公开吞吐 benchmark 和本地低利用率 910B vLLM 数据只用于参考展示，不参与校准。如果内部资料与公开资料冲突，以内部资料为准，并在输出中标注来源。

## 3. 输入假设

### 3.1 硬件假设

| GPU | 默认来源 | 显存 GB | 带宽 GB/s | BF16 TFLOPS | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| P800 | 用户确认 + 项目内规格 | 96 | 2000 | 280 | 显存使用用户确认的实测卡 96GB；带宽/算力沿用项目口径 |
| 910B | P800 等价/内部可替换 | 64 | 2000 | 280 | 默认按 P800/910B 等价保守估算 |
| 910C | 公开资料 + 双 910B 假设 | 128 | 4000 | 780 | BF16 来自公开报道；显存/带宽按双 910B 保守假设，需可配置 |

运行时效率不从 `data/910B*` 反推。默认设置三档情景：

| 情景 | runtime_efficiency | 用途 |
| --- | ---: | --- |
| 保守 | 0.55 | CANN/算子/通信尚未充分优化 |
| 基准 | 0.70 | 常规优化部署 |
| 乐观 | 0.85 | 算子、并行、通信均较充分优化 |

第一版输出使用“基准”情景，同时在假设 sheet 中保留三档参数。

### 3.2 模型假设

本次按 `DeepSeek R1 INT8` 的 P800 实测作为基线，用 `DeepSeek V4-Flash INT8` 作为目标模型。V4-Flash 的公开模型卡口径为 284B 总参数、13B 激活参数；如果内部资料确认 283B/14B，应只替换假设表并重算。

| 字段 | R1/P800 基线 | V4-Flash 目标 | 说明 |
| --- | ---: | ---: | --- |
| total_params_b | 671 | 284 | 权重驻留显存按总参数估算 |
| active_params_b | 37 | 13 | decode/prefill 主计算量按每 token 激活参数估算 |
| layers | 61 | 61 | V4-Flash 未公开完整结构时沿用 R1/V3 代理结构 |
| hidden_size | 7168 | 7168 | 同上 |
| num_kv_heads | 128 | 128 | 同上 |
| head_size | 128 | 128 | 同上 |
| weight_bytes | 1.0 | 1.0 | 用户确认 R1 采用 INT8；V4-Flash 本次也按 INT8 估算 |
| kv_bytes | 2.0 | 2.0 | KV cache 默认 BF16；若启用 FP8 KV，改为 1.0 |

如果用户提供 DeepSeek V4-Flash 的内部总参数、激活参数、量化方式或 KV 精度，必须覆盖默认代理值；输出中保留 `model_spec_source`。

## 4. 估算公式

### 4.1 显存可行性

每个基线行先判断 8 卡是否可部署：

```text
weights_gb = total_params_b * 1e9 * weight_bytes / 1024^3
kv_cache_gb = 2 * layers * num_kv_heads * head_size * kv_bytes
              * (input_tokens + output_tokens) * concurrency / 1024^3

required_per_card_gb =
  weights_gb / gpu_count
  + kv_cache_gb / gpu_count
  + framework_overhead_gb

feasible = required_per_card_gb <= memory_gb * memory_utilization
```

默认 `gpu_count=8`，`memory_utilization=0.90`，`framework_overhead_gb=3.0`。不可行行不填吞吐/时延估算，标记为 `memory_infeasible`。

### 4.2 Decode 吞吐

对每个 P800 基线行计算目标硬件比例：

```text
compute_scale =
  (target_tflops * target_runtime_efficiency)
  / (base_tflops * base_runtime_efficiency)
  * (base_active_params_b / target_active_params_b)

bandwidth_scale =
  (target_bandwidth_gbs * target_runtime_efficiency)
  / (base_bandwidth_gbs * base_runtime_efficiency)
  * (base_decode_bytes_per_token / target_decode_bytes_per_token)

decode_scale = min(compute_scale, bandwidth_scale)
throughput_target = throughput_base * decode_scale
```

`base_decode_bytes_per_token` 和 `target_decode_bytes_per_token` 包含权重读取与 KV cache 读写：

```text
decode_bytes_per_token =
  active_params_b * 1e9 * weight_bytes
  + 2 * layers * num_kv_heads * head_size * kv_bytes
    * (input_tokens + generated_tokens_so_far)
```

第一版使用行级近似，将 `generated_tokens_so_far` 取为 `output_tokens / 2`。

### 4.3 首 token 时延

TTFT 主要按 prefill 计算比例估算：

```text
prefill_compute_scale =
  (target_tflops * target_runtime_efficiency)
  / (base_tflops * base_runtime_efficiency)
  * (base_prefill_flops / target_prefill_flops)

ttft_mean_target = ttft_mean_base / prefill_compute_scale * queue_adjustment
ttft_p90_target  = ttft_p90_base  / prefill_compute_scale * queue_adjustment
ttft_p99_target  = ttft_p99_base  / prefill_compute_scale * queue_adjustment
ttft_max_target  = ttft_max_base  / prefill_compute_scale * queue_adjustment
```

prefill FLOPs 采用可解释近似：

```text
prefill_flops =
  2 * active_params_b * 1e9 * input_tokens
  + attention_factor * layers * input_tokens^2 * hidden_size
```

默认 `attention_factor=2`。同一 DeepSeek 代理模型下，模型比例接近 1，主要由硬件和运行时效率驱动。

### 4.4 增量时延

增量时延按 decode 吞吐比例反推：

```text
decode_latency_mean_target = decode_latency_mean_base / decode_scale
decode_latency_p90_target  = decode_latency_p90_base  / decode_scale
decode_latency_p99_target  = decode_latency_p99_base  / decode_scale
decode_latency_max_target  = decode_latency_max_base  / decode_scale
```

如果 `decode_scale` 来自带宽瓶颈，输出 `bottleneck=memory_bandwidth`；如果来自计算瓶颈，输出 `bottleneck=compute`。

## 5. 输出设计

产出一个估算工作簿，建议命名：

```text
outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx
```

Sheet 设计：

| Sheet | 内容 |
| --- | --- |
| `00_估算说明` | 版本、来源、适用边界、不能替代实测的说明 |
| `01_假设表` | 硬件、模型、运行时效率、显存参数、公式参数 |
| `02_910B_8卡估算` | 按 P800 基线行生成 910B 8 卡估算 |
| `03_910C_8卡估算` | 按 P800 基线行生成 910C 8 卡估算 |
| `04_P800基线` | 复制使用到的 `671B-P800-8测试数据` 行，便于追溯 |
| `05_910B低利用率参考` | 导入 `data/910B1/`、`data/910B3/` 汇总统计，只做参考 |
| `06_联网参考数据` | 记录公开论文/报道中的 910B、910C 测试或规格数据，只做对照说明 |

估算 sheet 保留原 12 列指标，并新增审计列：

| 新增列 | 说明 |
| --- | --- |
| `base_sheet` | 固定为 `671B-P800-8测试数据` |
| `base_row` | 基线行号 |
| `target_gpu` | `910B` 或 `910C` |
| `gpu_count` | 固定 8 |
| `runtime_efficiency` | 当前情景效率 |
| `compute_scale` | 计算比例 |
| `bandwidth_scale` | 带宽比例 |
| `decode_scale` | decode 采用比例 |
| `bottleneck` | `compute` / `memory_bandwidth` / `memory_infeasible` |
| `required_per_card_gb` | 估算单卡显存需求 |
| `confidence` | 估算置信度 |
| `assumption_notes` | 使用的关键假设 |

## 6. 910B 历史数据使用边界

`data/910B1/` 与 `data/910B3/` 中的数据来自 vLLM 部署，用户已确认利用率偏低。因此：

- 不用这些数据拟合 runtime_efficiency。
- 不用这些数据反推 910B/P800 的硬件比例。
- 不因为估算值高于历史 vLLM 数据就下调结果。
- 仅在 `05_910B低利用率参考` 中展示均值、最大吞吐、最小增量时延、TTFT 范围，并标注“低利用率参考”。

如果估算结果与低利用率参考差距过大，只输出风险提示：

```text
当前估算依赖硬件/模型比例，未用 vLLM 低利用率样本校准。实际部署若沿用相同 vLLM/CANN 栈，可能低于估算值。
```

## 6.1 联网参考数据使用边界

联网检索到的公开数据进入 `06_联网参考数据`，用于解释风险区间和公开对照。公开硬件规格可进入 `01_假设表` 并参与公式；公开吞吐 benchmark 不参与 `02_910B_8卡估算`、`03_910C_8卡估算` 的校准。第一版纳入以下参考项：

| 来源 | 硬件 | 模型/场景 | 可记录指标 | 使用边界 |
| --- | --- | --- | --- | --- |
| A-IO: Adaptive Inference Orchestration for Memory-Bound NPUs | Ascend 910B | OpenPangu 1B/7B，HuggingFace + PyTorch，非 vLLM/MindIE | 2K 场景 1B/7B TPS、32K 场景准确率、910B 64GB HBM、CANN/HDK 版本 | 单卡小模型参考，不能外推 DeepSeek 8 卡大 MoE |
| An Empirical Study of OpenPangu Quantization on Ascend NPUs | Ascend 910B1 | OpenPangu 1B/7B，PTQ 精度研究 | 910B1 64GB HBM、PyTorch/torch-npu/Transformers 版本、量化精度结论 | 主要是精度/量化参考，不是吞吐基准 |
| Serving Large Language Models on Huawei CloudMatrix384 | 384 个 Ascend 910C | DeepSeek-R1 + CloudMatrix-Infer | prefill 6688 tokens/s/NPU、decode 1943 tokens/s/NPU、15ms TPOT 下 538 tokens/s/NPU | 384 卡专用架构与 CloudMatrix-Infer，上限参考，不能直接当 8 卡结果 |
| Tom's Hardware CloudMatrix384 报道 | Ascend 910C | 公开规格/系统对比 | 910C 780 BF16 TFLOPS、128GB HBM、3.2TB/s HBM 带宽；CloudMatrix384 300 BF16 PFLOPS | 非厂商官方规格，作为默认硬件假设来源并标注可信度 |

如果公开参考与内部实测、内部规格冲突，以内部数据为准。输出工作簿必须在说明中标注这些数据的可比性限制。

## 7. 置信度设计

置信度采用 0 到 1 分数，不代表统计置信区间，只用于提示估算可靠性：

```text
confidence =
  0.30 * source_data_score
  + 0.25 * hardware_spec_score
  + 0.20 * model_spec_score
  + 0.15 * runtime_assumption_score
  + 0.10 * range_score
```

默认分值：

| 项 | 910B | 910C | 说明 |
| --- | ---: | ---: | --- |
| source_data_score | 1.00 | 1.00 | P800 基线为实测 |
| hardware_spec_score | 0.70 | 0.60 | 910B 按项目 P800 等价；910C 公开资料非厂商规格 |
| model_spec_score | 0.65 | 0.65 | 未拿到 DeepSeek v4 flash 精确配置时使用 V3/R1 代理 |
| runtime_assumption_score | 0.60 | 0.60 | 不使用低利用率数据校准 |
| range_score | 0.90 | 0.90 | 只在 P800 已测输入/并发范围内转换 |

基准情景默认置信度约为 0.76（910B）与 0.74（910C）。如果用户提供内部硬件和模型规格，置信度可提高；如果目标输入/并发超出 P800 基线范围，`range_score` 降低。

## 8. 错误处理与边界

- 缺少 P800 基线 sheet：停止生成，报告 `base_sheet_missing`。
- 缺少必要列：停止生成，列出缺失字段。
- DeepSeek v4 flash 默认代理参数被用户覆盖但不完整：使用默认补齐，并在 `assumption_notes` 标注。
- 显存不可行：不输出伪数值，标记 `memory_infeasible`。
- 比例异常：如果 `decode_scale > 5` 或 `< 0.2`，保留结果但标记 `scale_outlier`，提示必须实测验证。
- 外部链接不可访问：不阻塞计算，但在说明 sheet 标记来源验证失败。

## 9. 测试与验证计划

实现前先补测试，覆盖以下行为：

1. 能读取 `671B-P800-8测试数据` 并识别 12 个原始指标列。
2. 显存公式对 BF16 与 INT8 权重情景给出不同可行性结果。
3. `data/910B*` 不参与估算公式；修改参考 CSV 不影响 `02_910B_8卡估算` 的数值。
4. 同一 P800 基线行输出 910B 与 910C 两条结果，且包含 `base_row`、比例列和置信度。
5. 当 910C 算力高于 P800 时，基准情景下 `compute_scale` 大于 1。
6. 输出 Excel 不包含公式错误；如果使用公式列，必须重新计算并检查。

人工验证：

- 抽样 3 行手算 `compute_scale`、`bandwidth_scale`、`decode_scale`。
- 对比 `05_910B低利用率参考`，确认只展示参考统计，没有进入估算比例。
- 检查说明 sheet 明确写出“估算不能替代实测”。

## 10. 不在本次范围

- 不补测真实 910B/910C。
- 不把 910B vLLM 低利用率样本用于校准。
- 不修改现有在线预测 API。
- 不改变 `资源规划工具.xlsx` 原文件；估算结果输出到 `outputs/` 新工作簿。
- 不承诺 DeepSeek v4 flash 与 DeepSeek-V3/R1 架构完全一致；默认代理只是在缺少精确模型规格时的可审计假设。

## 11. 验收标准

1. 输出同时包含 `910B 8卡` 与 `910C 8卡` 估算数据。
2. 每行估算都有 P800 基线行、硬件比例、模型比例、瓶颈、显存需求和置信度。
3. 910B 历史 vLLM 数据和公开吞吐 benchmark 只出现在参考 sheet，不参与校准；公开硬件规格可以在假设表中参与估算。
4. 文档和输出工作簿都保留公式说明、来源链接和假设表。
5. 缺少关键规格时不静默硬编码，必须在假设表和说明中标注。
