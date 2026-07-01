# vLLM MTP 鉴权与真实数据集设计

- 日期：2026-07-01
- 分支：`feat/mtp-auth-real-data`
- worktree：`.worktrees/mtp-auth-real-data`
- 状态：待用户审查（brainstorming 产物，下一步进入 writing-plans）

## 1. 背景

用户提供的 `/tmp/results_260701` 日志在当前环境实际落在 `/tmp/results_260701.ICSoPs`，仓库内也有 `vllm_standalone_bench/results/results_260701.tar.gz`。排查结论：

1. 401 主要来自 vLLM 的 `GET /v1/metrics`。业务压测请求已经通过 `Authorization=Bearer local-bench-key` 访问 `/v1/chat/completions`，并能返回 200。
2. `vllm_bench/serve.py` 的 speculative decoding metrics 采集没有复用 benchmark 请求的鉴权 header，因此开启 `--api-key` 的 vLLM 服务会拒绝 `/v1/metrics`。
3. `serve.py` 已经会把 `spec_decode_acceptance_rate` 等指标放入 result dict，但 `run_bench_multi.py` 的 `_extract_row()` 和 CSV/XLSX 表头没有承接这些字段，所以即使 metrics 采集成功，自动化结果表也看不到 MTP 接受率。
4. 当前 MTP 压测使用 `dataset_name=random`。随机 token prompt 缺少真实任务语义，容易低估 MTP 接受率，不适合作为“接近正常 MTP”的主测试数据。
5. 结果中还暴露过一个配置易错点：`speculative-config.num_speculative_tokens` 少写 `--` 会导致 vLLM 启动失败。自动化框架应尽早报出这类明显错误，而不是等 ready check 超时。

## 2. 目标

- 解决 `/v1/metrics` 401，使 MTP 指标采集复用现有 API key / header 配置。
- 将 MTP acceptance 指标写入 CSV/XLSX，保留旧结果和非 MTP 场景的兼容性。
- 在现有 `auto_bench -> run_bench_multi.py -> vllm_bench/serve.py` 链路内支持内置 MTP 真实风格数据集。
- 通过 `bench_profiles[].dataset` 字段配置数据集，不破坏现有 `input_lens`、`output_lens`、`parallel_nums`、`epochs` 矩阵语义。
- 让 `input_lens` 在内置 MTP 数据集下表示“目标上下文长度桶”，而不是静默失效。
- 对 MTP 常见配置错误做快速校验，减少长时间 ready check 后才发现参数问题。

## 3. 非目标

- 不接入外部在线数据源，不依赖运行时网络。
- 不引入新的 benchmark 工具链或独立脚本；仍使用现有自动化测试工程。
- 不用日志解析作为主数据通道。`vllm.log` 中的 `SpecDecoding metrics` 只作为人工排查兜底。
- 不强行改变所有现有配置文件的默认数据集；未配置 `dataset` 时保持当前 random 行为。
- 不保证 `num_speculative_tokens > 1` 一定稳定或高接受率。vLLM 日志已提示该模式可能降低 acceptance，框架只负责准确测量和呈现。

## 4. 总体设计

保留现有三层边界：

```text
auto_bench.py
  读取 JSON 配置、启动 vLLM 容器、启动 bench runner

run_bench_multi.py
  展开 input_lens/output_lens/parallel_nums/epochs 矩阵，调用 serve.py

vllm_bench/serve.py
  加载数据集、发 OpenAI-compatible 请求、采集 metrics、返回 result dict
```

本次只扩展这些边界内的配置和数据流：

```text
bench_profiles[].dataset
  -> auto_bench build_bench_run_command
  -> run_bench_multi parser/base args
  -> serve.py dataset loader
  -> RequestFuncInput(prompt, prompt_len, output_len)
```

未配置 `dataset` 时，当前行为不变：

```text
dataset_name=random
input_lens 精确映射 random_input_len
output_lens 精确映射 random_output_len
```

配置 `dataset.name=builtin_mtp_chat` 时：

```text
input_lens 表示目标 prompt token 长度桶
output_lens 表示 max_tokens
parallel_nums/epochs 仍控制并发和请求总数
```

## 5. 鉴权设计

### 5.1 现状

`run_bench_multi.py` 已经把 `--api-key` 转为：

```text
--header Authorization=Bearer <api_key>
```

业务请求会把 header 传给 `endpoint_request_func.py`。ready check 也已经在 `auto_bench.py` 中对 `/v1/models` 带上 API key。

缺口在 `serve.py::fetch_spec_decode_metrics()`：

```text
GET <base_url>/metrics
```

该请求没有 headers，导致 vLLM 启用 `--api-key` 后返回 401。

### 5.2 改动

- `fetch_spec_decode_metrics(base_url, session, extra_headers=None)` 新增可选 headers 参数。
- `benchmark()` 在采集 before/after metrics 时传入 `extra_headers`。
- 401/403 时不抛异常中断压测，但打印清晰告警，说明 metrics 因鉴权失败不可用。
- 无 API key 的本地服务继续不带 header，保持兼容。

## 6. MTP 指标落盘

`serve.py` result dict 已有 MTP 指标出口，自动化结果层需要承接：

- `spec_decode_acceptance_rate`
- `spec_decode_acceptance_length`
- `spec_decode_num_drafts`
- `spec_decode_draft_tokens`
- `spec_decode_accepted_tokens`
- `spec_decode_per_position_acceptance_rates`

`run_bench_multi.py` 的 `_extract_row()` 新增这些字段：

- 有值时按数值/JSON 字符串写入。
- 非 MTP 或 metrics 不可用时写空字符串或 0，旧流程不报错。
- CSV/XLSX 表头追加 MTP 小分组，中文表头说明单位。

建议列顺序：

```text
..., token_source,
spec_decode_acceptance_rate,
spec_decode_acceptance_length,
spec_decode_num_drafts,
spec_decode_draft_tokens,
spec_decode_accepted_tokens,
spec_decode_per_position_acceptance_rates,
throughput_req_s, throughput_tok_s, ...
```

## 7. `dataset` 配置设计

### 7.1 配置形态

在 `bench_profiles[]` 下新增可选字段 `dataset`：

```json
{
  "name": "mtp_real_chat",
  "backend": "openai-chat",
  "dataset": {
    "name": "builtin_mtp_chat",
    "length_policy": "bucket",
    "input_len_tolerance": 0.2,
    "on_bucket_shortage": "error",
    "sampling": "shuffle"
  },
  "input_lens": [1024, 4096, 8192],
  "output_lens": [256, 1024],
  "parallel_nums": [1, 4, 8],
  "epochs": 3,
  "warmup_requests": 1,
  "cross_product": true
}
```

字段语义：

- `dataset.name`：数据集名称。第一版支持 `random` 和 `builtin_mtp_chat`；未配置时等价于 `random`。
- `length_policy`：长度解释策略。`random` 默认 `exact`；`builtin_mtp_chat` 默认 `bucket`。
- `input_len_tolerance`：bucket 允许偏差，默认 `0.2`，表示目标长度的 ±20%。
- `on_bucket_shortage`：样本不足时策略。默认 `error`，避免静默用短 prompt 污染结果。
- `sampling`：同一 bucket 内样本选择策略。默认 `shuffle`，受 seed 控制；也支持 `round_robin` 便于复现。

### 7.2 `input_lens` 是否失效

不失效。

在 `builtin_mtp_chat` 下，`input_lens` 从“精确生成 N 个随机 token”变为“选择 prompt token 数接近 N 的真实风格样本”。例如：

```text
input_len=4096
input_len_tolerance=0.2
有效样本范围约为 3277 到 4915 prompt tokens
```

这样保留现有矩阵比较能力：

```text
(input_len, output_len, parallel_num)
```

仍然能表达“1K/4K/8K 上下文下 MTP acceptance 和吞吐怎么变化”。

实际 token 长度继续通过已有列呈现：

- `input_len`：请求目标 bucket。
- `avg_input_tokens`：服务端或 tokenizer 实测均值。
- `input_compliance`：实测均值 / 目标 bucket。

### 7.3 `output_lens` 语义

`output_lens` 不跟随数据集失效，仍表示请求 `max_tokens`。

内置 MTP 数据集只提供 prompt/messages，不提供固定答案。这样避免把 “参考答案长度” 和 “模型实际生成长度” 混在一起，也保持现有 output compliance 统计逻辑。

### 7.4 tokenizer 前置条件

`builtin_mtp_chat` 必须有 tokenizer：

- `auto_bench` 场景使用 `models[].tokenizer_path`。
- 直接跑 `run_bench_multi.py` 时必须传 `--tokenizer`。
- 如果没有 tokenizer，应配置错误退出，不降级到 `skip_tokenizer_init=True`，因为 bucket 选择必须依赖真实 chat template 后的 token 数。

## 8. 内置 MTP 数据集设计

### 8.1 数据集定位

新增内置数据集名称：

```text
builtin_mtp_chat
```

目标是“真实风格、可离线、可复现”的 MTP benchmark prompt 池。它不追求覆盖所有业务领域，但要明显优于随机 token prompt。

### 8.2 样本内容

内置样本使用人工编写或模板化生成的合成真实任务，避免外部版权和网络依赖。类别覆盖：

- 中文长文总结与信息抽取。
- 多轮问答和上下文追问。
- 代码阅读、bug 定位、改写和补全。
- 日志分析与根因总结。
- 表格/JSON/配置理解。
- 数学和步骤推理。
- 中英混合技术问答。

样本应避免纯重复填充。长上下文可由多个主题段落、代码片段、日志片段、配置片段组合而成，保证 token 分布接近真实 chat prompt，而不是高熵随机 token 或单段重复文本。

### 8.3 存储与加载

推荐新增：

```text
vllm_standalone_bench/vllm_bench/datasets/builtin_mtp_chat.py
```

职责：

- 提供 deterministic prompt 模板和样本池。
- 按 tokenizer + chat template 计算 prompt token 数。
- 按目标 bucket、tolerance、seed 选样本。
- 返回现有 `SampleRequest` / `RequestFuncInput` 可消费的数据结构。

不把大体积数据文件放进结果目录，也不需要运行时下载数据。

### 8.4 chat template

对 `openai-chat` 后端，样本以 messages 形式发送：

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."}
]
```

prompt token 数用 tokenizer 的 chat template 计算，等价于真实 OpenAI-compatible chat 请求进入模型前的 prompt 长度。若 tokenizer 不支持 chat template，报清晰错误，提示改用支持 chat template 的 tokenizer 或 random dataset。

## 9. `auto_bench` 兼容设计

`auto_bench.py` 的 `BenchProfile` 增加可选 `dataset` 对象。

构建 bench runner 命令时：

- 未配置 `dataset`：不追加新参数，保持现有命令。
- `dataset.name=builtin_mtp_chat`：追加 `--dataset builtin_mtp_chat` 以及 bucket 相关参数。
- 仍然追加 `--input-lens`、`--output-lens`、`--parallel-nums`。

`run_bench_multi.py` 增加参数：

```text
--dataset
--dataset-length-policy
--dataset-input-len-tolerance
--dataset-on-bucket-shortage
--dataset-sampling
```

实现上由 `run_bench_multi.py` 将 `--dataset` 映射为 `serve.py` 的 `args.dataset_name`，但对配置文件用户暴露统一的 `dataset` 概念。

## 10. MTP 配置校验

对 `serve_profiles[].args` 增加轻量校验：

- 如果发现形如 `speculative-config.*` 且不以 `--` 开头的参数，直接报 `ConfigError`。
- 错误消息给出修正示例：`--speculative-config.num_speculative_tokens`。
- 不自动改写用户参数，避免隐藏配置错误。

该校验只覆盖明显拼写遗漏，不承担 vLLM 全部参数兼容性验证。

## 11. 错误处理

- metrics 401/403：压测继续，MTP 指标为空，日志明确说明鉴权失败。
- metrics endpoint 无 speculative decoding 指标：压测继续，MTP 指标为空。
- `builtin_mtp_chat` 缺 tokenizer：配置错误退出。
- bucket 样本不足且 `on_bucket_shortage=error`：该配置失败，说明目标长度、tolerance 和可用样本数。
- bucket 样本不足且未来支持非 error 策略时，结果必须记录降级策略；第一版默认不降级。
- 所有请求失败：现有 `n_success=0` 语义保持，MTP 指标为空或 0，不伪装成有效结果。

## 12. 测试策略

### 单元测试

- `test_serve_metrics.py`：验证 `fetch_spec_decode_metrics()` 带 header 请求 `/metrics`，401 时返回 None 并可诊断。
- `test_extract_row.py`：验证 `spec_decode_*` 字段从 result dict 写入 row；缺失时兼容。
- `test_auto_bench.py`：验证 `bench_profiles[].dataset` 解析、resolved config、docker command 参数。
- `test_auto_bench.py`：验证 `serve_profiles[].args` 中裸 `speculative-config.*` 会报配置错误。
- 新增 builtin dataset 测试：固定 seed、tokenizer stub，验证 bucket 选择、tolerance、样本不足报错。
- `test_random_dataset.py`：验证未配置 `dataset` 时仍是 random 行为，现有测试不变。

### 集成/烟测

- `pytest -q` 全绿。
- dry-run 一个含 `dataset.name=builtin_mtp_chat` 的 auto_bench 配置，确认命令仍包含 input/output/concurrency 矩阵参数。
- 在启用 `--api-key` 的 vLLM MTP 服务上跑小矩阵，确认 vLLM 日志不再出现 `/v1/metrics` 401，CSV/XLSX 出现 MTP acceptance 列。
- 对比 random 与 `builtin_mtp_chat`：内置数据集的 acceptance 应更接近真实 chat 任务；不设硬编码阈值，只要求指标可观测且数据链路正确。

## 13. 验收标准

1. 现有 random benchmark 配置不改也能继续运行，结果 schema 向后兼容。
2. `dataset.name=builtin_mtp_chat` 可以通过 JSON 配置启用，且不需要外部文件或网络。
3. `input_lens` 在内置 MTP 数据集下作为长度桶生效，结果中的 `avg_input_tokens` 接近目标长度，偏差超过 tolerance 时失败而非静默通过。
4. `/v1/metrics` 请求复用 API key，开启 vLLM `--api-key` 时不再产生 metrics 401。
5. CSV/XLSX 包含 MTP acceptance 指标列；非 MTP 场景列为空或 0，不影响旧流程。
6. 常见 MTP 参数漏写 `--` 时配置阶段失败，并给出明确修正提示。
7. `pytest -q` 通过。

## 14. 风险与取舍

- 内置数据集是合成真实风格数据，不等于用户生产流量。它适合做稳定、离线、可复现的 MTP benchmark baseline；生产流量回放可作为后续扩展。
- bucket 语义无法保证每条请求都精确等于 `input_len`。这是保留真实 prompt 结构的代价，用 `input_compliance` 和 tolerance 控制可解释性。
- `num_speculative_tokens > 1` 可能降低 acceptance 或触发 vLLM 版本缺陷。框架会准确记录失败和指标，但不在第一版规避 vLLM 内部 bug。
- 内置数据集要求 tokenizer 可用。相比无 tokenizer 的近似字符串模式，这是为了让长度桶和 chat template 语义可信。

## 15. 涉及文件

- 改：`vllm_standalone_bench/vllm_bench/serve.py`
- 改：`vllm_standalone_bench/run_bench_multi.py`
- 改：`vllm_standalone_bench/auto_bench.py`
- 新增：`vllm_standalone_bench/vllm_bench/datasets/builtin_mtp_chat.py`
- 改/新：`vllm_standalone_bench/tests/`
- 改：`vllm_standalone_bench/configs/auto_bench.example.json`
- 改：`vllm_standalone_bench/README.md`
