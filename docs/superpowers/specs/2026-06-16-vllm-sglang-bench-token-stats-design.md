# vLLM / SGLang 统一基准测试工程 —— 输出 token 统计修复与双框架兼容加固

- 日期：2026-06-16
- 范围：`vllm_standalone_bench/`（基于 `vllm_bench/serve.py` 的独立基准测试工程）
- 关联代码库：`vllm-main/`、`sglang-main/`
- 方案：方案 C —— 混合加固（保留 serve.py 引擎，聚焦统计/合规/校验层）

---

## 1. 背景与问题

工程目标：用同一套测试工程，通过 OpenAI 兼容接口同时压测 **vLLM** 与 **SGLang** 两个推理框架，并输出可信的 TTFT/TPOT/ITL/E2EL/吞吐指标。

用户反馈的异常：**输出 token 统计不正常**，具体表现为 CSV 的 `avg_input_tokens` / `avg_output_tokens` 列**永远等于请求的 `input_len` / `output_len`**，不反映服务端实际生成量。

附带诉求：
1. 让两个框架都能"按测试工程指定长度输出"；
2. 审查输入长度生成与统计是否正常；
3. 审查指标计算是否正常。

---

## 2. 现状代码定位（关键文件）

| 文件 | 职责 |
|---|---|
| `vllm_standalone_bench/run_bench_multi.py` | 多配置编排、结果提取（`_extract_row`）、CSV/XLSX 落盘、约束过滤 |
| `vllm_standalone_bench/run_bench_serve.py` | sys.modules shim，按路径加载本地 `vllm_bench/serve.py` |
| `vllm_standalone_bench/vllm_bench/serve.py` | 流量引擎（并发/Poisson/分位/预热）+ `calculate_metrics` + `main_async`（返回 result dict） |
| `vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py` | 请求构造 + SSE 解析，产出 `RequestFuncOutput` |

---

## 3. 根因分析（已用静态分析 + 历史 CSV 佐证，并经代码审计确认）

### 3.1 Bug ①（用户症状的直接原因）：avg 列回显配置值

数据流断裂点：
1. `serve.py:1012-1013` 把每请求 `input_lens` / `output_lens` 放入 result；
2. `serve.py:2025-2039` 在 `not args.save_detailed` 时**删除**这两个列表；
3. `run_bench_multi.py:152-153` 设置 `save_result=False`、`save_detailed=False`；
4. `run_bench_multi.py:198-201` 的 `_extract_row` 读 `result.get('input_lens')` 得 `None` → **回退**到传入的 `in_len` / `out_len`。

证据：历史 CSV 每行 `avg_output_tokens` 严格等于 `output_len`；前缀场景 `prefix_ratio=0.8` 时真实输入应 ≈ `input_len + prefix_tokens`（128→约 230），但列里仍是 128。

> 注意：`throughput_tok_s` 列是**真实测算**的（`output_throughput = Σ实际 completion_tokens / 时长`，`serve.py:584`，不受删除影响）。反算最新 CSV：`246 tok/s ÷ 1.92 req/s = 128 token/req`，与 `output_len=128` 吻合——说明那次 vLLM 服务端确实如实生成 128 token。即 Bug ① 只让 avg 列失真，未污染吞吐。

### 3.2 Bug ②：completions 与 chat 解析 usage 不一致
- `endpoint_request_func.py:263` completions 用 `elif usage`（choices 与 usage 同帧时会漏读 `completion_tokens` → `output_tokens=0`）；
- `endpoint_request_func.py:443` chat 用 `if usage`（安全）。
- 实践中 vLLM/SGLang 都把 usage 放在 `choices=[]` 的独立末块，故 `elif` 当前能工作，但脆弱；网关/continuous_usage_stats 场景会踩。

### 3.3 潜在隐患：无 tokenizer 时 TTFT/TPOT/ITL 整列丢失
- `serve.py:1088` `if task_type == TaskType.GENERATION and tokenizer:` 才把 TTFT/TPOT/ITL 写入 result。无 tokenizer（`run_bench_multi.py` 在缺 `--tokenizer` 时置 `skip_tokenizer_init=True`）时，这些指标虽在 `calculate_metrics` 内算出，却不进 result → CSV 整列 0。
- 现状 `run_bench.sh` 总是提供 TOKENIZER，故为隐患而非现网故障。

---

## 4. 跨框架兼容性审计结论（Workflow 已对抗式复核，全部 CONFIRMED）

> 详见审计原始结果：`/tmp/claude-0/.../tasks/wvm22bl44.output`。关键引用均带 file:line。

### 4.1 兼容性矩阵

| 维度 | vLLM | SGLang | 本工具现状 |
|---|---|---|---|
| 强制输出到指定长度 | completions 读 `max_tokens`（**漏发默认 16**）；chat 读 `max_completion_tokens`（回退 `max_tokens`） | 完全一致；chat 两字段都不发→默认 **128** | ✅ 已发对字段 |
| `ignore_eos` | 两端点都认（顶层 bool，流入 `SamplingParams`） | 两端点都认（**且会顺带禁 `stop_token_ids`**，`schedule_batch.py:1300`） | ✅ 已发 `ignore_eos:true` |
| 流式 usage | `include_usage=true`→末块 `choices=[]` + `{prompt_tokens, completion_tokens, total_tokens}` 后接 `[DONE]` | 同左（**甚至不传 include_usage 也发**） | ✅ 已发 `include_usage:true` |
| 推理 token 流字段 | `delta.reasoning`（非 `reasoning_content`） | `delta.reasoning_content` | ✅ 两个都读（`endpoint_request_func.py:424`） |
| completion_tokens 是否含推理 token | 含（无拆分；仅 `/v1/responses` 有 `reasoning_tokens`） | 含（顶层另有 `usage.reasoning_tokens`） | 口径一致 |

**核心结论：在 OpenAI 协议线上层面，本工具当前发出的请求字段已经能让 vLLM 与 SGLang 双方"按指定长度输出 + 如实上报 token 数"。跨框架兼容本已成立，用户的"统计不正常"是客户端本地 Bug ①，与框架无关。**

### 4.2 必须防住的 gotcha
1. vLLM completions 默认 `max_tokens=16`、SGLang chat 默认 128 —— 本工具总发 `output_len` 已规避，需**自检**防止配置漏发时静默截断。
2. `ignore_eos` 在 SGLang 会顺带禁 `stop_token_ids`（本工具不用，无影响，需文档说明）。
3. 推理模型 `completion_tokens` 含推理 token；TTFT 计到首 token（含推理）——需统一口径说明。
4. SGLang OpenAI 端点 `extra='ignore'`（未知字段静默丢弃，无 `extra_body` 透传）——采样参数须以顶层字段发送。
5. 末 usage 帧 `choices=[]`（空数组非 null）——解析器须容忍空 choices。

---

## 5. 设计（方案 C：混合加固）

### 5.1 设计原则与边界
- **不重写流量引擎**：保留 `vllm_bench/serve.py` 的并发/定时/分位/预热逻辑。
- **不做 per-framework payload 分叉**：统一字段已验证对两框架生效。
- **把"服务端是否真兑现指定长度"从黑盒变成有数据 + 有告警**。

### 5.2 模块边界（单一职责，可独立测试）
- **请求/解析层** `endpoint_request_func.py`：忠实还原每次请求的 token 数与停止原因（新增 `finish_reason`）。
- **流量+指标引擎** `serve.py`：并发/分位/吞吐 + 聚合 `finish_reason` 计数。
- **编排+落盘层** `run_bench_multi.py`：真实 avg 提取、合规比值、CSV/XLSX、约束过滤。
- **自检层**（新增 `probe.py` 或并入 multi）：启动探测 `usage` 可用性，决定 token 计数口径。

### 5.3 数据流
统一字段请求 → `serve.py` 并发发送 → 解析层每请求取 `usage.completion_tokens/prompt_tokens` + `finish_reason` → `calculate_metrics` 聚合（保留 `total_input_tokens`/`total_output_tokens`，新增 `finish_reason` 计数）→ `main_async` 返回 result（总量字段本就存活）→ `_extract_row` 算真实 avg + 合规比值 → CSV/XLSX。

### 5.4 具体修改点

**a) `run_bench_multi.py::_extract_row`（核心，修 Bug ①）**
- `avg_in/avg_out` 改用存活的 `total_input_tokens` / `total_output_tokens` ÷ `completed`（不再回退 requested）。
- 列新增：`avg_input_measured`、`avg_output_measured`、`output_compliance`（= measured/requested，%）、`finish_reason_length_pct`（`finish_reason=="length"` 占比）、`token_source`（`usage` / `tokenizer` / `none`）。
- 原 `input_len` / `output_len` 保留作 requested；prefix 场景新增 `total_input_len = input_len + prefix_tokens`（消除"128 实为 230"误导）。
- CSV/XLSX 表头（`CSV_HEADERS` / `CSV_HEADERS_ZH`，`run_bench_multi.py:244-268`）同步扩展。

**b) `endpoint_request_func.py`（修 Bug ② + 补 finish_reason）**
- completions 解析 `elif usage` → `if usage`（`:263`）。
- `RequestFuncOutput` 新增字段 `finish_reason: str = ""`；completions（`:246` 附近）与 chat（`:418` 附近）从末帧 `choices[0].finish_reason` 读入。

**c) `serve.py`（两处微调）**
- `calculate_metrics`（`:407`）：增加 `finish_reason` 计数，写入 result（如 `finish_reason_length` 计数）。
- 指标门控（`:1088`）：TTFT/E2EL 总输出；TPOT/ITL 在"有 usage 上报的 output_tokens 或有 tokenizer"时输出——两框架都报 usage，故无 tokenizer 也能出 TPOT。

**d) 新增启动自检（probe）**
首组配置前发 1 个流式请求探测 `usage`（**复用 `serve.py` 现有的首组 ready-check/warmup 请求**，不额外增加请求数；当 `ready_check_timeout_sec=0` 时退化为单独发 1 探测请求）：
- 拿到 → 走 `usage` 口径；
- 拿不到 → 告警 + 降级为 tokenizer 重编码计数（若有），并在每行打 `token_source` 标记，**绝不静默置 1**。

**e) 文档**
两框架差异表（max_tokens 默认值 / ignore_eos 语义 / reasoning token 口径 / 流字段名）写入本 spec 第 4 节及 README。

### 5.5 错误处理
- 服务端不发 usage → 自检降级 + 标记 `token_source`；无 tokenizer 则告警 + 合规判定跳过（不假装合规）。
- 合规阈值用新 CLI 参数 `--min-output-compliance`（默认 `0.95`，即 95%）：`output_compliance < 阈值` **或** `finish_reason=="length"` 占比 < 阈值 → 告警（提示 ignore_eos 未生效/配置错），可选像现有 `--max-ttft-ms`（`run_bench_multi.py:499`）那样跳过该组更高并发。
- `token_source` 取值优先级：`usage`（服务端上报）> `tokenizer`（降级重编码）> `none`（两者皆无，仅告警）。

---

## 6. 测试计划（TDD：先写测试再实现）

- **`_extract_row` 单测**：构造 result（带/不带 `input_lens`；带/不带 totals）→ 断言 avg 取自 totals 而非 requested；prefix 场景 `total_input_len` 正确。
- **解析层单测**：喂模拟 SSE（usage-only 帧 `choices=[]`；choices+usage 同帧）→ 断言 `output_tokens`/`finish_reason` 正确，completions 不漏 usage。
- **合规单测**：measured < requested → `output_compliance < 1` + 告警标志置位。
- **集成测试**：对本地 mock server（固定 usage）跑一组 → 断言 CSV `avg_output_measured` == 真实、`output_compliance` ≈ 100%。

测试位于 `vllm_standalone_bench/tests/`（新建），依赖仅 stdlib + 已有依赖。

---

## 7. 不在本次范围（Out of Scope）

- Burstiness(Gamma) / Ramp-up / Speculative Decoding 指标 / 多模态 / Timeline Plot（README 已标 ❌，保持不变）。
- 重写为 sglang `bench_serving` 那套双口径（`output_lens` + `retokenized_output_lens`）交叉校验——本次用 `usage` + 可选 tokenizer 降级覆盖；retokenized 交叉校验作为后续可选增强。
- 改造 `benchmark_tools/`（另一套旧工具）。

---

## 8. 验收标准

1. CSV `avg_input_measured` / `avg_output_measured` 反映服务端真实 token 数（与 requested 解耦）；打 vLLM 与 SGLang 两框架均正确。
2. `output_compliance` 列：ignore_eos 生效时 ≈ 100%；人为去掉 ignore_eos 时 < 100% 并告警。
3. 无 `--tokenizer` 时 TTFT/TPOT/ITL 不再整列 0。
4. 启动自检能识别"服务端不发 usage"并降级 + 标记，不静默失真。
5. 全部新增/修改有对应单测通过。
