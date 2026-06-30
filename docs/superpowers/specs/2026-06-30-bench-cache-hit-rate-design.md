# vLLM/SGLang 基准测试新增「缓存命中率」指标 — 设计文档

- 日期：2026-06-30
- 分支：`feat/bench-cache-hit-rate`
- worktree：`.worktrees/bench-cache-hit-rate`
- 状态：待审查（brainstorming 产物，下一步进入 writing-plans）

## 1. 背景与目标

现有 `vllm_standalone_bench` 统计结果里只有延迟（TTFT/TPOT/E2EL）与吞吐（req/s、tok/s）两类指标，以及 `prefix_ratio` / `prefix_tokens` 两个**输入配置**字段（用户配了多少前缀共享）。**缺一个实测的「服务端 KV 缓存（prefix cache）命中比例」指标**——即每个请求的 prompt 里有多少 token 是从缓存直接复用、而非重新 prefill 的。

用户诉求：在统计结果（CSV / XLSX / 控制台汇总 / 多引擎对比表）中新增**缓存命中率**指标，用来量化开启 `--enable-prefix-caching` 后、不同 `prefix_ratio` / 并发下的真实命中收益，并在 vLLM vs SGLang 之间可比。

vLLM 与 SGLang 的 OpenAI 兼容响应都在 `usage` 里携带缓存命中信息（`prompt_tokens_details.cached_tokens`），而本工程的请求函数已经在解析同一个 `usage` 字典以拿 `prompt_tokens` / `completion_tokens`——所以这是沿现有 token 统计管线「再加一段」的自然扩展，无需新开数据通道。

## 2. 现状分析（数据流）

```
每请求 usage 解析                  聚合                       行组装                     产出
endpoint_request_func.py  →  vllm_bench/serve.py  →  run_bench_multi.py  →  CSV/XLSX/控制台/对比表
RequestFuncOutput            BenchmarkMetrics         _extract_row
.prompt_len / .output_tokens total_input / total_output  → result.csv 行
```

核对到具体落点：

| 环节 | 现状 | 缺口 |
|---|---|---|
| 每请求载体 `RequestFuncOutput`（`endpoint_request_func.py:111`） | 已有 `prompt_len` / `output_tokens`，从 `usage` 解析（路径约 `:266` / `:453` / `:564` / `:606`） | **缺 `cached_tokens` 字段**；解析时丢弃了 `prompt_tokens_details.cached_tokens` |
| 聚合载体 `BenchmarkMetrics`（`serve.py:175`）+ `get_metrics`（`:418`） | `total_input += outputs[i].prompt_len`（`:469`）；有 `usage_reported_count`（`:450`）镜像「服务端是否上报 output_tokens」的诚实口径 | **缺 `total_cached_tokens` 与 `cached_reported_count`** |
| 结果字典出口（`serve.py:1017` / `:1042`） | `"total_input_tokens": metrics.total_input` | **缺 `"total_cached_tokens"` 键** |
| 行组装 `_extract_row`（`run_bench_multi.py:252`） | 从总量算 `avg_input_tokens = total_in/completed`、`throughput = total/duration` | **不算命中率** |
| 列定义 `CSV_HEADERS` / `CSV_HEADERS_ZH`（`:374` / `:390`） | 无缓存列 | **缺两列** |
| 终端汇总表（`:852`）/ XLSX 指标说明页（`:472`） | 无缓存条目 | **缺条目** |
| 多引擎对比 `bench_compare.py` | `COMPARE_METRICS`（`:18`）驱动 `{engine}__<metric>` 列；`_compare_fieldnames` / `_build_compare_rows` 自动跟进 | **未包含命中率** |

**结论**：改动是沿着现有 token 统计管线「接力加一段」，5 个文件，纯增量、不改语义、不引入新依赖。

## 3. 范围

### In scope

- 每请求解析 `usage.prompt_tokens_details.cached_tokens`（兼容平铺 `usage.cached_tokens`）。
- 聚合 `total_cached_tokens` + `cached_reported_count`（镜像 `usage_reported_count`）。
- `_extract_row` 产出 token 加权 `cache_hit_rate(%)` 与陪伴列 `avg_cached_tokens`。
- `CSV_HEADERS` / `CSV_HEADERS_ZH` 新增两列；终端汇总表 + XLSX 指标说明页加条目。
- `bench_compare.COMPARE_METRICS` 加入 `cache_hit_rate`（多引擎对比表自动多出 `vllm__cache_hit_rate` / `sglang__cache_hit_rate` 列）。
- 单测（解析、聚合、行组装含边界）。

### Out of scope（YAGNI）

- **不做** 抓取 vLLM `/metrics`（Prometheus）的 `gpu_prefix_cache_hits/queries` 口径——响应 `usage.cached_tokens` 已是每请求、按行可归因的更优来源，且 vLLM/SGLang 通用。
- **不做** 每请求比率的算术平均口径——token 加权（`Σcached / Σprompt`）与现有吞吐量同源、跨配置可比，且不被短 prompt 主导。
- **不进** `PLOT_METRICS` 绘图——命中率 0–100% 与 tok/s、ms 不同量纲，混绘需独立面板，留作后续。
- **不自动开启** 服务端 `--enable-prefix-caching`——命中率是否非零取决于服务端配置，仅在 README 提示前置条件。
- **不做** `cache_hit_rate` 的 `change_pct`（现 `bench_compare.py` 本就不算 change_pct，只并排各引擎值）。

## 4. 设计

### 4.1 数据源

每请求响应的 `usage`（OpenAI 兼容，vLLM/SGLang 均上报）：

```jsonc
"usage": {
  "prompt_tokens": 1024,
  "completion_tokens": 128,
  "prompt_tokens_details": { "cached_tokens": 819 }   // ← 新取此字段
}
```

兼容回退：部分版本/引擎可能平铺为 `usage.cached_tokens`，解析时按「嵌套优先，平铺兜底」取值，取不到记 0。

### 4.2 指标定义

- **`avg_cached_tokens`** = `total_cached_tokens / completed`（round 1；`completed == 0` → `0.0`，沿用现有回退约定）。
- **`cache_hit_rate`**（值域 0–100，%）= `total_cached_tokens / total_input_tokens * 100`（round 1；`total_input_tokens == 0` → `0.0`）。
  - **token 加权**口径，与 `throughput_tok_s = total_output/duration`、`avg_input_tokens = total_in/completed` 同源；跨配置（不同 `prefix_ratio` / 并发 / 输入长度）可比。
- `cached_reported_count`：成功请求里服务端上报过 `cached_tokens` 的数量。用于内部诊断（区分「服务端没上报缓存」与「上报了但命中 0」），镜像现有 `usage_reported_count` 的诚实风格；不单独占 CSV 列，仅日志可见。

### 4.3 各文件改动（5 处，纯增量）

**`vllm_bench/lib/endpoint_request_func.py`**

- `RequestFuncOutput`（`:111`）新增字段：`cached_tokens: int = 0`。
- 在所有解析 `usage` 的路径（约 `:266` / `:453` / `:564` / `:606`）追加：
  ```python
  if (cached := (usage.get("prompt_tokens_details") or {}).get("cached_tokens")) is not None \
          or (cached := usage.get("cached_tokens")) is not None:
      output.cached_tokens = int(cached)
  ```
  （实现时按各路径既有取值风格收敛，确保 streaming / non-streaming / chat / completions 一致。）

**`vllm_bench/serve.py`**

- `BenchmarkMetrics`（`:175`）新增：`total_cached_tokens: int = 0`、`cached_reported_count: int = 0`。
- `get_metrics`（`:418`）：在累加 `total_input` 的循环里同步 `total_cached += outputs[i].cached_tokens`；当某请求的 `usage` 中 `cached_tokens` 字段**存在（被服务端上报）**时 `cached_reported_count += 1`，无论其值为 0 或正（口径与 `usage_reported_count` 对齐——计「是否上报」而非「是否 >0」）。
- 结果字典出口（`:1017` / `:1042`）增 `"total_cached_tokens": metrics.total_cached_tokens`（两处分支都加）。

**`run_bench_multi.py`**

- `_extract_row`（`:252`）：取 `total_cached = _i('total_cached_tokens')`，计算并返回两个新键：
  - `'avg_cached_tokens'`: `round(total_cached / completed, 1) if completed > 0 else 0.0`
  - `'cache_hit_rate'`: `round(total_cached / total_in * 100, 1) if total_in > 0 else 0.0`（`total_in` 复用现有 `total_input_tokens`）。
- `CSV_HEADERS`（`:374`）：在 `token_source` 之后插入 `'avg_cached_tokens', 'cache_hit_rate'`（形成「缓存」小分组，紧跟请求统计组）。
- `CSV_HEADERS_ZH`（`:390`）：对应插 `「平均缓存命中tokens」, 「缓存命中率(%)」`。
- 终端汇总表（`:852`）：表头与行追加命中率列。
- XLSX 指标说明页 `explain`（`:472`）：加两条指标释义。

**`bench_compare.py`**

- `COMPARE_METRICS`（`:18`）追加 `"cache_hit_rate"`。
- `_compare_fieldnames` / `_build_compare_rows` 自动产出 `vllm__cache_hit_rate` / `sglang__cache_hit_rate` 列（缺失引擎填 `N/A`），无需额外改动。
- **不动** `PLOT_METRICS`。

**`vllm_standalone_bench/README.md`**：补一句——命中率非零需服务端开启 `--enable-prefix-caching`（vLLM）/ SGLang 对应缓存开关。

### 4.4 列 schema（缓存小分组）

```
..., finish_reason_length_pct, token_source,
avg_cached_tokens, cache_hit_rate,           ← 新增
throughput_req_s, throughput_tok_s, ...
```

列名取 `cache_hit_rate`（值 0–100，对齐 `input_compliance` / `output_compliance` 的「无后缀 %」约定）；中文表头 `缓存命中率(%)` 明示单位。

### 4.5 数据流（加一段后）

```
每请求 usage → RequestFuncOutput.cached_tokens
  → get_metrics 累加 → BenchmarkMetrics.total_cached_tokens (+ cached_reported_count)
  → 结果字典 "total_cached_tokens"
  → _extract_row 计算 → 行 {avg_cached_tokens, cache_hit_rate}
  → CSV / XLSX / 终端汇总表
  → bench_compare（多引擎时）→ vllm__cache_hit_rate / sglang__cache_hit_rate 列
```

## 5. 错误处理 / 边界

- **服务端未开 prefix caching**：`cached_tokens` 缺失 → 记 0 → 总量 0 → `cache_hit_rate = 0%`。`cached_reported_count == 0` 时日志提示「服务端未上报 cached_tokens，命中率为 0 属未开启缓存」，避免误读为「真 0% 命中」。
- **`completed == 0`（全失败）**：`avg_cached_tokens` 回退 `0.0`，与 `avg_input_tokens` 同口径（不回退 requested，避免复发既有「Bug①」类问题）。
- **`total_input_tokens == 0`**：`cache_hit_rate` 回退 `0.0`。
- **旧结果 CSV（无新列）**：读取端（`bench_compare` 等）用 `DictReader` + `.get`，缺列返回 `None` → 对比表填 `N/A`，不报错。
- **解析稳健**：`prompt_tokens_details` 缺失或非 dict → 平铺兜底 → 仍缺 → 记 0，不抛异常。
- **数值类型**：服务端可能返回字符串/浮点；统一 `int(cached)` 容错。

## 6. 测试策略（TDD，先红后绿）

1. **`tests/test_extract_row.py`（扩展）**：
   - 喂含 `total_cached_tokens` 的 `result` → 断言行里 `cache_hit_rate` / `avg_cached_tokens` 正确（token 加权）。
   - 边界：`completed == 0`、`total_input_tokens == 0`、缺 `total_cached_tokens` 键 → 均回退 `0.0`，不抛异常。
2. **`tests/test_serve_metrics.py`（扩展）**：
   - 构造多个 `RequestFuncOutput`（含/不含 `cached_tokens`）→ 断言 `BenchmarkMetrics.total_cached_tokens` 与 `cached_reported_count` 累加正确。
3. **`tests/test_endpoint_cache_parse`（新增，或并入既有 endpoint 解析测试）**：
   - mock 两条 `usage` JSON：① 嵌套 `prompt_tokens_details.cached_tokens`；② 平铺 `cached_tokens`；③ 两处都缺 → 断言 `output.cached_tokens` 分别取到正确值 / 0。
4. **`tests/test_bench_compare.py`（扩展）**：
   - 两份 mock `result.csv`（含 `cache_hit_rate` 列、不同引擎）→ 断言 `compare.csv` 含 `vllm__cache_hit_rate` / `sglang__cache_hit_rate` 列且对齐正确；缺引擎列填 `N/A`。

## 7. 依赖与前置准备

- **无新依赖**：复用既有 `aiohttp` / `numpy` / `openpyxl` / `matplotlib`。
- **服务端前置**：命中率非零需 vLLM `--enable-prefix-caching`（或 SGLang 对应缓存开关）；README 补说明，不强制自动开启。

## 8. 验收标准

1. `pytest -q` 全绿（含新增/扩展用例）。
2. 真实 smoke（服务端开 prefix caching）：`result.csv` 出现 `avg_cached_tokens` / `cache_hit_rate` 两列，且 `prefix_ratio` 高的场景命中率显著高于 `prefix_ratio=0`（缓存生效的可观测信号；不要求严格单调，高并发下驱逐可有扰动）。
3. `prefix_ratio=0`（无共享前缀）场景命中率近 0；`--enable-prefix-caching` 关闭时命中率 0 且 `cached_reported_count=0`。
4. 服务端未开缓存 / 字段缺失时，两列平稳回退 0，不报错、不影响既有列。
5. 多引擎 run 的 `compare.csv` 含 `vllm__cache_hit_rate` / `sglang__cache_hit_rate` 列；单引擎 run 仅有 `vllm__cache_hit_rate`，不报错。
6. 旧 `result.csv`（无新列）被 `bench_compare` 读取时该指标列填 `N/A`，不中断聚合。

## 9. 风险

- **vLLM/SGLang `cached_tokens` 字段路径与上报条件**：嵌套 `prompt_tokens_details.cached_tokens` 为 OpenAI 标准；实现时需在真实 smoke 中确认两引擎的确切路径与「是否需额外开关才上报」，解析按嵌套优先 + 平铺兜底覆盖。
- **口径差异**：vLLM 与 SGLang 的 `cached_tokens` 统计口径可能略有差异（如是否含部分命中）。沿用各服务端自报口径，在对比表中如实呈现，不做归一化（与本工程既有 token 口径策略一致）。
- **`cached_reported_count` 判定阈值**：以「该请求 `cached_tokens` 字段存在且为非负整数」计上报，与 `usage_reported_count` 口径对齐；实现时复核边界。

## 10. 涉及文件清单

- 改：`vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py`（`RequestFuncOutput.cached_tokens` + 4 处 usage 解析）
- 改：`vllm_standalone_bench/vllm_bench/serve.py`（`BenchmarkMetrics` 两字段 + `get_metrics` 累加 + 结果字典出口两处）
- 改：`vllm_standalone_bench/run_bench_multi.py`（`_extract_row` + `CSV_HEADERS` / `CSV_HEADERS_ZH` + 终端汇总表 + XLSX 指标说明页）
- 改：`vllm_standalone_bench/bench_compare.py`（`COMPARE_METRICS` 增 `cache_hit_rate`）
- 改：`vllm_standalone_bench/README.md`（前置条件提示）
- 改/新：`vllm_standalone_bench/tests/`（`test_extract_row` / `test_serve_metrics` / `test_bench_compare` 扩展 + 新增 endpoint cache 解析测试）
