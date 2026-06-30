# 基准测试「缓存命中率」指标 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `vllm_standalone_bench` 统计结果中新增 token 加权的「缓存命中率」指标（`cache_hit_rate` %）与陪伴列 `avg_cached_tokens`，数据取自响应 `usage.prompt_tokens_details.cached_tokens`。

**架构：** 沿现有 token 统计管线接力加一段——每请求在 `endpoint_request_func.py` 解析 `usage.cached_tokens` → `serve.py::calculate_metrics` 聚合成 `total_cached_tokens` → `run_bench_multi.py::_extract_row` 算出 `cache_hit_rate` / `avg_cached_tokens` → 落到 CSV / XLSX / 终端汇总表；`bench_compare.COMPARE_METRICS` 加入 `cache_hit_rate` 使多引擎对比表自动多出 `vllm__cache_hit_rate` / `sglang__cache_hit_rate` 列。

**技术栈：** Python 3.10、pytest 9.0.3、aiohttp、openpyxl。运行测试用 `python3 -m pytest`（注意：本环境 `python` 不在 PATH，必须用 `python3`）。

**规格：** `docs/superpowers/specs/2026-06-30-bench-cache-hit-rate-design.md`

**worktree：** `.worktrees/bench-cache-hit-rate`（分支 `feat/bench-cache-hit-rate`）。所有命令在 `vllm_standalone_bench/` 下执行：`cd /Resource_Planning_Tool/.worktrees/bench-cache-hit-rate/vllm_standalone_bench`。

**基线：** 219 passed（实现前已确认绿色）。

---

## 文件结构与职责

| 文件 | 职责 | 本次改动 |
|---|---|---|
| `vllm_bench/lib/endpoint_request_func.py` | 每请求解析响应 `usage`，产出 `RequestFuncOutput` | 加 `cached_tokens` / `cached_reported` 字段 + `_extract_cached_tokens` 辅助 + completions/chat 两路径调用 |
| `vllm_bench/serve.py` | 聚合每请求指标成 `BenchmarkMetrics`，输出结果字典 | `BenchmarkMetrics` 加 `total_cached_tokens` / `cached_reported_count`；`calculate_metrics` 累加；结果字典带出 |
| `run_bench_multi.py` | 从结果字典组装 CSV 行、定义列、打印终端表、写 XLSX 指标说明 | `_extract_row` 算两新列；`CSV_HEADERS` / `CSV_HEADERS_ZH` 加列；终端表加列；XLSX explain 加条目 |
| `bench_compare.py` | 多引擎结果对比聚合 | `COMPARE_METRICS` 追加 `cache_hit_rate` |
| `README.md` | 工程说明 | 补一句 prefix caching 前置条件 |
| `tests/test_endpoint_parse.py` | endpoint usage 解析单测 | 加 cached_tokens 嵌套/平铺/缺失用例 |
| `tests/test_serve_metrics.py` | 聚合单测 | 加 total_cached_tokens / cached_reported_count 用例 |
| `tests/test_extract_row.py` | 行组装单测 | 加命中率计算 + 边界用例；扩展 `_result` 与 `test_csv_headers_match_row_keys` |
| `tests/test_bench_compare.py` | 对比聚合单测 | CSV_HEADER 加列 + 断言 `vllm__cache_hit_rate` |

设计边界：每请求解析、聚合、行组装、对比各为独立单元，通过既有的 `RequestFuncOutput` → `BenchmarkMetrics` → 结果字典 → row 的数据契约通信。本次只在这条契约上「加字段」，不改既有字段语义。

---

## 任务 1：每请求解析 `cached_tokens`（endpoint 层）

**文件：**
- 修改：`vllm_bench/lib/endpoint_request_func.py`（`RequestFuncOutput` 数据类 @`108-126`；新增辅助函数；completions 路径 @`266-270`；chat 路径 @`453-457`）
- 测试：`tests/test_endpoint_parse.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_endpoint_parse.py` 末尾追加三个测试：

```python
def test_chat_parses_cached_tokens_nested():
    """OpenAI 标准：usage.prompt_tokens_details.cached_tokens（嵌套）。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 8,
                   "prompt_tokens_details": {"cached_tokens": 80}}},
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.cached_tokens == 80
    assert out.cached_reported is True


def test_completions_parses_cached_tokens_flat():
    """兼容：部分版本平铺为 usage.cached_tokens。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"text": "ab", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 8,
                   "cached_tokens": 60}},
        "[DONE]",
    )
    out = _run(completions_fn, RequestFuncInput, "/v1/completions", chunks)
    assert out.success
    assert out.cached_tokens == 60
    assert out.cached_reported is True


def test_cached_tokens_absent_keeps_zero_and_unreported():
    """服务端未上报 cached_tokens（如未开 prefix caching）：保持 0、reported=False。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.cached_tokens == 0
    assert out.cached_reported is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_endpoint_parse.py -v`
预期：3 个新测试 FAIL，报错 `AttributeError: 'RequestFuncOutput' object has no attribute 'cached_tokens'`（或 `cached_reported`）。

- [ ] **步骤 3：编写最少实现代码**

3a. 在 `RequestFuncOutput` 数据类（`endpoint_request_func.py:108-126`）的字段末尾（`finish_reason` 之后）新增两个字段：

```python
    finish_reason: str = ""  # 停止原因（"length"/"stop"/...），来自末帧 choices[0].finish_reason
    cached_tokens: int = 0   # 命中 prefix cache 的 prompt token 数（来自 usage.cached_tokens）
    cached_reported: bool = False  # 服务端是否在 usage 中上报了 cached_tokens 字段
```

3b. 在该文件顶部 import 区下方、`_validate_api_url` 之前（或任意模块级辅助函数区）新增辅助函数：

```python
def _extract_cached_tokens(usage: dict) -> int | None:
    """从 usage 取缓存命中的 prompt token 数。

    OpenAI 标准：嵌套于 prompt_tokens_details.cached_tokens；
    兼容回退：平铺于 usage.cached_tokens。
    任一存在即返回 int 值；都不存在返回 None（表示服务端未上报）。
    """
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return int(details["cached_tokens"])
    if usage.get("cached_tokens") is not None:
        return int(usage["cached_tokens"])
    return None
```

3c. completions 路径（`async_request_openai_completions`，约 `:266-270`，在解析 `prompt_tokens` 之后）追加：

```python
                            if usage := data.get("usage"):
                                if (ct := usage.get("completion_tokens")) is not None:
                                    output.output_tokens = ct
                                if (pt := usage.get("prompt_tokens")) is not None:
                                    output.prompt_len = pt
                                if (cached := _extract_cached_tokens(usage)) is not None:
                                    output.cached_tokens = cached
                                    output.cached_reported = True
```

3d. chat 路径（`async_request_openai_chat_completions`，约 `:453-457`，在解析 `prompt_tokens` 之后）追加同样的两行：

```python
                            if usage := data.get("usage"):
                                if (ct := usage.get("completion_tokens")) is not None:
                                    output.output_tokens = ct
                                if (pt := usage.get("prompt_tokens")) is not None:
                                    output.prompt_len = pt
                                if (cached := _extract_cached_tokens(usage)) is not None:
                                    output.cached_tokens = cached
                                    output.cached_reported = True
```

> 范围说明：audio（`async_request_openai_audio` @`564`）与 embeddings（`async_request_openai_embeddings` @`604`）路径不被本 bench 调用，且 embeddings 无 prefix cache 语义，本次不改（YAGNI）。

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_endpoint_parse.py -v`
预期：全部 PASS（原有 + 3 新增）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py \
        vllm_standalone_bench/tests/test_endpoint_parse.py
git commit -m "feat(bench): 每请求解析 usage.cached_tokens 为缓存命中指标取数"
```

---

## 任务 2：聚合 `total_cached_tokens`（serve 层）

**文件：**
- 修改：`vllm_bench/serve.py`（`BenchmarkMetrics` @`175-210`；`calculate_metrics` 累加循环 @`432-481` 与构造 @`589-629`；结果字典 @`1017-1040`）
- 测试：`tests/test_serve_metrics.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_serve_metrics.py` 中：

1a. 扩展 `_out()` 辅助（@`13-20`），增加两个关键字参数：

```python
def _out(success, *, output_tokens=0, finish_reason="", prompt_len=10,
         ttft=0.05, latency=1.0, itl=None, generated_text="abc",
         cached_tokens=0, cached_reported=False):
    return RequestFuncOutput(
        success=success, output_tokens=output_tokens,
        finish_reason=finish_reason, prompt_len=prompt_len,
        ttft=ttft, latency=latency, itl=itl or [0.05, 0.05],
        generated_text=generated_text,
        cached_tokens=cached_tokens, cached_reported=cached_reported,
    )
```

1b. 在文件末尾追加两个测试：

```python
def test_metrics_aggregate_cached_tokens():
    """total_cached_tokens 累加成功请求的 cached_tokens；cached_reported_count 计上报数。"""
    outputs = [
        _out(True, output_tokens=8, finish_reason="length",
             cached_tokens=80, cached_reported=True),    # 命中 80
        _out(True, output_tokens=8, finish_reason="length",
             cached_tokens=0, cached_reported=True),     # 上报了但命中 0
        _out(True, output_tokens=0, finish_reason="stop"),  # 未上报 cached
    ]
    inputs = [_req(100, 8) for _ in outputs]
    metrics, _lens = serve.calculate_metrics(
        input_requests=inputs, outputs=outputs, dur_s=2.0,
        tokenizer=None, selected_percentiles=[50, 90], goodput_config_dict={},
    )
    assert metrics.total_cached_tokens == 80       # 80 + 0 + 0
    assert metrics.cached_reported_count == 2      # 前两个上报


def test_metrics_cached_fields_default_zero():
    """无 cached 数据时两个字段为 0，且结果字典带 total_cached_tokens 键。"""
    outputs = [_out(True, output_tokens=8, finish_reason="length")]
    metrics, _ = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=None, selected_percentiles=[50], goodput_config_dict={},
    )
    assert metrics.total_cached_tokens == 0
    assert metrics.cached_reported_count == 0
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_serve_metrics.py -v`
预期：2 个新测试 FAIL，报错 `AttributeError: 'BenchmarkMetrics' object has no attribute 'total_cached_tokens'`。

- [ ] **步骤 3：编写最少实现代码**

3a. `BenchmarkMetrics` 数据类（`serve.py:175-210`）末尾（`tokenizer_fallback_count` 之后）新增两字段：

```python
    finish_reason_length: int = 0      # finish_reason == "length" 的成功请求数
    usage_reported_count: int = 0      # 服务端流式上报了 output_tokens 的成功请求数
    tokenizer_fallback_count: int = 0  # 未上报 usage 时用 tokenizer 回退统计的成功请求数
    total_cached_tokens: int = 0       # 命中 prefix cache 的 prompt token 总数（usage.cached_tokens 累计）
    cached_reported_count: int = 0     # 服务端上报了 cached_tokens 字段的成功请求数
```

3b. `calculate_metrics` 顶部局部累加器（`serve.py:432-437`）新增两个：

```python
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    finish_reason_length = 0
    usage_reported_count = 0
    tokenizer_fallback_count = 0
    total_cached = 0
    cached_reported_count = 0
```

3c. 在累加循环里（`serve.py:469`，`total_input += outputs[i].prompt_len` 那一行之后）追加累加：

```python
            actual_output_lens.append(output_len)
            total_input += outputs[i].prompt_len
            total_cached += outputs[i].cached_tokens
            if outputs[i].cached_reported:
                cached_reported_count += 1
```

3d. `BenchmarkMetrics(...)` 构造（`serve.py:589-629`）末尾（`tokenizer_fallback_count=...` 之后）追加两个实参：

```python
        finish_reason_length=finish_reason_length,
        usage_reported_count=usage_reported_count,
        tokenizer_fallback_count=tokenizer_fallback_count,
        total_cached_tokens=total_cached,
        cached_reported_count=cached_reported_count,
    )
```

3e. 结果字典（`serve.py:1017-1040`，`if isinstance(metrics, BenchmarkMetrics):` 分支）末尾（`tokenizer_fallback_count` 之后）追加键：

```python
            "finish_reason_length": metrics.finish_reason_length,
            "usage_reported_count": metrics.usage_reported_count,
            "tokenizer_fallback_count": metrics.tokenizer_fallback_count,
            "total_cached_tokens": metrics.total_cached_tokens,
            "cached_reported_count": metrics.cached_reported_count,
        }
```

> 说明：仅 `BenchmarkMetrics` 分支加（本 bench 走该分支）；`else`（EmbedBenchmarkMetrics）分支不加。`cached_reported_count` 暂只透出到结果字典、不进 CSV（诊断用，见规格 §4.2/§5）。

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_serve_metrics.py -v`
预期：全部 PASS。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/serve.py \
        vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "feat(bench): calculate_metrics 聚合 total_cached_tokens 与 cached_reported_count"
```

---

## 任务 3：行组装 `cache_hit_rate` / `avg_cached_tokens` + CSV 列

**文件：**
- 修改：`run_bench_multi.py`（`_extract_row` @`252-369`；`CSV_HEADERS` @`374-388`；`CSV_HEADERS_ZH` @`390-403`）
- 测试：`tests/test_extract_row.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_extract_row.py` 中：

1a. 扩展 `_result()` 辅助（@`50-62`），增加可选的 `total_cached` 参数：

```python
def _result(total_in=30, total_out=24, completed=3, usage_reported=3,
            finish_reason_length=3, tokenizer_fallback=0, total_cached=0):
    """构造 serve.main_async 风格的最小 result dict（仅本测试关心的键）。"""
    return {
        "completed": completed, "failed": 0,
        "total_input_tokens": total_in, "total_output_tokens": total_out,
        "total_cached_tokens": total_cached,
        "usage_reported_count": usage_reported,
        "tokenizer_fallback_count": tokenizer_fallback,
        "finish_reason_length": finish_reason_length,
        "num_prompts": completed,
        "request_throughput": 1.0, "output_throughput": 12.0,
        "duration": 2.0,
    }
```

1b. 在 `test_extract_row_prefix_total_input_len_uses_total_input_budget`（@`109`）之后追加命中率测试：

```python
def test_extract_row_cache_hit_rate_token_weighted():
    """cache_hit_rate = total_cached / total_input * 100（token 加权）；avg_cached per 请求。"""
    row = m._extract_row(
        _result(total_in=300, total_out=24, completed=3, total_cached=150),
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["cache_hit_rate"] == 50.0        # 150 / 300 * 100
    assert row["avg_cached_tokens"] == 50.0     # 150 / 3


def test_extract_row_cache_hit_rate_zero_when_no_cache():
    """无缓存数据（total_cached=0）→ 命中率 0、avg 0，不报错。"""
    row = m._extract_row(
        _result(total_in=300, total_out=24, completed=3, total_cached=0),
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["cache_hit_rate"] == 0.0
    assert row["avg_cached_tokens"] == 0.0


def test_extract_row_cache_hit_rate_safe_when_totals_zero():
    """全失败（completed=0、total_in=0）或键缺失 → 命中率回退 0，不抛除零。"""
    # completed=0 / total_in=0
    row_zero = m._extract_row(
        _result(total_in=0, total_out=0, completed=0, total_cached=0,
                usage_reported=0, finish_reason_length=0),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row_zero["cache_hit_rate"] == 0.0
    assert row_zero["avg_cached_tokens"] == 0.0
    # 结果字典缺 total_cached_tokens 键（旧 serve 产物）
    row_missing = m._extract_row(
        {"completed": 3, "total_input_tokens": 300, "total_output_tokens": 24,
         "usage_reported_count": 3, "tokenizer_fallback_count": 0,
         "finish_reason_length": 3, "num_prompts": 3,
         "request_throughput": 1.0, "output_throughput": 12.0, "duration": 2.0},
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row_missing["cache_hit_rate"] == 0.0
    assert row_missing["avg_cached_tokens"] == 0.0
```

1c. 扩展 `test_csv_headers_match_row_keys`（@`169-188`）的 required 清单，把两个新列锁进表头。在现有 `for required in (...)` 元组里追加 `'avg_cached_tokens'`, `'cache_hit_rate'`：

```python
    for required in ("total_input_len", "input_compliance", "output_compliance",
                     "finish_reason_length_pct", "token_source", "seed",
                     "input_throughput_tok_s", "prefill_effective_tok_s",
                     "decode_effective_tok_s",
                     "avg_cached_tokens", "cache_hit_rate"):
        assert required in m.CSV_HEADERS, f"新列 {required} 未进 CSV_HEADERS"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_extract_row.py -v`
预期：新测试 FAIL（`row` 无 `cache_hit_rate` 键 → KeyError），且 `test_csv_headers_match_row_keys` 因新列不在 CSV_HEADERS 而 FAIL。

- [ ] **步骤 3：编写最少实现代码**

3a. `_extract_row`（`run_bench_multi.py:284-292` 附近，已有 `total_in`、`completed`）之后新增缓存计算：

```python
    completed = _i('completed')
    total_in = _i('total_input_tokens')
    total_out = _i('total_output_tokens')
    total_cached = _i('total_cached_tokens')
```

并在 `avg_in` / `avg_out` 计算（@`291-292`）之后追加：

```python
    avg_in = round(total_in / completed, 1) if completed > 0 else 0.0
    avg_out = round(total_out / completed, 1) if completed > 0 else 0.0
    avg_cached_tokens = round(total_cached / completed, 1) if completed > 0 else 0.0
    cache_hit_rate = (
        round(total_cached / total_in * 100, 1) if total_in > 0 else 0.0
    )
```

3b. 在返回字典（@`324-369`）的 `'token_source': token_source,` 之后插入两键：

```python
        'input_compliance':    input_compliance,
        'output_compliance':   output_compliance,
        'finish_reason_length_pct': finish_reason_length_pct,
        'token_source':        token_source,
        'avg_cached_tokens':   avg_cached_tokens,   # 平均命中缓存的 prompt token 数
        'cache_hit_rate':      cache_hit_rate,       # token 加权缓存命中率 (%) = total_cached/total_in*100
        # ── 吞吐量 ──────────────────────────────────
```

3c. `CSV_HEADERS`（@`374-388`）在 `'token_source',` 之后插入两列：

```python
    'input_compliance', 'output_compliance',
    'finish_reason_length_pct', 'token_source',
    'avg_cached_tokens', 'cache_hit_rate',
    'throughput_req_s', 'throughput_tok_s', 'input_throughput_tok_s',
```

3d. `CSV_HEADERS_ZH`（@`390-403`）在 `'token来源',` 之后插入对应中文：

```python
    '输入长度合规(%)', '输出长度合规(%)', 'length停止占比(%)', 'token来源',
    '平均缓存命中tokens', '缓存命中率(%)',
    '请求吞吐(req/s)', '输出Token系统吞吐(tok/s)', '输入Token系统吞吐(tok/s)',
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_extract_row.py -v`
预期：全部 PASS（含 `test_csv_headers_match_row_keys` 的中英文表头等长断言）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py \
        vllm_standalone_bench/tests/test_extract_row.py
git commit -m "feat(bench): _extract_row 产出 cache_hit_rate 与 avg_cached_tokens 两列"
```

---

## 任务 4：终端汇总表 + XLSX 指标说明页

**文件：**
- 修改：`run_bench_multi.py`（终端汇总表 @`852-878`；XLSX 指标说明 `explain` @`472-488`）

> 这一层是展示，既有代码不为 print/写表写单测；用「全量 pytest 绿 + 表头对齐人工核对」验证。

- [ ] **步骤 1：终端汇总表加列**

1a. 表头 f-string（`run_bench_multi.py:855-860`）在 `{'req/s':>8}` 之后插入缓存列：

```python
    print(
        f"{'输入':>6} {'输出':>6} {'并发':>5} "
        f"{'out_sys':>10} {'in_sys':>10} {'prefill':>10} {'decode':>10} {'req/s':>8} "
        f"{'命中%':>8} "
        f"{'TTFT均值':>10} {'TTFT_P90':>10} "
        f"{'TPOT均值':>10} {'E2EL均值':>10} {'成功':>6}"
    )
```

1b. 行 f-string（`run_bench_multi.py:863-872`）在 `{r['throughput_req_s']:>8.3f}` 之后插入：

```python
        print(
            f"{r['input_len']:>6} {r['output_len']:>6} {r['parallel_num']:>5} "
            f"{r['throughput_tok_s']:>10.1f} "
            f"{r['input_throughput_tok_s']:>10.1f} "
            f"{r['prefill_effective_tok_s']:>10.1f} "
            f"{r['decode_effective_tok_s']:>10.1f} "
            f"{r['throughput_req_s']:>8.3f} "
            f"{r['cache_hit_rate']:>8.1f} "
            f"{r['ttft_mean_ms']:>10.1f} {r['ttft_p90_ms']:>10.1f} "
            f"{r['tpot_mean_ms']:>10.3f} {r['e2el_mean_ms']:>10.1f} {r['n_success']:>6}"
        )
```

- [ ] **步骤 2：XLSX 指标说明页加条目**

在 `explain` 列表（`run_bench_multi.py:472-488`）的 `decode_effective_tok_s` 条目之后、`P50/P90/P99` 条目之前插入两条：

```python
        ('decode_effective_tok_s', 'Decode 有效速率', '1 ÷ mean_TPOT_s；基于 TPOT 的 next-token decode 近似速率'),
        ('avg_cached_tokens', '平均缓存命中 tokens', 'total_cached_tokens ÷ completed（服务端 usage.cached_tokens 累计）'),
        ('cache_hit_rate', '缓存命中率(%)', 'total_cached_tokens ÷ total_input_tokens × 100（token 加权；仅服务端开启 prefix caching 时非零）'),
        ('P50/P90/P99', '百分位数', 'P90 表示 90% 请求低于该延迟值'),
```

- [ ] **步骤 3：验证（全量测试 + 列对齐）**

运行：`python3 -m pytest -q`
预期：219（基线）+ 本特性新增用例，全部 PASS。

人工核对（grep 确认列在两处表头出现且顺序一致）：

```bash
grep -n "cache_hit_rate\|avg_cached_tokens\|命中%\|缓存命中率" run_bench_multi.py
```

预期：命中 `CSV_HEADERS`、`CSV_HEADERS_ZH`、`_extract_row` 返回字典、终端表头、终端行、XLSX explain 各处。

- [ ] **步骤 4：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py
git commit -m "feat(bench): 终端汇总表与 XLSX 指标说明展示缓存命中率"
```

---

## 任务 5：`bench_compare` 对比表纳入 `cache_hit_rate`

**文件：**
- 修改：`bench_compare.py`（`COMPARE_METRICS` @`18`）
- 测试：`tests/test_bench_compare.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_bench_compare.py` 中：

1a. 扩展 `CSV_HEADER`（@`9-14`），在 `avg_output_tokens` 之后插入两列：

```python
CSV_HEADER = (
    "model,backend,input_len,output_len,parallel_num,epochs,num_prompts,n_success,"
    "n_failed,avg_input_tokens,avg_output_tokens,avg_cached_tokens,cache_hit_rate,"
    "throughput_req_s,throughput_tok_s,"
    "ttft_mean_ms,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,tpot_mean_ms,tpot_p50_ms,"
    "tpot_p90_ms,tpot_p99_ms,e2el_mean_ms,e2el_p50_ms,e2el_p90_ms,e2el_p99_ms,duration_s"
)
```

1b. 扩展 `_write_result_csv`（@`17-24`），在 `avg_output_tokens`(=32) 之后、`throughput_req_s`(=1.0) 之前写入两个新值，并新增 `hitrate` 形参：

```python
def _write_result_csv(path: Path, parallel: int, ttft_p50: int, tput: int,
                      hitrate: float = 80.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        CSV_HEADER + "\n"
        f"m,openai-chat,64,32,{parallel},1,1,1,0,64,32,51.2,{hitrate},"
        f"1.0,{tput},10,{ttft_p50},20,30,5,4,6,8,50,40,60,70,10\n",
        encoding="utf-8",
    )
```

1c. 在 `test_aggregate_aligns_two_engines_and_preserves_originals`（@`39`）里给两个引擎传不同命中率，并在断言末尾追加 cache 列断言：

```python
    _write_result_csv(vllm_csv, parallel=1, ttft_p50=11, tput=100, hitrate=80.0)
    _write_result_csv(sglang_csv, parallel=1, ttft_p50=22, tput=200, hitrate=40.0)
```

```python
    assert row["vllm__ttft_p50_ms"] == "11"
    assert row["sglang__ttft_p50_ms"] == "22"
    assert row["vllm__cache_hit_rate"] == "80.0"
    assert row["sglang__cache_hit_rate"] == "40.0"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_bench_compare.py -v`
预期：`test_aggregate_aligns_two_engines_and_preserves_originals` FAIL（`row["vllm__cache_hit_rate"]` 为 `""` 或 KeyError——因 `cache_hit_rate` 不在 `COMPARE_METRICS`，对比表无此列）。

- [ ] **步骤 3：编写最少实现代码**

`bench_compare.py:18`，`COMPARE_METRICS` 元组追加 `"cache_hit_rate"`：

```python
COMPARE_METRICS = ("throughput_tok_s", "ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms", "cache_hit_rate")
```

> `_compare_fieldnames` / `_build_compare_rows` 自动产出 `vllm__cache_hit_rate` / `sglang__cache_hit_rate` 列，无需其他改动。**不动** `PLOT_METRICS`（规格 §3 Out of scope）。

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_bench_compare.py -v`
预期：全部 PASS。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/bench_compare.py \
        vllm_standalone_bench/tests/test_bench_compare.py
git commit -m "feat(bench): bench_compare 对比表纳入 cache_hit_rate 指标"
```

---

## 任务 6：README 前置条件 + 全量验证 + 收尾

**文件：**
- 修改：`README.md`

- [ ] **步骤 1：补 README 说明**

在 `README.md` 中「prefix_ratio / 前缀缓存」相关章节（搜索 `prefix` 定位）补一句前置条件，说明 `cache_hit_rate` 非零需要服务端开启缓存。建议措辞：

```markdown
> **缓存命中率（`cache_hit_rate`）**：统计结果中的 `cache_hit_rate` / `avg_cached_tokens`
> 取自响应 `usage.cached_tokens`。该值非零需要服务端开启前缀缓存
> （vLLM `--enable-prefix-caching`；SGLang 对应缓存开关）；未开启时命中率为 0。
```

> 实现时按 README 既有章节结构与措辞收敛插入位置。

- [ ] **步骤 2：全量测试**

运行：`cd /Resource_Planning_Tool/.worktrees/bench-cache-hit-rate/vllm_standalone_bench && python3 -m pytest -q`
预期：全部 PASS（基线 219 + 新增用例，0 失败）。

- [ ] **步骤 3：Commit**

```bash
git add vllm_standalone_bench/README.md
git commit -m "docs(bench): README 补充缓存命中率指标的前置条件"
```

- [ ] **步骤 4：收尾检查（finishing-a-development-branch）**

- `git -C .worktrees/bench-cache-hit-rate log --oneline main..HEAD` 确认 6 个提交齐全。
- 全量 `python3 -m pytest -q` 再跑一次绿。
- 用 `git diff --check` 确认无空白错误。
- 其后按 `AGENTS.md` 标准流程第 6 步：回 `main` → `git merge --no-ff feat/bench-cache-hit-rate` → 清理 worktree（由 finishing-a-development-branch 技能引导，需真实 GPU+镜像的 smoke 验收由用户在合并后执行）。

---

## 自检（writing-plans 内联检查）

**1. 规格覆盖度：**
- §4.1 数据源 usage.cached_tokens → 任务 1 ✅
- §4.2 指标定义（token 加权 rate + avg_cached + cached_reported_count）→ 任务 1/2/3 ✅
- §4.3 五文件改动：endpoint(任务1)、serve(任务2)、run_bench_multi 行+列(任务3)、终端表+XLSX(任务4)、bench_compare(任务5)、README(任务6) ✅
- §4.4 列 schema（token_source 后插两列）→ 任务 3 ✅
- §5 边界（completed=0、total_in=0、缺键、未上报）→ 任务 1(test_cached_tokens_absent) + 任务 3(test_extract_row_cache_hit_rate_safe_when_totals_zero) ✅
- §6 测试策略（extract_row / serve_metrics / endpoint parse / bench_compare）→ 任务 1/2/3/5 ✅
- §8 验收：单测绿(各任务)、CSV 新列(任务3)、prefix_ratio 命中趋势、未开缓存=0、对比表多引擎列(任务5)、旧 CSV 缺列 N/A（既有 `_build_compare_rows` 用 `.get(metric, "")`，单引擎缺引擎填 N/A 已被既有测试覆盖）✅
  - 真实 smoke 验收（GPU+镜像）属合并后手动验收，非本计划代码任务。

**2. 占位符扫描：** 无 TODO/待定/「类似任务N」；每个代码步骤都给出完整代码块。✅

**3. 类型/命名一致性：**
- `RequestFuncOutput.cached_tokens` / `cached_reported` — 任务 1 定义，任务 2 (`outputs[i].cached_tokens`、`outputs[i].cached_reported`) 与测试 `_out()` 使用一致 ✅
- `BenchmarkMetrics.total_cached_tokens` / `cached_reported_count` — 任务 2 定义、构造、结果字典键 `"total_cached_tokens"` 一致；任务 3 `_i('total_cached_tokens')` 读取键一致 ✅
- 行键与列名 `'avg_cached_tokens'` / `'cache_hit_rate'` — 任务 3 `_extract_row` 返回键、`CSV_HEADERS`、终端表 `r['cache_hit_rate']`、`bench_compare.COMPARE_METRICS` 全部一致 ✅
- 辅助函数 `_extract_cached_tokens` — 任务 1 定义并调用，返回 `int | None`，调用处 `is not None` 判断一致 ✅
