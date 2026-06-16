# vLLM/SGLang 基准测试 token 统计修复与双框架兼容加固 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复 `vllm_standalone_bench/` 工程的输出 token 统计失真（avg 列回显配置值），并加固为可同时正确压测 vLLM 与 SGLang 的统一基准工具——把"服务端是否真按指定长度输出"从黑盒变成有数据、有告警。

**架构：** 方案 C（混合加固）。保留 `vllm_bench/serve.py` 的流量引擎，改动聚焦三层：(1) 解析层 `endpoint_request_func.py`——补 `finish_reason`、修 completions 的 `elif usage`；(2) 引擎层 `serve.py`——聚合 `finish_reason`/`usage` 上报计数、放宽无 tokenizer 时的指标门控；(3) 编排层 `run_bench_multi.py`——真实 avg（用存活的总量字段）、合规列、token 来源、合规告警与跳过。跨框架靠统一字段（`max_tokens`/`max_completion_tokens` + `ignore_eos:true` + `include_usage:true`），**不做 per-framework 分叉**。

**技术栈：** Python 3.10+，aiohttp（流式 SSE），numpy，tqdm，pytest 9（测试）。无 vllm/torch 依赖（经 `run_bench_serve.py` 的 sys.modules shim）。

**规格：** `docs/superpowers/specs/2026-06-16-vllm-sglang-bench-token-stats-design.md`

---

## 规格偏离说明（相对 spec §5.4，均为意图等价的简化）

1. **"启动自检（probe）"→ 每行从 `usage_reported_count` 派生 `token_source`（任务 4+6）。**
   spec §5.4d 字面要求发一个探测请求判断服务端是否上报 usage。本计划改为：`serve.calculate_metrics` 统计 `usage_reported_count`（任务 4），`run_bench_multi` 据此逐行派生 `token_source`（任务 6）。
   - **为何改**：零额外请求（比 spec 的"复用 warmup"更省）、逐配置准确（而非全局假设）。
   - **意图保留**：spec §5.5 要求"不发 usage→降级 tokenizer + 标记 + 告警 + 跳过"。无 usage 且无 tokenizer 时，`token_source="none"`、`avg_output` 回退为 1/请求 → `output_compliance` 极低 → 触发 `--min-output-compliance` 告警与跳过（任务 7）。即"绝不静默置 1"通过合规机制兜底。
   - **取舍**：放弃"启动前一次性告警"的时序，换取实现简洁与零额外请求。若坚持启动前探测，可后续追加一个独立 `probe.py`（视为增强）。

2. **`avg_input_measured`/`avg_output_measured`（新列）→ 就地修正现有 `avg_input_tokens`/`avg_output_tokens`。**
   spec §5.4a 字面新增 `_measured` 列。本计划直接让现有 `avg_*_tokens` 列取真实值（任务 6），不新增冗余列。
   - **为何改**：避免两列同值；现有下游（如对比脚本）读到的列直接被修正，无回归。
   - **新增列**仅保留确有新信息的：`total_input_len`、`output_compliance`、`finish_reason_length_pct`、`token_source`。

---

## 关键背景（实现者必读）

- `serve.py`、`endpoint_request_func.py` 是从 `vllm-main` 复制来的**供应商副本（vendored copy）**。本计划对它们做局部定点修改；若日后从上游重新拷贝，需重新套用这些修改。
- 测试导入时必须先 `import run_bench_serve`——它在模块级注入全部 sys.modules shim（含 `regex → stdlib re`），并按路径加载 `serve.py`。之后 `run_bench_serve._serve` 即 serve 模块，`_serve.ASYNC_REQUEST_FUNCS` / `_serve.RequestFuncInput` / `_serve.RequestFuncOutput` 可用。`run_bench_multi` 模块导入时也会 exec `run_bench_serve`，效果相同。
- 历史背景：`avg_input_tokens`/`avg_output_tokens` 之所以永远等于配置值，是因为 `serve.py` 在 `save_detailed=False`（`run_bench_multi.py` 默认）时删除了每请求 `input_lens`/`output_lens`，而 `_extract_row` 回退到 requested 值。**修复用存活字段 `total_input_tokens`/`total_output_tokens` ÷ `completed`，不碰删除逻辑。**

## 文件结构（锁定分解）

| 文件 | 职责 | 动作 |
|---|---|---|
| `vllm_standalone_bench/tests/conftest.py` | 测试基础设施：把 bench 目录加入 sys.path；提供 `FakeSession`/`FakeResponse`/`sse()` SSE 夹具 | 新建 |
| `vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py` | 请求构造 + SSE 解析；`RequestFuncOutput` 加 `finish_reason`；completions `elif→if` | 修改 |
| `vllm_standalone_bench/vllm_bench/serve.py` | 流量引擎 + 指标：聚合 `finish_reason_length`/`usage_reported_count`；放宽指标门控 | 修改 |
| `vllm_standalone_bench/run_bench_multi.py` | 编排 + 落盘：`decide_token_usage_source` + `_extract_row` 真实 avg + 合规列；CSV/XLSX 表头；`--min-output-compliance` + 告警/跳过 | 修改 |
| `vllm_standalone_bench/requirements.txt` | 增加 pytest（开发依赖） | 修改 |
| `vllm_standalone_bench/tests/test_extract_row.py` | `_extract_row` 与 `decide_token_usage_source` 单测 | 新建 |
| `vllm_standalone_bench/tests/test_endpoint_parse.py` | SSE 解析（finish_reason、usage、elif 修复）单测 | 新建 |
| `vllm_standalone_bench/tests/test_serve_metrics.py` | `calculate_metrics` 聚合（finish_reason_length、usage_reported_count、无 tokenizer 也出 TPOT）单测 | 新建 |
| `vllm_standalone_bench/tests/test_integration.py` | 端到端冒烟：FakeSession 跑一组，断言 CSV avg 真实、合规≈100% | 新建 |

---

## 任务 1：测试基础设施（conftest + FakeSession 夹具）

**文件：**
- 创建：`vllm_standalone_bench/tests/conftest.py`
- 修改：`vllm_standalone_bench/requirements.txt`（加 pytest）

- [ ] **步骤 1：在 requirements.txt 末尾追加 pytest**

把 `vllm_standalone_bench/requirements.txt` 改为：

```
# 独立 vLLM Benchmark 工具依赖
# 无需安装 vllm 包，仅需以下轻量依赖：
aiohttp>=3.9.0
numpy>=1.24.0
tqdm>=4.65.0
transformers>=4.36.0

# 开发/测试
pytest>=8.0.0
```

- [ ] **步骤 2：创建 conftest.py（sys.path + FakeSession 夹具）**

创建 `vllm_standalone_bench/tests/conftest.py`：

```python
"""测试基础设施：确保 bench 目录可导入 + 提供 SSE/HTTP 夹具。"""
import json
import os
import sys

# 把 vllm_standalone_bench/ 加入 sys.path，使 run_bench_serve / run_bench_multi 可直接 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.dirname(_HERE)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)


class _FakeChunkStream:
    """模拟 aiohttp response.content，按 iter_any() 逐块吐出 SSE 字节。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for c in self._chunks:
            yield c


class FakeResponse:
    """模拟 aiohttp.ClientResponse 的最小子集（status/content/text/上下文）。"""

    def __init__(self, chunks, status=200, reason="OK", body=b""):
        self.status = status
        self.reason = reason
        self.content = _FakeChunkStream(chunks)
        self._body = body

    async def text(self):
        return self._body.decode("utf-8", "replace")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """模拟 aiohttp.ClientSession.post —— 总是返回同一份预置 SSE 响应。"""

    def __init__(self, chunks, status=200):
        self._chunks = chunks
        self._status = status

    def post(self, *args, **kwargs):
        return FakeResponse(self._chunks, status=self._status)

    async def close(self):
        pass


def sse(*events):
    """把若干 data 事件打包成原始 SSE 字节流。

    每个 event：dict → `data: {json}\n\n`；str → `data: {str}\n\n`（用于 "[DONE]"）。
    """
    out = b""
    for ev in events:
        payload = ev if isinstance(ev, str) else json.dumps(ev)
        out += f"data: {payload}\n\n".encode("utf-8")
    return out
```

- [ ] **步骤 3：验证夹具可导入**

运行：`cd vllm_standalone_bench && python -c "import tests.conftest; print('conftest OK', hasattr(tests.conftest, 'FakeSession'), hasattr(tests.conftest, 'sse'))"`
预期：`conftest OK True True`

- [ ] **步骤 4：验证 run_bench_serve shim 在测试进程内可用**

运行：`cd vllm_standalone_bench && python -c "import run_bench_serve; s=run_bench_serve._serve; print('serve OK', hasattr(s,'ASYNC_REQUEST_FUNCS'), hasattr(s,'calculate_metrics'))"`
预期：`serve OK True True`

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/tests/conftest.py vllm_standalone_bench/requirements.txt
git commit -m "test: 新增测试基础设施 conftest + FakeSession SSE 夹具"
```

---

## 任务 2：解析层补 finish_reason（completions + chat）

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py`
  - `RequestFuncOutput`（约 107-121 行）加字段
  - `async_request_openai_completions`（约 246-266 行）解析 finish_reason
  - `async_request_openai_chat_completions`（约 418-448 行）解析 finish_reason
- 测试：`vllm_standalone_bench/tests/test_endpoint_parse.py`

- [ ] **步骤 1：编写失败的测试**

创建 `vllm_standalone_bench/tests/test_endpoint_parse.py`：

```python
import pytest

# conftest 已把 bench 目录加入 sys.path 并提供 FakeSession/sse
from conftest import FakeSession, sse


def _load():
    """先触发 shim，再取解析函数与数据类。"""
    import run_bench_serve
    serve = run_bench_serve._serve
    return (
        serve.ASYNC_REQUEST_FUNCS["openai"],
        serve.ASYNC_REQUEST_FUNCS["openai-chat"],
        serve.RequestFuncInput,
    )


@pytest.mark.asyncio
async def _run(fn, RequestFuncInput, backend_path, chunks):
    inp = RequestFuncInput(
        prompt="hi", api_url=f"http://x{backend_path}",
        prompt_len=1, output_len=8, model="m",
    )
    return await fn(inp, FakeSession(chunks))


# 注：asyncio 标记依赖 pytest-asyncio；若未安装，改用 asyncio.run 包裹（见步骤3说明）。
```

> 说明：若环境未装 `pytest-asyncio`，**不要**引入新依赖。改用下面的同步包装写法（本计划统一采用此写法，零额外依赖）：

实际采用（替换上面 async 写法），创建测试文件用 `asyncio.run`：

```python
import asyncio
from conftest import FakeSession, sse


def _load():
    import run_bench_serve
    serve = run_bench_serve._serve
    return (
        serve.ASYNC_REQUEST_FUNCS["openai"],
        serve.ASYNC_REQUEST_FUNCS["openai-chat"],
        serve.RequestFuncInput,
    )


def _run(fn, RequestFuncInput, backend_path, chunks):
    inp = RequestFuncInput(
        prompt="hi", api_url=f"http://x{backend_path}",
        prompt_len=1, output_len=8, model="m",
    )
    return asyncio.run(fn(inp, FakeSession(chunks)))


def test_chat_finish_reason_and_usage():
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.finish_reason == "length"
    assert out.output_tokens == 8
    assert out.prompt_len == 5


def test_completions_finish_reason_and_usage():
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"text": "ab"}, "finish_reason": None]},
        {"choices": [{"text": ""}, "finish_reason": "length"]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(completions_fn, RequestFuncInput, "/v1/completions", chunks)
    assert out.success
    assert out.finish_reason == "length"
    assert out.output_tokens == 8
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_endpoint_parse.py -v`
预期：FAIL —— `AttributeError: 'RequestFuncOutput' object has no attribute 'finish_reason'`（或 `out.finish_reason == ''`）。

- [ ] **步骤 3：给 `RequestFuncOutput` 加 `finish_reason` 字段**

在 `endpoint_request_func.py` 的 `RequestFuncOutput`（约 107-121 行）末尾、`input_audio_duration` 之后追加一行：

```python
    input_audio_duration: float = 0.0  # in seconds
    finish_reason: str = ""  # 停止原因（"length"/"stop"/...），来自末帧 choices[0].finish_reason
```

- [ ] **步骤 4：completions 解析 finish_reason**

在 `async_request_openai_completions` 的 `if choices := data.get("choices"):` 块内（约 246-262 行），在 `generated_text += text or ""` 之后追加：

```python
                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                                if (fr := choices[0].get("finish_reason")) is not None:
                                    output.finish_reason = fr
```

（即紧跟现有 `generated_text += text or ""` 那一行后插入两行。）

- [ ] **步骤 5：chat 解析 finish_reason**

在 `async_request_openai_chat_completions` 的 `if choices := data.get("choices"):` 块内（约 418-439 行），在 `generated_text += token_text` 之后、`if usage := data.get("usage"):` 之前追加：

```python
                                generated_text += token_text
                                if (fr := choices[0].get("finish_reason")) is not None:
                                    output.finish_reason = fr
```

- [ ] **步骤 6：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_endpoint_parse.py -v`
预期：PASS（2 passed）。

- [ ] **步骤 7：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py vllm_standalone_bench/tests/test_endpoint_parse.py
git commit -m "feat: 解析层补 finish_reason（completions + chat 末帧）"
```

---

## 任务 3：修复 completions 的 `elif usage` → `if usage`（Bug ②）

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py`（`async_request_openai_completions` 约 263 行）
- 测试：`vllm_standalone_bench/tests/test_endpoint_parse.py`（追加用例）

- [ ] **步骤 1：追加失败的测试（choices 与 usage 同帧）**

在 `tests/test_endpoint_parse.py` 末尾追加：

```python
def test_completions_usage_in_same_chunk_as_choices():
    """回归 Bug ②：choices 与 usage 出现在同一帧时，completions 不能漏读 completion_tokens。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        # 服务端在最后一块同时带 choices（finish_reason）和 usage
        {"choices": [{"text": "ab", "finish_reason": "length"}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(completions_fn, RequestFuncInput, "/v1/completions", chunks)
    assert out.success
    assert out.output_tokens == 8, "completions 在 choices+usage 同帧时漏读了 usage"
    assert out.finish_reason == "length"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_endpoint_parse.py::test_completions_usage_in_same_chunk_as_choices -v`
预期：FAIL —— `AssertionError: ... output_tokens == 8`（实际为 0，因 `elif` 在 choices 存在时跳过了 usage）。

- [ ] **步骤 3：把 `elif usage` 改为 `if usage`**

在 `async_request_openai_completions`（约 263 行）将：

```python
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                                if (pt := usage.get("prompt_tokens")) is not None:
                                    output.prompt_len = pt
```

改为（`elif` → `if`）：

```python
                            if usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                                if (pt := usage.get("prompt_tokens")) is not None:
                                    output.prompt_len = pt
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_endpoint_parse.py -v`
预期：PASS（3 passed）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py vllm_standalone_bench/tests/test_endpoint_parse.py
git commit -m "fix: completions 解析 usage 由 elif 改为 if，防 choices+usage 同帧漏读"
```

---

## 任务 4：serve.py 聚合 finish_reason_length 与 usage_reported_count

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
  - `BenchmarkMetrics`（约 174-206 行）加两字段
  - `calculate_metrics`（约 407-616 行）计数并回填
  - `benchmark()` 结果 dict（约 1002-1022 行）写出两个字段
- 测试：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：编写失败的测试**

创建 `vllm_standalone_bench/tests/test_serve_metrics.py`：

```python
import run_bench_serve

serve = run_bench_serve._serve
RequestFuncOutput = serve.RequestFuncOutput
SampleRequest = serve.SampleRequest  # 由 datasets shim 注入，已挂在 serve 上


def _req(prompt_len, out_len):
    return SampleRequest(prompt="x", prompt_len=prompt_len,
                         expected_output_len=out_len)


def _out(success, *, output_tokens=0, finish_reason="", prompt_len=10,
         ttft=0.05, latency=1.0, itl=None):
    return RequestFuncOutput(
        success=success, output_tokens=output_tokens,
        finish_reason=finish_reason, prompt_len=prompt_len,
        ttft=ttft, latency=latency, itl=itl or [0.05, 0.05],
        generated_text="abc",
    )


def test_metrics_aggregate_finish_reason_and_usage():
    outputs = [
        _out(True, output_tokens=8, finish_reason="length"),   # usage 上报 + length 停止
        _out(True, output_tokens=8, finish_reason="length"),   # usage 上报 + length 停止
        _out(True, output_tokens=0, finish_reason="stop"),     # 未上报 usage（将回退 tokenizer/1）
    ]
    inputs = [_req(10, 8) for _ in outputs]
    metrics, _lens = serve.calculate_metrics(
        input_requests=inputs, outputs=outputs, dur_s=2.0,
        tokenizer=None, selected_percentiles=[50, 90],
        goodput_config_dict={},
    )
    assert metrics.completed == 3
    assert metrics.finish_reason_length == 2   # 两个 "length"
    assert metrics.usage_reported_count == 2   # 两个 output_tokens>0（来自 usage）


def test_result_dict_carries_new_fields():
    """benchmark() 结果 dict 应包含 finish_reason_length / usage_reported_count。"""
    # 直接校验字段存在于 calculate_metrics 返回的 metrics 上（result dict 由 benchmark() 拼装，
    # 字段挂载在任务4步骤5验证；此处先保证 metrics 字段就绪）
    outputs = [_out(True, output_tokens=8, finish_reason="length")]
    metrics, _ = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=None, selected_percentiles=[50], goodput_config_dict={},
    )
    assert hasattr(metrics, "finish_reason_length")
    assert hasattr(metrics, "usage_reported_count")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_serve_metrics.py -v`
预期：FAIL —— `AttributeError: 'BenchmarkMetrics' object has no attribute 'finish_reason_length'`。

- [ ] **步骤 3：给 `BenchmarkMetrics` 加两字段**

在 `serve.py` 的 `BenchmarkMetrics`（约 206 行，`rtfx` 那一行）后追加：

```python
    rtfx: float = 0.0  # Inverse Real-Time Factor for ASR benchmarks
    # —— 基准加固新增：停止原因与 usage 上报统计 ——
    finish_reason_length: int = 0      # finish_reason == "length" 的成功请求数
    usage_reported_count: int = 0      # 服务端流式上报了 output_tokens 的成功请求数
```

- [ ] **步骤 4：在 `calculate_metrics` 中计数并回填**

在 `calculate_metrics`（约 428-431 行）的计数器初始化区，`completed = 0` 附近追加两个计数器初始化：

```python
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    finish_reason_length = 0
    usage_reported_count = 0
    good_completed = 0
```

在成功分支（约 439-440 行），紧接 `output_len = outputs[i].output_tokens` 之后、`if not output_len:` 之前，插入两行：

```python
        if outputs[i].success:
            output_len = outputs[i].output_tokens
            if outputs[i].output_tokens:  # 来自服务端 usage 上报（非回退）
                usage_reported_count += 1
            if outputs[i].finish_reason == "length":
                finish_reason_length += 1
            if not output_len:
```

在 `BenchmarkMetrics(...)` 构造（约 577-614 行）的参数列表末尾、`rtfx=input_audio_duration / dur_s,` 之后追加：

```python
        rtfx=input_audio_duration / dur_s,
        finish_reason_length=finish_reason_length,
        usage_reported_count=usage_reported_count,
    )
```

- [ ] **步骤 5：在 `benchmark()` 结果 dict 写出新字段**

在 `serve.py` 结果 dict（约 1019-1021 行）的 `"max_concurrent_requests": ...` / `"rtfx": ...` 附近追加：

```python
            "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
            "max_concurrent_requests": metrics.max_concurrent_requests,
            "rtfx": metrics.rtfx,
            "finish_reason_length": metrics.finish_reason_length,
            "usage_reported_count": metrics.usage_reported_count,
        }
```

- [ ] **步骤 6：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_serve_metrics.py -v`
预期：PASS（2 passed）。

- [ ] **步骤 7：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "feat: serve.calculate_metrics 聚合 finish_reason_length 与 usage_reported_count 并写入结果"
```

---

## 任务 5：serve.py 放宽指标门控（无 tokenizer 也输出 TTFT/TPOT/ITL）

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`（约 1088 行）
- 测试：`vllm_standalone_bench/tests/test_serve_metrics.py`（追加用例）

- [ ] **步骤 1：追加失败的测试**

在 `tests/test_serve_metrics.py` 末尾追加：

```python
def test_metrics_reported_without_tokenizer_when_usage_present():
    """无 tokenizer 但服务端上报了 output_tokens 时，TPOT 等指标仍应可算出（非 0）。"""
    outputs = [_out(True, output_tokens=8, finish_reason="length",
                    ttft=0.05, latency=0.5, itl=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])]
    metrics, _ = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=None, selected_percentiles=[50], goodput_config_dict={},
    )
    # output_len=8 → tpots 非空 → mean_tpot_ms > 0
    assert metrics.mean_tpot_ms > 0
    assert metrics.mean_ttft_ms > 0
```

> 注：本任务改的是 `process_one_metric` 的**门控**（结果 dict 是否写入），而单测 `calculate_metrics` 已能算出 `mean_tpot_ms>0`。门控的端到端验证放在任务 8 的集成测试（无 tokenizer 跑通并落列）。此步骤的单测先锁定"指标可算出"这一前提。

- [ ] **步骤 2：运行测试验证（应已通过，作为前提锁定）**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_serve_metrics.py::test_metrics_reported_without_tokenizer_when_usage_present -v`
预期：PASS（`calculate_metrics` 不依赖 tokenizer 即可算出 TPOT）。若 FAIL，说明计数/字段有问题，先修任务 4。

- [ ] **步骤 3：放宽门控**

在 `serve.py`（约 1088 行）将：

```python
    if task_type == TaskType.GENERATION and tokenizer:
        process_one_metric("ttft", "TTFT", "Time to First Token")
        process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
        process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
```

改为（去掉 `and tokenizer`；TTFT/TPOT/ITL 不依赖 tokenizer，仅依赖 output 的 ttft/itl/latency 与 output_tokens）：

```python
    if task_type == TaskType.GENERATION:
        process_one_metric("ttft", "TTFT", "Time to First Token")
        process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
        process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
```

- [ ] **步骤 4：回归全部 serve 单测**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_serve_metrics.py -v`
预期：PASS（3 passed）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "fix: 放宽 serve 指标门控，无 tokenizer 时也输出 TTFT/TPOT/ITL"
```

---

## 任务 6：run_bench_multi —— `decide_token_usage_source` + `_extract_row` 真实 avg 与新列

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
  - 新增纯函数 `decide_token_usage_source`（放在 `_extract_row` 之前）
  - 重写 `_extract_row`（约 167-239 行）：真实 avg + 新列
  - 修改 `_extract_row` 调用处（约 477-481 行）：传 `has_tokenizer`
- 测试：`vllm_standalone_bench/tests/test_extract_row.py`

- [ ] **步骤 1：编写失败的测试**

创建 `vllm_standalone_bench/tests/test_extract_row.py`：

```python
import run_bench_multi as m


# ---------- decide_token_usage_source ----------
def test_token_source_usage_when_all_reported():
    assert m.decide_token_usage_source(
        usage_reported_count=3, completed=3, has_tokenizer=True) == "usage"


def test_token_source_tokenizer_when_none_reported_but_has_tok():
    assert m.decide_token_usage_source(
        usage_reported_count=0, completed=3, has_tokenizer=True) == "tokenizer"


def test_token_source_none_when_nothing():
    assert m.decide_token_usage_source(
        usage_reported_count=0, completed=3, has_tokenizer=False) == "none"


def test_token_source_none_when_all_failed():
    assert m.decide_token_usage_source(
        usage_reported_count=0, completed=0, has_tokenizer=True) == "none"


# ---------- _extract_row: 真实 avg（不再回显 requested） ----------
def _result(total_in=30, total_out=24, completed=3, usage_reported=3,
            finish_reason_length=3):
    """构造 serve.main_async 风格的最小 result dict（仅本测试关心的键）。"""
    return {
        "completed": completed, "failed": 0,
        "total_input_tokens": total_in, "total_output_tokens": total_out,
        "usage_reported_count": usage_reported,
        "finish_reason_length": finish_reason_length,
        "num_prompts": completed,
        "request_throughput": 1.0, "output_throughput": 12.0,
        "duration": 2.0,
    }


def test_extract_row_real_avg_from_totals():
    row = m._extract_row(
        _result(total_in=30, total_out=24, completed=3),  # 请求 out_len=8
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    # 真实平均 = 30/3=10（输入）, 24/3=8（输出）—— 而非回显 requested
    assert row["avg_input_tokens"] == 10.0
    assert row["avg_output_tokens"] == 8.0
    assert row["token_source"] == "usage"
    assert row["output_compliance"] == 100.0  # 8/8
    assert row["finish_reason_length_pct"] == 100.0


def test_extract_row_compliance_when_undergenerated():
    # 服务端只生成了 12 token（请求 8×3=24，实测 12/3=4 < 8）
    row = m._extract_row(
        _result(total_out=12, completed=3, usage_reported=3),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["avg_output_tokens"] == 4.0
    assert row["output_compliance"] == 50.0  # 4/8
    assert row["token_source"] == "usage"


def test_extract_row_prefix_total_input_len():
    row = m._extract_row(
        _result(completed=3, total_in=690, total_out=24),  # prefix 场景
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai", prefix_tokens=102,
        prefix_ratio=0.8, has_tokenizer=True)
    assert row["total_input_len"] == 128 + 102
    assert row["input_len"] == 128  # requested 后缀长度


def test_extract_row_token_source_tokenizer_when_no_usage():
    row = m._extract_row(
        _result(total_out=24, completed=3, usage_reported=0),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["token_source"] == "tokenizer"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_extract_row.py -v`
预期：FAIL —— `AttributeError: module 'run_bench_multi' has no attribute 'decide_token_usage_source'`，且 `avg_output_tokens` 仍等于 requested（8 而非取决于 totals；本测试喂的 totals 使真实=8，但 `token_source`/`output_compliance`/`total_input_len` 键缺失会 KeyError）。

- [ ] **步骤 3：新增 `decide_token_usage_source` 纯函数**

在 `run_bench_multi.py` 的 `_extract_row` 定义（约 167 行）**之前**插入：

```python
def decide_token_usage_source(*, usage_reported_count: int, completed: int,
                              has_tokenizer: bool) -> str:
    """决定每行结果的 token 计数来源（写入 CSV 的 token_source 列）。

    - "usage"：所有成功请求的服务端流式 usage 都上报了 completion_tokens（最可信）
    - "tokenizer"：服务端未上报，但有本地 tokenizer 可重编码 generated_text 估算
    - "none"：两者皆无（统计不可信，应告警）
    """
    if completed <= 0:
        return "none"
    if usage_reported_count >= completed:
        return "usage"
    if has_tokenizer:
        return "tokenizer"
    return "none"
```

- [ ] **步骤 4：重写 `_extract_row` 的平均与新列部分**

在 `run_bench_multi.py` 中，把 `_extract_row`（约 167-239 行）整体替换为：

```python
def _extract_row(
    result: dict,
    in_len: int,
    out_len: int,
    parallel_num: int,
    epochs: int,
    model: str,
    backend: str,
    prefix_tokens: int = 0,
    prefix_ratio: float = 0.0,
    has_tokenizer: bool = False,
) -> dict:
    """从 _serve.main_async() 返回的字典中提取并重命名需要的指标字段。

    重要：avg_input_tokens / avg_output_tokens 取自【存活】的总量字段
    total_input_tokens / total_output_tokens ÷ completed，【不再】回退到 requested 值。
    （serve.py 在 save_detailed=False 时会删除每请求 input_lens/output_lens，
     故不能依赖它们；总量字段不受删除影响。）
    """
    def _f(key: str, default=0.0) -> float:
        v = result.get(key, default)
        return round(float(v), 4) if v is not None else 0.0

    def _i(key: str, default=0) -> int:
        return int(result.get(key, default) or default)

    completed = _i('completed')
    total_in = _i('total_input_tokens')
    total_out = _i('total_output_tokens')

    # ── 真实平均 token 数（来自服务端上报的总量）─────────────────────────────
    avg_in = round(total_in / completed, 1) if completed > 0 else in_len
    avg_out = round(total_out / completed, 1) if completed > 0 else out_len

    # ── token 计数来源 ────────────────────────────────────────────────────────
    token_source = decide_token_usage_source(
        usage_reported_count=_i('usage_reported_count'),
        completed=completed,
        has_tokenizer=has_tokenizer,
    )

    # ── 长度合规：实测输出 / 请求输出 ──────────────────────────────────────────
    output_compliance = round(avg_out / out_len * 100, 1) if out_len > 0 else 0.0
    finish_reason_length_pct = (
        round(_i('finish_reason_length') / completed * 100, 1)
        if completed > 0 else 0.0
    )

    return {
        # ── 测试配置 ────────────────────────────────
        'model':           model,
        'backend':         backend,
        'input_len':       in_len,            # requested 后缀长度
        'output_len':      out_len,           # requested 输出长度
        'total_input_len': in_len + prefix_tokens,  # 含共享前缀的总输入（requested 口径）
        'prefix_ratio':    round(prefix_ratio, 3),
        'prefix_tokens':   prefix_tokens,
        'parallel_num':    parallel_num,
        'epochs':          epochs,
        'num_prompts':     _i('num_prompts', parallel_num * epochs),
        # ── 请求统计 ────────────────────────────────
        'n_success':           completed,
        'n_failed':            _i('failed'),
        'avg_input_tokens':    avg_in,    # 真实（曾因 Bug①回显 requested，现已修正）
        'avg_output_tokens':   avg_out,   # 真实
        'output_compliance':   output_compliance,
        'finish_reason_length_pct': finish_reason_length_pct,
        'token_source':        token_source,
        # ── 吞吐量 ──────────────────────────────────
        'throughput_req_s':   _f('request_throughput'),
        'throughput_tok_s':   _f('output_throughput'),
        # ── TTFT (ms) ───────────────────────────────
        'ttft_mean_ms':  _f('mean_ttft_ms'),
        'ttft_p50_ms':   _f('p50_ttft_ms'),
        'ttft_p90_ms':   _f('p90_ttft_ms'),
        'ttft_p99_ms':   _f('p99_ttft_ms'),
        # ── TPOT (ms) ───────────────────────────────
        'tpot_mean_ms':  _f('mean_tpot_ms'),
        'tpot_p50_ms':   _f('p50_tpot_ms'),
        'tpot_p90_ms':   _f('p90_tpot_ms'),
        'tpot_p99_ms':   _f('p99_tpot_ms'),
        # ── E2EL (ms) ───────────────────────────────
        'e2el_mean_ms':  _f('mean_e2el_ms'),
        'e2el_p50_ms':   _f('p50_e2el_ms'),
        'e2el_p90_ms':   _f('p90_e2el_ms'),
        'e2el_p99_ms':   _f('p99_e2el_ms'),
        # ── 其他 ────────────────────────────────────
        'duration_s':    _f('duration'),
    }
```

- [ ] **步骤 5：在调用处传入 `has_tokenizer`**

在 `run_bench_multi.py::_run_all`（约 477-481 行），把：

```python
            row = _extract_row(result, in_len, out_len, parallel_num,
                               our_args.epochs, model, our_args.backend,
                               prefix_tokens=prefix_tokens,
                               prefix_ratio=prefix_ratio)
```

改为：

```python
            row = _extract_row(result, in_len, out_len, parallel_num,
                               our_args.epochs, model, our_args.backend,
                               prefix_tokens=prefix_tokens,
                               prefix_ratio=prefix_ratio,
                               has_tokenizer=bool(our_args.tokenizer))
```

- [ ] **步骤 6：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_extract_row.py -v`
预期：PASS（8 passed）。

- [ ] **步骤 7：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py
git commit -m "fix: _extract_row 用总量字段算真实 avg，新增长度合规/token来源/总输入列"
```

---

## 任务 7：CSV/XLSX 表头扩展 + `--min-output-compliance` + 合规告警/跳过

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
  - `CSV_HEADERS` / `CSV_HEADERS_ZH`（约 244-268 行）加列
  - `_parse_args`（约束过滤组，约 611-620 行）加 `--min-output-compliance`
  - `_run_all`（约 507-516 行）加合规告警与跳过更高并发
- 测试：`vllm_standalone_bench/tests/test_extract_row.py`（追加表头一致性用例）

- [ ] **步骤 1：追加失败的测试（表头与新行键一致）**

在 `tests/test_extract_row.py` 末尾追加：

```python
def test_csv_headers_match_row_keys():
    row = m._extract_row(
        {"completed": 1, "total_input_tokens": 5, "total_output_tokens": 8,
         "usage_reported_count": 1, "finish_reason_length": 1, "num_prompts": 1,
         "request_throughput": 1.0, "output_throughput": 8.0, "duration": 1.0},
        in_len=5, out_len=8, parallel_num=1, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    missing = [h for h in m.CSV_HEADERS if h not in row]
    assert not missing, f"CSV_HEADERS 有列在 row 中缺失: {missing}"
```

- [ ] **步骤 2：运行测试验证（应通过，作为键一致性锁定）**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_extract_row.py::test_csv_headers_match_row_keys -v`
预期：PASS（任务 6 已让 row 含全部键；若有缺失则 FAIL，需补 `CSV_HEADERS` 或 row 键）。

- [ ] **步骤 3：扩展 `CSV_HEADERS` 与 `CSV_HEADERS_ZH`**

把 `run_bench_multi.py::CSV_HEADERS`（约 244-255 行）替换为：

```python
CSV_HEADERS = [
    'model', 'backend',
    'input_len', 'output_len', 'total_input_len', 'prefix_ratio', 'prefix_tokens',
    'parallel_num', 'epochs', 'num_prompts',
    'n_success', 'n_failed',
    'avg_input_tokens', 'avg_output_tokens',
    'output_compliance', 'finish_reason_length_pct', 'token_source',
    'throughput_req_s', 'throughput_tok_s',
    'ttft_mean_ms', 'ttft_p50_ms', 'ttft_p90_ms', 'ttft_p99_ms',
    'tpot_mean_ms', 'tpot_p50_ms', 'tpot_p90_ms', 'tpot_p99_ms',
    'e2el_mean_ms', 'e2el_p50_ms', 'e2el_p90_ms', 'e2el_p99_ms',
    'duration_s',
]
```

把 `CSV_HEADERS_ZH`（约 257-268 行）替换为（与上面对齐）：

```python
CSV_HEADERS_ZH = [
    '模型', '接口类型',
    '输入长度(token)', '输出长度(token)', '总输入长度(token)', '前缀比例', '前缀tokens数',
    '并发数', '测试轮数', '总请求数',
    '成功请求数', '失败请求数',
    '平均实际输入tokens', '平均实际输出tokens',
    '长度合规(%)', 'length停止占比(%)', 'token来源',
    '请求吞吐(req/s)', '输出Token吞吐(tok/s)',
    'TTFT均值(ms)', 'TTFT_P50(ms)', 'TTFT_P90(ms)', 'TTFT_P99(ms)',
    'TPOT均值(ms)', 'TPOT_P50(ms)', 'TPOT_P90(ms)', 'TPOT_P99(ms)',
    'E2EL均值(ms)', 'E2EL_P50(ms)', 'E2EL_P90(ms)', 'E2EL_P99(ms)',
    '测试耗时(s)',
]
```

> `save_xlsx` 内部按 `CSV_HEADERS` 遍历写列（`run_bench_multi.py:303-325`），无需改动即自动适配新列；ZH 表头同步增长即可。`save_csv` 用 `DictWriter(fieldnames=CSV_HEADERS, extrasaction='ignore')`，同样自动适配。

- [ ] **步骤 4：加 `--min-output-compliance` 参数**

在 `run_bench_multi.py::_parse_args` 的"约束过滤"组（约 611-620 行，`--min-throughput-tok-s` 之后）追加：

```python
    limit.add_argument('--min-output-compliance', type=float, default=0.95,
                       help='最低输出长度合规比例（0~1，avg_output_measured/requested）。'
                            '低于此值告警并跳过该 (input,output) 组合的更高并发测试。'
                            'ignore_eos 生效时合规应≈1.0；偏低通常意味服务端提前停止'
                            '（如 ignore_eos 未生效/被截断）。默认 0.95。')
```

- [ ] **步骤 5：在 `_run_all` 加合规告警与跳过**

在 `run_bench_multi.py::_run_all` 的现有跳过检查之后（约 507-516 行，`--min-throughput-tok-s` 跳过块之后、`# 配置间隔等待` 之前）追加：

```python
            # 约束检查：输出长度合规偏低（服务端未按指定长度输出）则跳过更高并发
            compliance_frac = row['output_compliance'] / 100.0
            if (not skip_higher_parallel
                    and row['n_success'] > 0
                    and our_args.min_output_compliance is not None
                    and compliance_frac < our_args.min_output_compliance):
                logger.warning(
                    "  output_compliance (%.1f%%) < 约束下限 (%.1f%%)，"
                    "token_source=%s —— 服务端可能未按指定长度输出（ignore_eos 未生效/被截断），"
                    "跳过 input=%d output=%d 的更高并发测试",
                    row['output_compliance'], our_args.min_output_compliance * 100,
                    row['token_source'], in_len, out_len,
                )
                skip_higher_parallel = True
```

- [ ] **步骤 6：运行全部单测**

运行：`cd vllm_standalone_bench && python -m pytest tests/ -v`
预期：PASS（全部用例）。

- [ ] **步骤 7：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py
git commit -m "feat: CSV/XLSX 扩展合规列，新增 --min-output-compliance 告警与跳过"
```

---

## 任务 8：端到端集成冒烟测试（FakeSession 跑通 → 断言 CSV 真实）

**文件：**
- 创建：`vllm_standalone_bench/tests/test_integration.py`

> 本任务不连真实服务，而是把 `serve.py` 的请求函数通过 monkeypatch 替换为返回固定 `RequestFuncOutput`（含真实 output_tokens/finish_reason）的假函数，再走 `run_bench_multi._run_all` 的提取与落盘路径，断言 CSV 的 avg 与合规列真实。

- [ ] **步骤 1：编写集成测试**

创建 `vllm_standalone_bench/tests/test_integration.py`：

```python
import asyncio
import csv
import os
import tempfile

import run_bench_multi as m
import run_bench_serve

serve = run_bench_serve._serve
RequestFuncOutput = serve.RequestFuncOutput


def _fake_result(in_len, out_len, completed, *, undergen=False):
    """构造一个 serve.main_async 风格的 result dict。undergen=True 表示服务端少生成。"""
    real_out = (out_len // 2) if undergen else out_len
    return {
        "duration": 2.0,
        "completed": completed, "failed": 0,
        "total_input_tokens": in_len * completed,
        "total_output_tokens": real_out * completed,
        "request_throughput": completed / 2.0,
        "output_throughput": real_out * completed / 2.0,
        "total_token_throughput": (in_len + real_out) * completed / 2.0,
        "usage_reported_count": completed,
        "finish_reason_length": completed if not undergen else 0,
        "num_prompts": completed,
        # TTFT/TPOT/E2EL 等字段（_extract_row 经 _f/_i 读，缺失则 0）
        "mean_ttft_ms": 50.0, "p50_ttft_ms": 50.0, "p90_ttft_ms": 60.0, "p99_ttft_ms": 70.0,
        "mean_tpot_ms": 30.0, "p50_tpot_ms": 30.0, "p90_tpot_ms": 31.0, "p99_tpot_ms": 32.0,
        "mean_e2el_ms": 1000.0, "p50_e2el_ms": 1000.0, "p90_e2el_ms": 1100.0, "p99_e2el_ms": 1200.0,
    }


def _run_to_rows(monkeypatch, results_seq):
    """把 _serve.main_async 替换为依次返回 results_seq 的假实现，跑 _run_all。"""
    it = iter(results_seq)

    async def _fake_main_async(cfg):
        return next(it)

    monkeypatch.setattr(serve, "main_async", _fake_main_async)

    our_args = m.argparse.Namespace(
        model="m", served_model_name=None, backend="openai-chat",
        base_url=None, host="127.0.0.1", port=8000, insecure=False, api_key=None,
        tokenizer="/some/tok",  # has_tokenizer=True
        input_lens=[128], output_lens=[8], cross_product=False,
        parallel_nums=[1], epochs=1, sleep_between=0, warmup_requests=0,
        prefix_ratio=0.0,
        max_ttft_ms=None, min_throughput_tok_s=None, min_output_compliance=0.95,
        output_csv=None, output_xlsx=None, result_dir=None,
    )
    return m._run_all(our_args)


def test_csv_records_real_avg_and_compliance(tmp_path, monkeypatch):
    csv_path = str(tmp_path / "bench.csv")
    # 给 _run_all 一个 output_csv，使其每步落盘
    rows = _run_to_rows(monkeypatch, [_fake_result(128, 8, 3)])
    assert rows[0]["avg_output_tokens"] == 8.0          # 真实，非 requested 回显
    assert rows[0]["output_compliance"] == 100.0
    assert rows[0]["token_source"] == "usage"

    # 落盘校验
    import run_bench_multi as mm
    mm.save_csv(rows, csv_path)
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    assert len(data) == 1
    assert float(data[0]["avg_output_tokens"]) == 8.0
    assert float(data[0]["output_compliance"]) == 100.0
    assert data[0]["token_source"] == "usage"
    assert int(data[0]["total_input_len"]) == 128


def test_csv_flags_undergeneration(tmp_path, monkeypatch):
    rows = _run_to_rows(monkeypatch, [_fake_result(128, 8, 3, undergen=True)])
    assert rows[0]["avg_output_tokens"] == 4.0          # 服务端只生成一半
    assert rows[0]["output_compliance"] == 50.0         # 4/8
    assert rows[0]["finish_reason_length_pct"] == 0.0   # 都不是 length 停止
```

- [ ] **步骤 2：运行集成测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_integration.py -v`
预期：PASS（2 passed）。

> 若 `_run_all` 因 Namespace 缺字段报错，按报错补 `our_args` 缺失的属性（保持最小集合即可，因为 `_build_base_args` 会从 `_serve.add_cli_args` 解析空参拿默认值，再覆盖）。

- [ ] **步骤 3：运行全部测试（回归）**

运行：`cd vllm_standalone_bench && python -m pytest tests/ -v`
预期：PASS（全部用例）。

- [ ] **步骤 4：手动冒烟（可选，需本地服务）**

若本机有 vLLM 或 SGLang 服务，跑一次真实压测确认 CSV 新列正常：

```bash
cd vllm_standalone_bench
# 编辑 run_bench.sh 的 HOST/PORT/MODEL 后：
./run_bench.sh
# 检查 results/bench_*.csv：avg_output_tokens 应≈output_len，output_compliance≈100，token_source=usage
```

预期：CSV `avg_output_tokens` ≈ `output_len`、`output_compliance` ≈ 100、`token_source` = `usage`。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/tests/test_integration.py
git commit -m "test: 新增端到端集成冒烟，断言 CSV 真实 avg 与合规列"
```

---

## 完成定义（Definition of Done）

1. `cd vllm_standalone_bench && python -m pytest tests/ -v` 全绿。
2. `avg_input_tokens`/`avg_output_tokens` 取自服务端上报的总量（与 requested 解耦）；打 vLLM 与 SGLang 均正确。
3. `output_compliance` 在 ignore_eos 生效时 ≈ 100%；人为去掉 ignore_eos 时 < 100% 并触发告警/跳过。
4. 无 `--tokenizer` 时 TTFT/TPOT/ITL 不再整列 0（服务端上报 usage 即可）。
5. `token_source` 正确反映 usage / tokenizer / none，绝不静默置 1。
6. 新列出现在 CSV 与 XLSX（`total_input_len`、`output_compliance`、`finish_reason_length_pct`、`token_source`）。

## 实现者备忘

- `serve.py` / `endpoint_request_func.py` 是 vendored 副本；修改是定点的，重新从上游拷贝需重套。
- 跨框架靠统一字段，**不要**加 `--framework` 分叉。vLLM completions 默认 `max_tokens=16`、SGLang chat 默认 128——本工具总发 `output_len` 已规避；合规列会在配置错误时告警。
- 若实现中发现 `_run_all` 的 Namespace 集合（任务 8）与真实 `_parse_args` 漂移，以 `_parse_args` 为准补齐测试夹具，不要改 `_run_all` 签名。
