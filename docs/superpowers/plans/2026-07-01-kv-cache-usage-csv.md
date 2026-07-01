# KV Cache Usage CSV 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在每轮 vLLM benchmark 期间采样 GPU KV cache usage，并把平均值和峰值写入现有 CSV/XLSX，同时兼容 SpecDecode accepted/draft token 的两组字段名。

**架构：** `vllm_bench/serve.py` 继续作为单轮 benchmark 执行核心，复用现有 `/metrics` URL 归一和鉴权 header 透传，在请求运行期间后台轮询 Prometheus 文本。`run_bench_multi.py` 继续只负责把 `serve.py` 返回的 result dict 映射成 CSV/XLSX 行，并在这里处理旧字段和新字段的兼容。

**技术栈：** Python 3、asyncio、aiohttp、pytest、现有 run_bench_serve shim、Prometheus text exposition 解析。

---

## 文件职责

- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
  - 新增 runtime metrics 解析函数，能从 Prometheus 文本中读取 GPU KV cache usage gauge。
  - 新增 benchmark 生命周期内的轻量后台采样器。
  - 保持 `fetch_spec_decode_metrics()` 现有调用兼容，同时让它复用新的 metrics 解析。
  - 在 `benchmark()` 返回的 result dict 中写入 `avg_gpu_kv_cache_usage` 和 `peak_gpu_kv_cache_usage`。

- 修改：`vllm_standalone_bench/run_bench_multi.py`
  - `_extract_row()` 新增两列 GPU KV cache usage。
  - `_extract_row()` 兼容 `spec_decode_num_accepted_tokens`、`spec_decode_accepted_tokens`、`spec_decode_num_draft_tokens`、`spec_decode_draft_tokens`。
  - `CSV_HEADERS`、`CSV_HEADERS_ZH`、XLSX “指标说明”页同步新增说明。

- 修改：`vllm_standalone_bench/tests/test_serve_metrics.py`
  - 覆盖 metrics 文本解析、鉴权 header 透传、采样器 avg/peak 聚合和失败降级。

- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
  - 覆盖 CSV 行字段、默认值、表头、中文表头和 SpecDecode 字段别名。

---

### 任务 0：确认基线

**文件：**
- 读取：`vllm_standalone_bench/tests`

- [ ] **步骤 1：运行现有测试基线**

运行：

```bash
pytest -q vllm_standalone_bench/tests
```

预期：测试通过，输出包含 `284 passed`。

---

### 任务 1：新增 runtime metrics 解析和采样器测试

**文件：**
- 修改：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：为 GPU KV cache usage 文本解析编写失败测试**

在 `vllm_standalone_bench/tests/test_serve_metrics.py` 追加以下测试：

```python
def test_parse_runtime_metrics_reads_gpu_kv_cache_usage_percent():
    metrics = serve.parse_runtime_metrics_text(
        "\n".join(
            [
                "# HELP vllm:gpu_cache_usage_perc GPU cache usage",
                'vllm:gpu_cache_usage_perc{engine="0"} 9.8',
            ]
        )
    )

    assert metrics.gpu_kv_cache_usage == 9.8
    assert metrics.spec_decode is None
```

- [ ] **步骤 2：为多 worker 单次 scrape 取最大值编写失败测试**

在同一个测试文件追加：

```python
def test_parse_runtime_metrics_uses_max_gpu_kv_cache_usage_per_scrape():
    metrics = serve.parse_runtime_metrics_text(
        "\n".join(
            [
                'vllm:gpu_cache_usage_perc{engine="0"} 7.2',
                'vllm:gpu_cache_usage_perc{engine="1"} 12.5',
                'vllm:gpu_cache_usage_perc{engine="2"} 4.1',
            ]
        )
    )

    assert metrics.gpu_kv_cache_usage == 12.5
```

- [ ] **步骤 3：为比例值归一化编写失败测试**

在同一个测试文件追加：

```python
def test_parse_runtime_metrics_normalizes_fraction_gpu_kv_cache_usage():
    metrics = serve.parse_runtime_metrics_text(
        "\n".join(
            [
                "vllm:kv_cache_usage_perc 0.098",
                "ray_vllm_kv_cache_usage_perc 0.052",
            ]
        )
    )

    assert metrics.gpu_kv_cache_usage == 9.8
```

- [ ] **步骤 4：为 SpecDecode 兼容解析编写失败测试**

在同一个测试文件追加：

```python
def test_parse_runtime_metrics_keeps_spec_decode_metrics():
    metrics = serve.parse_runtime_metrics_text(
        "\n".join(
            [
                "vllm:spec_decode_num_drafts_total 4",
                "vllm:spec_decode_num_accepted_tokens_total 3",
                "vllm:spec_decode_num_draft_tokens_total 8",
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 3',
            ]
        )
    )

    assert metrics.spec_decode is not None
    assert metrics.spec_decode.num_drafts == 4
    assert metrics.spec_decode.num_accepted_tokens == 3
    assert metrics.spec_decode.num_draft_tokens == 8
    assert metrics.spec_decode.accepted_per_pos == {0: 3}
```

- [ ] **步骤 5：为采样器聚合编写失败测试**

在同一个测试文件追加：

```python
def test_runtime_metrics_summary_aggregates_avg_and_peak():
    summary = serve.RuntimeMetricsSummary.from_samples([7.0, 11.0, 9.0])

    assert summary.avg_gpu_kv_cache_usage == 9.0
    assert summary.peak_gpu_kv_cache_usage == 11.0
```

- [ ] **步骤 6：为采样器空样本默认值编写失败测试**

在同一个测试文件追加：

```python
def test_runtime_metrics_summary_defaults_zero_without_samples():
    summary = serve.RuntimeMetricsSummary.from_samples([])

    assert summary.avg_gpu_kv_cache_usage == 0.0
    assert summary.peak_gpu_kv_cache_usage == 0.0
```

- [ ] **步骤 7：为 `/metrics` header 透传和 runtime fetch 编写失败测试**

在同一个测试文件追加：

```python
def test_fetch_runtime_metrics_passes_headers_and_normalizes_url():
    class FakeResponse:
        status = 200

        async def text(self):
            return "vllm:gpu_cache_usage_perc 9.8"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, headers=None):
            self.calls.append((url, headers))
            return FakeResponse()

    session = FakeSession()
    headers = {"Authorization": "Bearer local-bench-key"}

    metrics = asyncio.run(
        serve.fetch_runtime_metrics("http://127.0.0.1:8000/v1", session, headers)
    )

    assert session.calls == [
        ("http://127.0.0.1:8000/metrics", headers),
    ]
    assert metrics is not None
    assert metrics.gpu_kv_cache_usage == 9.8
```

- [ ] **步骤 8：运行新增测试并确认失败**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_serve_metrics.py -q
```

预期：失败，报错包含 `AttributeError`，指出 `parse_runtime_metrics_text`、`RuntimeMetricsSummary` 或 `fetch_runtime_metrics` 尚不存在。

---

### 任务 2：实现 runtime metrics 解析、fetch wrapper 和 summary

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
- 测试：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：新增 runtime metrics 数据结构**

在 `SpecDecodeMetrics` 后面加入：

```python
@dataclass
class RuntimeMetrics:
    """Runtime metrics parsed from the server's Prometheus endpoint."""

    spec_decode: SpecDecodeMetrics | None = None
    gpu_kv_cache_usage: float | None = None


@dataclass
class RuntimeMetricsSummary:
    """Aggregated runtime metrics for a single benchmark run."""

    avg_gpu_kv_cache_usage: float = 0.0
    peak_gpu_kv_cache_usage: float = 0.0

    @classmethod
    def from_samples(cls, samples: Iterable[float]) -> "RuntimeMetricsSummary":
        values = [float(sample) for sample in samples]
        if not values:
            return cls()
        return cls(
            avg_gpu_kv_cache_usage=round(sum(values) / len(values), 4),
            peak_gpu_kv_cache_usage=round(max(values), 4),
        )
```

- [ ] **步骤 2：新增 Prometheus 行解析 helper**

在 `_metrics_url_from_base()` 后面加入：

```python
GPU_KV_CACHE_USAGE_METRICS = {
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "ray_vllm_kv_cache_usage_perc",
}


def _metric_name_from_prometheus_line(line: str) -> str:
    return line.split(None, 1)[0].split("{", 1)[0]


def _metric_value_from_prometheus_line(line: str) -> float | None:
    parts = line.split()
    if not parts:
        return None
    with contextlib.suppress(ValueError, OverflowError):
        value = float(parts[-1])
        if np.isfinite(value):
            return value
    return None


def _normalize_gpu_kv_cache_usage(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value
```

- [ ] **步骤 3：实现统一文本解析函数**

用 `parse_runtime_metrics_text()` 承载 SpecDecode 和 GPU KV cache usage 解析：

```python
def parse_runtime_metrics_text(text: str) -> RuntimeMetrics:
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_per_pos: dict[int, int] = {}
    found_spec_decode = False
    gpu_kv_cache_usage_samples: list[float] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        metric_name = _metric_name_from_prometheus_line(line)
        value = _metric_value_from_prometheus_line(line)
        if value is None:
            continue

        if metric_name in GPU_KV_CACHE_USAGE_METRICS:
            gpu_kv_cache_usage_samples.append(_normalize_gpu_kv_cache_usage(value))
            continue

        if not metric_name.startswith("vllm:spec_decode"):
            continue
        if not metric_name.endswith("_total"):
            continue

        found_spec_decode = True
        if "num_drafts" in metric_name:
            num_drafts += int(value)
        elif "num_draft_tokens" in metric_name:
            num_draft_tokens += int(value)
        elif "num_accepted_tokens_per_pos" in metric_name:
            pos_label = 'position="'
            if pos_label in line:
                with contextlib.suppress(ValueError):
                    start = line.index(pos_label) + len(pos_label)
                    end = line.index('"', start)
                    pos = int(line[start:end])
                    accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + int(value)
        elif "num_accepted_tokens" in metric_name:
            num_accepted_tokens += int(value)

    spec_decode = None
    if found_spec_decode:
        spec_decode = SpecDecodeMetrics(
            num_drafts=num_drafts,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            accepted_per_pos=accepted_per_pos,
        )

    gpu_kv_cache_usage = (
        max(gpu_kv_cache_usage_samples) if gpu_kv_cache_usage_samples else None
    )
    return RuntimeMetrics(
        spec_decode=spec_decode,
        gpu_kv_cache_usage=gpu_kv_cache_usage,
    )
```

- [ ] **步骤 4：新增 runtime fetch 并保留 SpecDecode wrapper**

把 `fetch_spec_decode_metrics()` 改成委托 `fetch_runtime_metrics()`：

```python
async def fetch_runtime_metrics(
    base_url: str,
    session: aiohttp.ClientSession,
    extra_headers: dict[str, str] | None = None,
) -> RuntimeMetrics | None:
    """Fetch runtime metrics from the server's Prometheus endpoint."""
    metrics_url = _metrics_url_from_base(base_url)
    try:
        async with session.get(metrics_url, headers=extra_headers) as response:
            if response.status != 200:
                return None
            return parse_runtime_metrics_text(await response.text())
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def fetch_spec_decode_metrics(
    base_url: str,
    session: aiohttp.ClientSession,
    extra_headers: dict[str, str] | None = None,
) -> SpecDecodeMetrics | None:
    """Fetch speculative decoding metrics from the server's Prometheus endpoint.

    Returns None if speculative decoding is not enabled or metrics are not available.
    """
    metrics = await fetch_runtime_metrics(base_url, session, extra_headers)
    if metrics is None:
        return None
    return metrics.spec_decode
```

- [ ] **步骤 5：运行 serve metrics 测试**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_serve_metrics.py
```

预期：通过。

- [ ] **步骤 6：提交 parser 和 fetch 变更**

运行：

```bash
git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "feat: parse gpu kv cache runtime metrics"
```

---

### 任务 3：把 runtime sampler 接入 benchmark 结果

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
- 修改：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：为 sampler start/stop 编写失败测试**

在 `test_serve_metrics.py` 追加：

```python
def test_runtime_metrics_sampler_scrapes_start_and_stop():
    class FakeResponse:
        status = 200

        def __init__(self, text):
            self._text = text

        async def text(self):
            return self._text

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self):
            self.calls = []
            self.responses = [
                "vllm:gpu_cache_usage_perc 8.0",
                "vllm:gpu_cache_usage_perc 12.0",
            ]

        def get(self, url, headers=None):
            self.calls.append((url, headers))
            return FakeResponse(self.responses.pop(0))

    async def run_sampler():
        session = FakeSession()
        sampler = serve.RuntimeMetricsSampler(
            base_url="http://127.0.0.1:8000/v1",
            session=session,
            extra_headers={"Authorization": "Bearer local-bench-key"},
            interval_s=60.0,
        )
        await sampler.start()
        summary = await sampler.stop()
        return session.calls, summary

    calls, summary = asyncio.run(run_sampler())

    assert calls == [
        (
            "http://127.0.0.1:8000/metrics",
            {"Authorization": "Bearer local-bench-key"},
        ),
        (
            "http://127.0.0.1:8000/metrics",
            {"Authorization": "Bearer local-bench-key"},
        ),
    ]
    assert summary.avg_gpu_kv_cache_usage == 10.0
    assert summary.peak_gpu_kv_cache_usage == 12.0
```

- [ ] **步骤 2：为 result dict 字段注入编写失败测试**

在 `test_serve_metrics.py` 追加：

```python
def test_add_runtime_metrics_to_result_sets_gpu_kv_fields():
    result = {"completed": 1}
    summary = serve.RuntimeMetricsSummary(
        avg_gpu_kv_cache_usage=9.5,
        peak_gpu_kv_cache_usage=13.0,
    )

    serve.add_runtime_metrics_to_result(result, summary)

    assert result["avg_gpu_kv_cache_usage"] == 9.5
    assert result["peak_gpu_kv_cache_usage"] == 13.0
```

- [ ] **步骤 3：运行新增测试并确认失败**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_serve_metrics.py -q
```

预期：失败，报错包含 `AttributeError`，指出 `RuntimeMetricsSampler` 或 `add_runtime_metrics_to_result` 尚不存在。

- [ ] **步骤 4：实现 RuntimeMetricsSampler**

在 `serve.py` 的 runtime metrics helper 后加入：

```python
class RuntimeMetricsSampler:
    """Background sampler for runtime Prometheus metrics during a benchmark."""

    def __init__(
        self,
        *,
        base_url: str,
        session: aiohttp.ClientSession,
        extra_headers: dict[str, str] | None = None,
        interval_s: float = 1.0,
    ) -> None:
        self.base_url = base_url
        self.session = session
        self.extra_headers = extra_headers
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        await self._scrape_once()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> RuntimeMetricsSummary:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._scrape_once()
        return RuntimeMetricsSummary.from_samples(self._samples)

    async def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                await self._scrape_once()

    async def _scrape_once(self) -> None:
        metrics = await fetch_runtime_metrics(
            self.base_url, self.session, self.extra_headers
        )
        if metrics is not None and metrics.gpu_kv_cache_usage is not None:
            self._samples.append(metrics.gpu_kv_cache_usage)
```

- [ ] **步骤 5：实现 result 注入 helper**

在 `serve.py` 中加入：

```python
def add_runtime_metrics_to_result(
    result: dict[str, Any],
    summary: RuntimeMetricsSummary,
) -> None:
    result["avg_gpu_kv_cache_usage"] = summary.avg_gpu_kv_cache_usage
    result["peak_gpu_kv_cache_usage"] = summary.peak_gpu_kv_cache_usage
```

- [ ] **步骤 6：接入 benchmark 生命周期**

在 `benchmark()` 中拉取 `spec_decode_metrics_before` 的语句后加入：

```python
    runtime_sampler = RuntimeMetricsSampler(
        base_url=base_url,
        session=session,
        extra_headers=extra_headers,
    )
    await runtime_sampler.start()
```

把：

```python
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
```

改为：

```python
    try:
        outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    finally:
        runtime_metrics_summary = await runtime_sampler.stop()
```

在 result dict 创建完成且 SpecDecode 写入前加入：

```python
    add_runtime_metrics_to_result(result, runtime_metrics_summary)
```

- [ ] **步骤 7：运行 serve metrics 测试**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_serve_metrics.py
```

预期：通过。

- [ ] **步骤 8：提交 sampler 接入变更**

运行：

```bash
git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "feat: sample gpu kv cache usage during benchmark"
```

---

### 任务 4：CSV/XLSX 行抽取和 SpecDecode 字段别名

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`

- [ ] **步骤 1：为 GPU KV cache usage 行字段编写失败测试**

在 `vllm_standalone_bench/tests/test_extract_row.py` 追加：

```python
def test_extract_row_includes_gpu_kv_cache_usage():
    result = {
        **_result(),
        "avg_gpu_kv_cache_usage": 9.12345,
        "peak_gpu_kv_cache_usage": 12.67891,
    }

    row = m._extract_row(
        result,
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True,
    )

    assert row["avg_gpu_kv_cache_usage"] == 9.1235
    assert row["peak_gpu_kv_cache_usage"] == 12.6789
```

- [ ] **步骤 2：为 GPU KV cache usage 默认值编写失败测试**

在同一个测试文件追加：

```python
def test_extract_row_gpu_kv_cache_usage_defaults_when_missing():
    row = m._extract_row(
        _result(),
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True,
    )

    assert row["avg_gpu_kv_cache_usage"] == 0.0
    assert row["peak_gpu_kv_cache_usage"] == 0.0
```

- [ ] **步骤 3：为 SpecDecode token 字段别名编写失败测试**

在同一个测试文件追加：

```python
def test_extract_row_spec_decode_token_aliases_from_serve_result():
    result = {
        **_result(),
        "spec_decode_accepted_tokens": 61,
        "spec_decode_draft_tokens": 280,
    }

    row = m._extract_row(
        result,
        in_len=1024, out_len=512, parallel_num=4, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True,
    )

    assert row["spec_decode_num_accepted_tokens"] == 61
    assert row["spec_decode_num_draft_tokens"] == 280
```

- [ ] **步骤 4：更新表头测试的 required 列**

把 `test_csv_headers_match_row_keys()` 里的 `required` 元组扩展为包含：

```python
"avg_gpu_kv_cache_usage",
"peak_gpu_kv_cache_usage",
```

- [ ] **步骤 5：运行新增测试并确认失败**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_extract_row.py -q
```

预期：失败，报错包含 `KeyError` 或断言失败，指出新增 GPU KV cache usage 列或 SpecDecode alias 还没有实现。

- [ ] **步骤 6：实现 `_i_any()` helper**

在 `_extract_row()` 的 `_i()` 后加入：

```python
    def _i_any(*keys: str, default=0) -> int:
        for key in keys:
            if result.get(key) is not None:
                return int(result.get(key) or default)
        return default
```

- [ ] **步骤 7：在 row 中加入 GPU KV cache usage 字段**

在 `_extract_row()` 返回 dict 的缓存列后加入：

```python
        'avg_gpu_kv_cache_usage': _f('avg_gpu_kv_cache_usage'),
        'peak_gpu_kv_cache_usage': _f('peak_gpu_kv_cache_usage'),
```

- [ ] **步骤 8：兼容 SpecDecode accepted/draft token 字段名**

把返回 dict 中这两行：

```python
        'spec_decode_num_accepted_tokens': _i('spec_decode_num_accepted_tokens'),
        'spec_decode_num_draft_tokens': _i('spec_decode_num_draft_tokens'),
```

改为：

```python
        'spec_decode_num_accepted_tokens': _i_any(
            'spec_decode_num_accepted_tokens',
            'spec_decode_accepted_tokens',
        ),
        'spec_decode_num_draft_tokens': _i_any(
            'spec_decode_num_draft_tokens',
            'spec_decode_draft_tokens',
        ),
```

- [ ] **步骤 9：更新英文和中文表头**

把 `CSV_HEADERS` 中：

```python
    'avg_cached_tokens', 'cache_hit_rate',
    'spec_decode_acceptance_rate', 'spec_decode_system_efficiency',
```

改为：

```python
    'avg_cached_tokens', 'cache_hit_rate',
    'avg_gpu_kv_cache_usage', 'peak_gpu_kv_cache_usage',
    'spec_decode_acceptance_rate', 'spec_decode_system_efficiency',
```

把 `CSV_HEADERS_ZH` 中：

```python
    '平均缓存命中tokens', '缓存命中率(%)',
    'SpecDecode接受率(%)', 'SpecDecode系统效率',
```

改为：

```python
    '平均缓存命中tokens', '缓存命中率(%)',
    '平均GPU KV缓存占用率(%)', '峰值GPU KV缓存占用率(%)',
    'SpecDecode接受率(%)', 'SpecDecode系统效率',
```

- [ ] **步骤 10：更新 XLSX 指标说明**

在 `explain` 中 `cache_hit_rate` 后加入：

```python
        ('avg_gpu_kv_cache_usage', '平均 GPU KV cache 占用率(%)', 'benchmark 运行期间定期采样 /metrics 中 GPU KV cache usage 后求平均值'),
        ('peak_gpu_kv_cache_usage', '峰值 GPU KV cache 占用率(%)', 'benchmark 运行期间定期采样 /metrics 中 GPU KV cache usage 后取峰值'),
```

- [ ] **步骤 11：运行 extract row 测试**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_extract_row.py
```

预期：通过。

- [ ] **步骤 12：提交 CSV/XLSX 变更**

运行：

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py
git commit -m "feat: export gpu kv cache usage metrics"
```

---

### 任务 5：完整验证和文档收尾

**文件：**
- 修改：`docs/superpowers/plans/2026-07-01-kv-cache-usage-csv.md`

- [ ] **步骤 1：运行完整测试**

运行：

```bash
pytest -q vllm_standalone_bench/tests
```

预期：通过。

- [ ] **步骤 2：运行 diff whitespace 检查**

运行：

```bash
git diff --check
```

预期：无输出，退出码为 0。

- [ ] **步骤 3：查看本分支提交**

运行：

```bash
git log --oneline main..HEAD
```

预期：包含以下提交主题：

```text
feat: export gpu kv cache usage metrics
feat: sample gpu kv cache usage during benchmark
feat: parse gpu kv cache runtime metrics
docs: design gpu kv cache usage metrics
```

- [ ] **步骤 4：提交计划文档**

运行：

```bash
git add docs/superpowers/plans/2026-07-01-kv-cache-usage-csv.md
git commit -m "docs: plan gpu kv cache usage metrics"
```

如果计划文档已经在更早步骤提交，此步骤运行 `git status --short`，预期没有该计划文件的未提交变更。

---

## 自检

- 规格覆盖度：计划覆盖了 `/metrics` 采样、鉴权 header 透传、GPU KV cache usage avg/peak、CSV/XLSX 英文和中文表头、XLSX 指标说明、SpecDecode token 字段别名和失败默认值。
- 类型一致性：`RuntimeMetrics.gpu_kv_cache_usage` 是单次 scrape 的百分比值或 `None`；`RuntimeMetricsSummary.avg_gpu_kv_cache_usage` 和 `RuntimeMetricsSummary.peak_gpu_kv_cache_usage` 是写入 result 和 CSV 的浮点百分比值。
- 兼容性：`fetch_spec_decode_metrics()` 保留现有签名，调用方无需改变；CSV 原有列不删除，新增列位于 prefix cache 指标后。
