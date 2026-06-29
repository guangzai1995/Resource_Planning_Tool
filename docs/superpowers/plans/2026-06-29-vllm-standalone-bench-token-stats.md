# vLLM Standalone Bench Token Stats 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修正 `vllm_standalone_bench` 的输入/输出 token 统计口径，确保本地 tokenizer 用于请求构造与校验，API usage 用于最终实测统计，缺失 usage 时明确标记 fallback 来源。

**架构：** 在 `run_bench_serve.py` 的 random dataset shim 中增加小型 token 长度校正辅助函数，避免 prefix 场景只记录 suffix 长度。在 `vllm_bench/serve.py` 中把 output token 来源拆成 usage 与 tokenizer fallback 计数。在 `run_bench_multi.py` 中输出 `input_compliance`、更精确的 `token_source`，并保持 requested length 与 measured length 分离。

**技术栈：** Python 3、pytest、现有 vLLM standalone shim、OpenAI-compatible streaming usage。

---

## 文件结构

- 修改：`vllm_standalone_bench/run_bench_serve.py`
  - 负责生成 random/sharegpt 请求样本，并注入 vLLM benchmark shim。
  - 新增 tokenizer 长度校正辅助函数，保证 prefix + suffix 的 `SampleRequest.prompt_len` 使用实际重编码长度。
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
  - 负责从 `RequestFuncOutput` 计算 benchmark metrics。
  - 新增 tokenizer fallback 计数，保留 usage 上报计数。
- 修改：`vllm_standalone_bench/run_bench_multi.py`
  - 负责多配置调用和 CSV/XLSX 聚合。
  - 新增 `input_compliance` 列和更精确的 `token_source`。
- 新增或修改测试：`vllm_standalone_bench/tests/test_random_dataset.py`
  - 覆盖 random/prefix 请求生成的 full prompt length。
- 修改测试：`vllm_standalone_bench/tests/test_extract_row.py`
  - 覆盖 token source、input compliance、表头一致性。
- 修改测试：`vllm_standalone_bench/tests/test_serve_metrics.py`
  - 覆盖 usage 与 tokenizer fallback 计数。

## 任务 1：锁定 random prefix 请求长度

**文件：**
- 创建：`vllm_standalone_bench/tests/test_random_dataset.py`
- 修改：`vllm_standalone_bench/run_bench_serve.py`

- [ ] **步骤 1：编写失败的测试**

新增测试文件，使用一个确定性 fake tokenizer，验证 `random_prefix_len + random_input_len` 进入 `SampleRequest.prompt_len`。

```python
import argparse

import run_bench_serve as rbs


class FakeTokenizer:
    vocab_size = 128

    def decode(self, token_ids):
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = text.split() if text else []

        return Encoded()


def test_random_prefix_prompt_len_includes_prefix_and_suffix():
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=3,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [11, 11]
    assert [req.expected_output_len for req in requests] == [4, 4]
    assert requests[0].prompt != requests[1].prompt
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest vllm_standalone_bench/tests/test_random_dataset.py -q`

预期：失败；当前实现可能因 prefix 与 suffix 拼接无分隔或无校正导致 `prompt_len` 不是 `11`。

- [ ] **步骤 3：实现最少代码**

在 `run_bench_serve.py` 中新增辅助函数，并在 `_generate_random_requests` 使用它。

```python
def _encode_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _decode_to_target_len(tokenizer, token_ids: list[int],
                          target_len: int, max_retry: int = 8) -> tuple[str, int]:
    vocab_size = max(int(getattr(tokenizer, "vocab_size", 32000)), 1)
    current_ids = list(token_ids)
    for _ in range(max_retry + 1):
        text = tokenizer.decode(current_ids)
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        actual_len = len(encoded)
        if actual_len == target_len:
            return text, actual_len
        if actual_len > target_len:
            current_ids = current_ids[:max(target_len, 1)]
        else:
            current_ids.extend(random.randrange(vocab_size)
                               for _ in range(target_len - actual_len))

    text = tokenizer.decode(current_ids)
    return text, _encode_len(tokenizer, text)
```

Use one combined token sequence for prefix plus suffix:

```python
combined_ids = shared_prefix_ids + suffix_ids
prompt, actual_len = _decode_to_target_len(
    tokenizer, combined_ids, prefix_len + in_len)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest vllm_standalone_bench/tests/test_random_dataset.py -q`

预期：新增测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/run_bench_serve.py vllm_standalone_bench/tests/test_random_dataset.py
git commit -m "fix(bench): 校正 standalone random 输入长度（任务 1）"
```

## 任务 2：区分 usage 与 tokenizer fallback 统计来源

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
- 修改：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：编写失败的测试**

在 `test_serve_metrics.py` 增加 tokenizer fallback 计数断言。

```python
def test_metrics_counts_tokenizer_fallback_outputs():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            class Encoded:
                input_ids = text.split()
            return Encoded()

    outputs = [
        _out(True, output_tokens=0, finish_reason="stop",
             generated_text="a b c", ttft=0.05, latency=0.35, itl=[0.1, 0.1]),
    ]
    metrics, lens = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=Tok(), selected_percentiles=[50], goodput_config_dict={},
    )

    assert lens == [3]
    assert metrics.usage_reported_count == 0
    assert metrics.tokenizer_fallback_count == 1
    assert metrics.total_output == 3
```

Update `_out` helper to accept `generated_text`:

```python
def _out(..., generated_text="abc"):
    return RequestFuncOutput(..., generated_text=generated_text)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest vllm_standalone_bench/tests/test_serve_metrics.py::test_metrics_counts_tokenizer_fallback_outputs -q`

预期：失败，`BenchmarkMetrics` 没有 `tokenizer_fallback_count`。

- [ ] **步骤 3：实现最少代码**

在 `BenchmarkMetrics` 添加字段：

```python
tokenizer_fallback_count: int = 0
```

在 `calculate_metrics` 中，当 `output_tokens` 缺失且 tokenizer 成功重编码时递增：

```python
tokenizer_fallback_count = 0
...
if not output_len:
    if tokenizer is None:
        output_len = 1
    else:
        output_len = len(tokenizer(outputs[i].generated_text,
                         add_special_tokens=False).input_ids)
        tokenizer_fallback_count += 1
```

构造 metrics 和 result dict 时传出该字段：

```python
tokenizer_fallback_count=tokenizer_fallback_count
```

```python
"tokenizer_fallback_count": metrics.tokenizer_fallback_count,
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest vllm_standalone_bench/tests/test_serve_metrics.py -q`

预期：相关 metrics 测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "fix(bench): 标记 tokenizer fallback 输出统计（任务 2）"
```

## 任务 3：修正 CSV/XLSX token source 与 input compliance

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
- 修改：`vllm_standalone_bench/tests/test_integration.py`

- [ ] **步骤 1：编写失败的测试**

更新 token source 断言：

```python
def test_token_source_tokenizer_fallback_when_no_usage_but_has_tok():
    assert m.decide_token_usage_source(
        usage_reported_count=0, tokenizer_fallback_count=3,
        completed=3, has_tokenizer=True) == "tokenizer_fallback"


def test_token_source_usage_partial_when_some_usage_missing():
    assert m.decide_token_usage_source(
        usage_reported_count=2, tokenizer_fallback_count=1,
        completed=3, has_tokenizer=True) == "partial_usage"


def test_token_source_client_estimate_without_usage_or_tokenizer():
    assert m.decide_token_usage_source(
        usage_reported_count=0, tokenizer_fallback_count=0,
        completed=3, has_tokenizer=False) == "client_estimate"
```

Add input compliance assertion:

```python
def test_input_compliance_uses_total_input_len():
    row = m._extract_row(
        _result(total_in=690, total_out=24, completed=3),
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai", prefix_tokens=102,
        prefix_ratio=0.8, has_tokenizer=True)

    assert row["avg_input_tokens"] == 230.0
    assert row["total_input_len"] == 230
    assert row["input_compliance"] == 100.0
```

Update header test required fields:

```python
for required in ("total_input_len", "input_compliance", "output_compliance",
                 "finish_reason_length_pct", "token_source"):
    assert required in m.CSV_HEADERS
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest vllm_standalone_bench/tests/test_extract_row.py -q`

预期：失败；当前 `decide_token_usage_source` 签名不含 `tokenizer_fallback_count`，且没有 `input_compliance`。

- [ ] **步骤 3：实现最少代码**

Update function signature and source policy:

```python
def decide_token_usage_source(*, usage_reported_count: int,
                              tokenizer_fallback_count: int,
                              completed: int,
                              has_tokenizer: bool) -> str:
    if completed <= 0:
        return "none"
    if usage_reported_count >= completed:
        return "usage"
    if usage_reported_count > 0:
        return "partial_usage"
    if tokenizer_fallback_count > 0 or has_tokenizer:
        return "tokenizer_fallback"
    return "client_estimate"
```

Compute input compliance:

```python
raw_avg_in = total_in / completed if completed > 0 else 0.0
total_input_len = in_len + prefix_tokens
input_compliance = (
    round(raw_avg_in / total_input_len * 100, 1)
    if total_input_len > 0 else 0.0
)
```

Add `input_compliance` to row, `CSV_HEADERS`, and `CSV_HEADERS_ZH`.

Pass `tokenizer_fallback_count=_i("tokenizer_fallback_count")` into
`decide_token_usage_source`.

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest vllm_standalone_bench/tests/test_extract_row.py vllm_standalone_bench/tests/test_integration.py -q`

预期：提取和集成测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py vllm_standalone_bench/tests/test_integration.py
git commit -m "fix(bench): 输出输入合规和 token 来源（任务 3）"
```

## 任务 4：全量验证并提交计划与剩余 shim 文件

**文件：**
- 创建：`vllm_standalone_bench/vllm_bench/__init__.py`
- 创建：`vllm_standalone_bench/vllm_bench/lib/__init__.py`
- 创建：`vllm_standalone_bench/vllm_bench/lib/ready_checker.py`
- 创建：`vllm_standalone_bench/vllm_bench/lib/utils.py`
- 创建：`docs/superpowers/plans/2026-06-29-vllm-standalone-bench-token-stats.md`

- [ ] **步骤 1：确认未跟踪 shim 文件是测试依赖**

运行：`git status --short vllm_standalone_bench`

预期：剩余 shim 文件显示为未跟踪，并且 baseline 测试需要这些 shim 支持 `run_bench_serve.py` 导入。

- [ ] **步骤 2：运行全量 standalone 测试**

运行：`python -m pytest vllm_standalone_bench/tests -q`

预期：全部测试通过。

- [ ] **步骤 3：检查差异范围**

运行：`git diff --stat`

预期：只包含 `vllm_standalone_bench` 和本计划文件。

- [ ] **步骤 4：Commit**

```bash
git add docs/superpowers/plans/2026-06-29-vllm-standalone-bench-token-stats.md
git add vllm_standalone_bench/vllm_bench/__init__.py
git add vllm_standalone_bench/vllm_bench/lib/__init__.py
git add vllm_standalone_bench/vllm_bench/lib/ready_checker.py
git add vllm_standalone_bench/vllm_bench/lib/utils.py
git commit -m "chore(bench): 跟踪 standalone shim 依赖（任务 4）"
```

## 任务 5：合并回 main 并验证

**文件：**
- 无直接代码编辑。

- [ ] **步骤 1：切回主工作区**

运行：`cd /work/development-code/Resource_Planning_Tool`

预期：当前目录为主工作区。

- [ ] **步骤 2：合并分支**

运行：`git merge --no-ff fix/vllm-standalone-token-stats`

预期：合并成功。如遇冲突，只解决 `vllm_standalone_bench` 和计划文件相关冲突，不改动无关用户文件。

- [ ] **步骤 3：主工作区验证**

运行：`python -m pytest vllm_standalone_bench/tests -q`

预期：全部测试通过。

- [ ] **步骤 4：报告结果**

报告合并 commit、测试命令、通过数量，以及任何未处理的主工作区既有脏文件。
