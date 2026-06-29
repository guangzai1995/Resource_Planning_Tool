# vLLM Prefix Ratio 固定总输入语义实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `vllm_standalone_bench` 的 `prefix_ratio` 从“额外追加到 input”修正为“占总 input 的比例”。

**架构：** `run_bench_multi.py` 统一计算 prefix/suffix 长度并保持 `input_len` 为总输入预算；`run_bench_serve.py` 按 `shared_prefix + unique_suffix = random_input_len` 生成随机 prompt；报表继续输出 `prefix_tokens`，但 `total_input_len` 在新语义下等于 `input_len`。

**技术栈：** Python 3.10、pytest、现有 `vllm_standalone_bench` shim 测试工程。

---

## 工作区与基线

- 工作区：`/Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input`
- 分支：`feat/vllm-prefix-ratio-total-input`
- 已验证基线：在 `vllm_standalone_bench/` 运行 `python3 -m pytest tests -q`，结果 `25 passed in 0.26s`。

## 文件结构

| 文件 | 职责 |
|---|---|
| `vllm_standalone_bench/run_bench_multi.py` | CLI 参数校验、prefix/suffix 长度计算、批量配置日志、CSV/XLSX 行抽取。 |
| `vllm_standalone_bench/run_bench_serve.py` | random 数据集请求生成，保证共享前缀和唯一后缀合计为总输入长度。 |
| `vllm_standalone_bench/run_bench.sh` | shell wrapper 注释，说明 `PREFIX_RATIO` 新语义。 |
| `vllm_standalone_bench/tests/test_random_dataset.py` | 锁定 random 请求生成长度：prefix 是 input 内部比例，不额外增加 prompt。 |
| `vllm_standalone_bench/tests/test_extract_row.py` | 锁定报表口径：`total_input_len == input_len`，合规按总输入算。 |

## 执行粒度说明

任务 1 和任务 2 构成同一个 TDD 红绿循环：先写失败测试，再实现最少代码让测试通过。执行时由同一个实现子智能体连续完成任务 1 和任务 2，并在任务 2 的 commit 之后进行规格合规审查和代码质量审查。

## 任务 1：用测试锁住 random prompt 总长度

**文件：**
- 修改：`vllm_standalone_bench/tests/test_random_dataset.py`
- 测试：`vllm_standalone_bench/tests/test_random_dataset.py`

- [ ] **步骤 1：编写失败的测试**

将现有测试替换为固定总输入预算的断言，并补一个全 prefix 边界：

```python
def test_random_prefix_prompt_len_uses_total_input_budget():
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=3,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [8, 8]
    assert [req.expected_output_len for req in requests] == [4, 4]
    assert requests[0].prompt != requests[1].prompt


def test_random_full_prefix_keeps_total_input_budget():
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=8,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [8, 8]
    assert [req.expected_output_len for req in requests] == [4, 4]
    assert requests[0].prompt == requests[1].prompt
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_random_dataset.py -q
```

预期：FAIL。第一个测试会看到旧行为生成 `11` 个 token；第二个测试会看到旧行为生成 `16` 个 token。

- [ ] **步骤 3：暂不实现，进入任务 2**

任务 1 只负责先写失败测试。实现放在任务 2，保持 TDD 失败信号清晰。

## 任务 2：实现 random 数据集固定总输入生成

**文件：**
- 修改：`vllm_standalone_bench/run_bench_serve.py:295-385`
- 测试：`vllm_standalone_bench/tests/test_random_dataset.py`

- [ ] **步骤 1：修改生成器文档和长度语义**

将 `_generate_random_requests()` docstring 中的旧语义：

```python
# prompt_len ≈ random_prefix_len + random_input_len
```

改成：

```python
# prompt_len ≈ random_input_len
# random_prefix_len 是 random_input_len 内部的共享前缀长度
```

- [ ] **步骤 2：编写最小实现代码**

在每个请求内把 `random_input_len` 视为总长度，按当前请求长度切 shared prefix 并生成 suffix：

```python
for i in range(args.num_prompts):
    in_len = _rand_len(args.random_input_len)
    out_len = _rand_len(args.random_output_len)
    effective_prefix_len = min(prefix_len, in_len)
    suffix_len = max(in_len - effective_prefix_len, 0)

    if tokenizer is not None and hasattr(tokenizer, 'decode'):
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
        suffix_ids = [random.randrange(vocab_size) for _ in range(suffix_len)]
        prompt, actual_len = _decode_to_target_len(
            shared_prefix_ids[:effective_prefix_len] + suffix_ids,
            in_len,
        )
    else:
        suffix_text = ' '.join(
            str(random.randint(0, 31999)) for _ in range(suffix_len)
        )
        prefix_text = shared_prefix_text
        if effective_prefix_len < prefix_len and shared_prefix_text:
            prefix_text = ' '.join(shared_prefix_text.split()[:effective_prefix_len])
        prompt = ' '.join(part for part in (prefix_text, suffix_text) if part)
        actual_len = in_len
```

保留后面的 `SampleRequest(...)` 构造不变。

- [ ] **步骤 3：运行 random dataset 测试验证通过**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_random_dataset.py -q
```

预期：PASS，输出包含 `2 passed`。

- [ ] **步骤 4：运行相关回归测试**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_random_dataset.py tests/test_integration.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/run_bench_serve.py vllm_standalone_bench/tests/test_random_dataset.py
git commit -m "fix(bench): 固定 prefix random 总输入长度"
```

## 任务 3：新增 prefix/suffix 计算 helper 和参数校验

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
- 测试：`vllm_standalone_bench/tests/test_extract_row.py`

- [ ] **步骤 1：编写失败的 helper 测试**

在 `tests/test_extract_row.py` 中加入：

```python
import pytest
```

并添加测试：

```python
def test_derive_prefix_suffix_tokens_from_total_input():
    assert m._derive_prefix_suffix_tokens(128, 0.8) == (102, 26)
    assert m._derive_prefix_suffix_tokens(128, 0.0) == (0, 128)
    assert m._derive_prefix_suffix_tokens(128, 1.0) == (128, 0)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_derive_prefix_suffix_tokens_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="--prefix-ratio"):
        m._derive_prefix_suffix_tokens(128, ratio)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_extract_row.py::test_derive_prefix_suffix_tokens_from_total_input tests/test_extract_row.py::test_derive_prefix_suffix_tokens_rejects_invalid_ratio -q
```

预期：FAIL，报错包含 `AttributeError: module 'run_bench_multi' has no attribute '_derive_prefix_suffix_tokens'`。

- [ ] **步骤 3：实现 helper**

在 `run_bench_multi.py` 的 `decide_token_usage_source()` 之后、`_extract_row()` 之前添加：

```python
def _derive_prefix_suffix_tokens(input_len: int, prefix_ratio: float) -> Tuple[int, int]:
    """Return shared-prefix and unique-suffix lengths within total input_len."""
    if prefix_ratio < 0.0 or prefix_ratio > 1.0:
        raise ValueError("--prefix-ratio must be between 0.0 and 1.0")
    prefix_tokens = int(input_len * prefix_ratio)
    suffix_tokens = input_len - prefix_tokens
    return prefix_tokens, suffix_tokens
```

`Tuple` 已在文件顶部导入，无需新增 typing import。

- [ ] **步骤 4：运行 helper 测试验证通过**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_extract_row.py::test_derive_prefix_suffix_tokens_from_total_input tests/test_extract_row.py::test_derive_prefix_suffix_tokens_rejects_invalid_ratio -q
```

预期：PASS，输出包含 `3 passed`。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py
git commit -m "test(bench): 锁定 prefix ratio 总输入计算"
```

## 任务 4：修正报表 total_input_len 和 input_compliance

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py:190-257`
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
- 测试：`vllm_standalone_bench/tests/test_extract_row.py`

- [ ] **步骤 1：改写失败的报表测试**

将 `test_extract_row_prefix_total_input_len()` 改为：

```python
def test_extract_row_prefix_total_input_len_uses_total_input_budget():
    row = m._extract_row(
        _result(completed=3, total_in=384, total_out=24),
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai", prefix_tokens=102,
        prefix_ratio=0.8, has_tokenizer=True)
    assert row["total_input_len"] == 128
    assert row["input_len"] == 128
    assert row["prefix_tokens"] == 102
    assert row["avg_input_tokens"] == 128.0
    assert row["input_compliance"] == 100.0
```

将 `test_input_compliance_uses_unrounded_mean()` 改为：

```python
def test_input_compliance_uses_unrounded_mean():
    """输入合规基于未取整均值和总输入目标长度。"""
    row = m._extract_row(
        _result(total_in=383, total_out=24, completed=3),
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", prefix_tokens=102,
        prefix_ratio=0.8, has_tokenizer=True)
    assert row["avg_input_tokens"] == 127.7
    assert row["input_compliance"] == 99.7
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_extract_row.py::test_extract_row_prefix_total_input_len_uses_total_input_budget tests/test_extract_row.py::test_input_compliance_uses_unrounded_mean -q
```

预期：FAIL。旧实现会把 `total_input_len` 算成 `230`，`input_compliance` 算成约 `55.5`。

- [ ] **步骤 3：实现报表语义修正**

在 `_extract_row()` 中把：

```python
total_input_len = in_len + prefix_tokens
```

改为：

```python
total_input_len = in_len
```

同步更新返回 dict 注释：

```python
'input_len':       in_len,           # requested 总输入长度
'total_input_len': total_input_len,  # 新语义下等于 input_len
```

- [ ] **步骤 4：运行 extract row 测试验证通过**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_extract_row.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py
git commit -m "fix(bench): 按总输入计算 prefix 报表口径"
```

## 任务 5：修正 batch runner 日志、参数校验和帮助文本

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py:486-511`
- 修改：`vllm_standalone_bench/run_bench_multi.py:669-675`
- 修改：`vllm_standalone_bench/run_bench.sh:95-110`
- 测试：`vllm_standalone_bench/tests/test_extract_row.py`

- [ ] **步骤 1：用 helper 更新 batch 配置计算**

在 `_run_all()` 内、`logger.info(...)` 之前计算一次：

```python
prefix_ratio = our_args.prefix_ratio
prefix_tokens, suffix_tokens = _derive_prefix_suffix_tokens(in_len, prefix_ratio)
```

更新日志参数，让每组显示总输入、prefix 和 suffix：

```python
logger.info(
    "\n%s\n[%d/%d] 开始测试: input=%d, output=%d, parallel=%d, "
    "num_prompts=%d (=%d×%d epochs)%s\n%s",
    "─" * 65,
    config_count, total_configs,
    in_len, out_len, parallel_num,
    parallel_num * our_args.epochs, parallel_num, our_args.epochs,
    (f"  total_input={in_len} prefix={prefix_tokens}tok"
     f" suffix={suffix_tokens}tok({prefix_ratio * 100:.0f}%)"
     if prefix_ratio > 0 else ""),
    "─" * 65,
)
```

把旧注释：

```python
# prefix_tokens 不计入 input_len（input_len 仅表示后缀唯一部分）
# 实际 prompt_len ≈ prefix_tokens + input_len
```

改为：

```python
# input_len 表示总 prompt token 预算；prefix_tokens 是其中共享前缀部分。
# run_bench_serve.py 会生成 shared_prefix + unique_suffix，总长约等于 input_len。
```

并保留：

```python
cfg.input_len = in_len
cfg.random_prefix_len = prefix_tokens
```

- [ ] **步骤 2：确保非法 prefix_ratio 在跑服务前失败**

在 `_run_all()` 的 `base = _build_base_args(our_args)` 之后加入：

```python
_derive_prefix_suffix_tokens(1, our_args.prefix_ratio)
```

这样 `--prefix-ratio -0.1` 或 `1.1` 会在构造矩阵前抛出明确 `ValueError`。

- [ ] **步骤 3：更新 CLI help**

将 `--prefix-ratio` help 中的旧句子：

```python
'实际 prompt_len ≈ input_len × (1 + prefix_ratio)。'
```

替换为：

```python
'实际 prompt_len ≈ input_len；共享前缀 token 数约为 input_len × prefix_ratio。'
```

- [ ] **步骤 4：更新 shell wrapper 注释**

将 `run_bench.sh` 中旧说明：

```bash
# 原理：取值 X 时，每组测试中所有请求共享同一段前缀文本（占 input_len 的 X 倍 token），
#        后缀部分每个请求独立随机生成，保证请求间差异性。
#        实际 prompt_len ≈ input_len × (1 + X)
#
# 典型用法：
#   PREFIX_RATIO=0.0   → 全随机，无共享（测试无缓存基准）
#   PREFIX_RATIO=0.5   → 50% 前缀共享（input_len=512 → 前缀256 + 后缀512）
#   PREFIX_RATIO=0.9   → 90% 前缀共享（高缓存命中率场景）
```

替换为：

```bash
# 原理：取值 X 时，每组测试中所有请求共享 input_len * X 的前缀 token，
#        后缀部分每个请求独立随机生成，prefix + suffix 总长仍约等于 input_len。
#
# 典型用法：
#   PREFIX_RATIO=0.0   → 全随机，无共享（测试无缓存基准）
#   PREFIX_RATIO=0.5   → 50% 前缀共享（input_len=512 → 前缀256 + 后缀256）
#   PREFIX_RATIO=0.9   → 90% 前缀共享（input_len=512 → 前缀460 + 后缀52）
```

- [ ] **步骤 5：运行相关测试**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests/test_extract_row.py tests/test_random_dataset.py -q
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/run_bench.sh vllm_standalone_bench/tests/test_extract_row.py
git commit -m "fix(bench): 更新 prefix ratio 批量语义"
```

## 任务 6：全量回归和最终核对

**文件：**
- 验证：`vllm_standalone_bench/tests/`
- 验证：`docs/superpowers/specs/2026-06-29-vllm-prefix-ratio-total-input-design.md`

- [ ] **步骤 1：运行完整测试套件**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input/vllm_standalone_bench
python3 -m pytest tests -q
```

预期：PASS，所有测试通过。

- [ ] **步骤 2：扫描旧语义文案**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input
rg -n "input_len × \\(1 \\+ prefix_ratio\\)|prefix_tokens 不计入 input_len|random_prefix_len \\+ random_input_len|前缀256 \\+ 后缀512" vllm_standalone_bench
```

预期：无输出。若出现输出，更新对应注释或帮助文本，再重新运行扫描。

- [ ] **步骤 3：查看最终 diff**

运行：

```bash
cd /Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input
git diff --stat main...HEAD
git status --short
```

预期：只包含本任务相关文件；`git status --short` 为空。

- [ ] **步骤 4：记录验证结果**

在最终回复中报告：

```text
工作树：/Resource_Planning_Tool/.worktrees/vllm-prefix-ratio-total-input
分支：feat/vllm-prefix-ratio-total-input
验证：python3 -m pytest tests -q
旧语义扫描：rg ... 无输出
```

## 执行注意事项

- 不要修改主工作区 `/Resource_Planning_Tool`；所有实现都在 worktree 内进行。
- 不要提交 `model/`、`results/`、图片或主工作区已有未跟踪文件。
- 如果 `git commit` 因缺少身份失败，使用一次性参数提交：

```bash
git -c user.name=Codex -c user.email=codex@local commit -m "<message>"
```

- 如果某个测试失败，先按 `systematic-debugging` 调查，不要直接改实现猜测。
