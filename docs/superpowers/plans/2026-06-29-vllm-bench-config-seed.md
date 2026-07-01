# vLLM Bench 配置级 Seed 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让 `vllm_standalone_bench/run_bench_multi.py` 默认对每个 benchmark 配置使用稳定派生的独立 seed，避免高并发档位复用低并发档位的 prompt 序列。

**架构：** 在批量 runner 内新增 seed 校验、稳定哈希派生和兼容模式开关。`run_bench_multi.py` 在每次调用 `_serve.main_async(cfg)` 前写入本配置的 `cfg.seed`，并把实际 seed 落到 CSV/XLSX 结果中；`run_bench.sh` 暴露 shell 变量并转成 Python 参数。

**技术栈：** Python 标准库 `hashlib`、`argparse`、现有 pytest 测试；Bash 参数数组。

---

## 文件结构

- 修改：`vllm_standalone_bench/run_bench_multi.py`
  - 增加 seed 派生 helper。
  - 增加 CLI 参数 `--seed` 和 `--no-vary-seed-by-config`。
  - 在 `_run_all()` 中设置每组 `cfg.seed`。
  - 在 `_extract_row()` 和 CSV/XLSX 表头中新增 `seed`。
- 修改：`vllm_standalone_bench/run_bench.sh`
  - 增加 `SEED` 和 `VARY_SEED_BY_CONFIG` 配置。
  - 把配置转换为 `run_bench_multi.py` 参数。
  - 在摘要中打印 seed 策略。
- 创建：`vllm_standalone_bench/tests/test_config_seed.py`
  - 单测 seed 校验、派生稳定性、不同配置差异和兼容模式。
- 修改：`vllm_standalone_bench/tests/test_integration.py`
  - 给测试 namespace 补齐新参数。
  - 验证 `_run_all()` 传给 `main_async` 的 `cfg.seed`。
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
  - 验证 `seed` 在 row 和 CSV 表头中存在。

---

### 任务 1：为 seed helper 写失败测试

**文件：**
- 创建：`vllm_standalone_bench/tests/test_config_seed.py`
- 修改：`vllm_standalone_bench/tests/test_integration.py`

- [ ] **步骤 1：编写 seed helper 单测**

创建 `vllm_standalone_bench/tests/test_config_seed.py`：

```python
import pytest

import run_bench_multi as m


def test_derive_config_seed_is_stable():
    first = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )
    second = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )

    assert first == second
    assert 0 <= first < 2**32


def test_derive_config_seed_changes_with_parallel():
    low_parallel = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=1,
        prefix_ratio=0.8,
        config_index=1,
    )
    high_parallel = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=2,
    )

    assert low_parallel != high_parallel


def test_effective_config_seed_uses_base_seed_when_vary_disabled():
    assert m.effective_config_seed(
        base_seed=123,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
        vary_seed_by_config=False,
    ) == 123


def test_validate_seed_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="--seed"):
        m.validate_seed(-1)

    with pytest.raises(ValueError, match="--seed"):
        m.validate_seed(2**32)
```

- [ ] **步骤 2：运行 seed helper 单测验证失败**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_config_seed.py
```

预期：FAIL，错误包含：

```text
AttributeError: module 'run_bench_multi' has no attribute 'derive_config_seed'
```

- [ ] **步骤 3：给 integration helper 补新参数的失败测试**

修改 `vllm_standalone_bench/tests/test_integration.py` 的 `_run_to_rows()`，在 `argparse.Namespace(...)` 中补上新参数：

```python
        seed=0, no_vary_seed_by_config=False,
```

在文件末尾增加：

```python
def _run_and_capture_seeds(monkeypatch, *, no_vary_seed_by_config=False):
    import argparse

    captured = []

    async def _fake_main_async(cfg):
        captured.append(cfg.seed)
        return _fake_result(cfg.input_len, cfg.output_len, completed=cfg.num_prompts)

    monkeypatch.setattr(serve, "main_async", _fake_main_async)

    our_args = argparse.Namespace(
        model="m", served_model_name=None, backend="openai-chat",
        base_url=None, host="127.0.0.1", port=8000, insecure=False, api_key=None,
        tokenizer="/some/tok",
        input_lens=[128], output_lens=[8], cross_product=False,
        parallel_nums=[1, 4, 8], epochs=1, sleep_between=0, warmup_requests=0,
        prefix_ratio=0.8,
        seed=123, no_vary_seed_by_config=no_vary_seed_by_config,
        max_ttft_ms=None, min_throughput_tok_s=None, min_output_compliance=0.95,
        output_csv=None, output_xlsx=None, result_dir=None,
    )
    rows = m._run_all(our_args)
    return captured, rows


def test_run_all_varies_seed_by_config_by_default(monkeypatch):
    captured, rows = _run_and_capture_seeds(monkeypatch)

    assert len(captured) == 3
    assert len(set(captured)) == 3
    assert [row["seed"] for row in rows] == captured


def test_run_all_can_use_fixed_seed_for_compatibility(monkeypatch):
    captured, rows = _run_and_capture_seeds(
        monkeypatch,
        no_vary_seed_by_config=True,
    )

    assert captured == [123, 123, 123]
    assert [row["seed"] for row in rows] == [123, 123, 123]
```

- [ ] **步骤 4：运行 integration 新测试验证失败**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_integration.py
```

预期：FAIL，错误包含：

```text
AttributeError: 'Namespace' object has no attribute 'seed'
```

或：

```text
KeyError: 'seed'
```

- [ ] **步骤 5：Commit 失败测试**

```bash
git add vllm_standalone_bench/tests/test_config_seed.py vllm_standalone_bench/tests/test_integration.py
git commit -m "test: 覆盖 vllm bench 配置级 seed"
```

---

### 任务 2：实现 seed helper 和参数校验

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
- 测试：`vllm_standalone_bench/tests/test_config_seed.py`

- [ ] **步骤 1：在 `run_bench_multi.py` 增加 import 和 helper**

在 import 区加入：

```python
import hashlib
```

在 `decide_token_usage_source()` 前增加：

```python
MAX_SEED_VALUE = 2**32


def validate_seed(seed: int) -> None:
    if not 0 <= seed < MAX_SEED_VALUE:
        raise ValueError(f"--seed 必须满足 0 <= seed < {MAX_SEED_VALUE}，当前值: {seed}")


def derive_config_seed(
    *,
    base_seed: int,
    input_len: int,
    output_len: int,
    parallel_num: int,
    prefix_ratio: float,
    config_index: int,
) -> int:
    key = (
        f"{base_seed}:{input_len}:{output_len}:{parallel_num}:"
        f"{prefix_ratio:.12g}:{config_index}"
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def effective_config_seed(
    *,
    base_seed: int,
    input_len: int,
    output_len: int,
    parallel_num: int,
    prefix_ratio: float,
    config_index: int,
    vary_seed_by_config: bool,
) -> int:
    validate_seed(base_seed)
    if not vary_seed_by_config:
        return base_seed
    return derive_config_seed(
        base_seed=base_seed,
        input_len=input_len,
        output_len=output_len,
        parallel_num=parallel_num,
        prefix_ratio=prefix_ratio,
        config_index=config_index,
    )
```

- [ ] **步骤 2：运行 seed helper 单测验证通过**

运行：

```bash
pytest -q vllm_standalone_bench/tests/test_config_seed.py
```

预期：

```text
4 passed
```

- [ ] **步骤 3：Commit helper 实现**

```bash
git add vllm_standalone_bench/run_bench_multi.py
git commit -m "feat: 添加配置级 seed 派生"
```

---

### 任务 3：接入批量 runner、CSV 和 CLI

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
- 测试：`vllm_standalone_bench/tests/test_integration.py`
- 测试：`vllm_standalone_bench/tests/test_extract_row.py`

- [ ] **步骤 1：扩展 `_extract_row()` 接收并输出 seed**

修改 `_extract_row()` 函数签名：

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
    seed: int = 0,
) -> dict:
```

在返回 dict 的配置字段中加入：

```python
        'seed':            seed,
```

位置放在 `'num_prompts'` 后面。

- [ ] **步骤 2：更新 CSV 表头**

修改 `CSV_HEADERS`：

```python
    'parallel_num', 'epochs', 'num_prompts', 'seed',
```

修改 `CSV_HEADERS_ZH`：

```python
    '并发数', '测试轮数', '总请求数', '随机种子',
```

- [ ] **步骤 3：更新 extract row 测试**

在 `vllm_standalone_bench/tests/test_extract_row.py::test_csv_headers_match_row_keys` 中，把必需列检查改为：

```python
    for required in ("total_input_len", "input_compliance", "output_compliance",
                     "finish_reason_length_pct", "token_source", "seed"):
        assert required in m.CSV_HEADERS, f"新列 {required} 未进 CSV_HEADERS"
    assert row["seed"] == 0
```

新增测试：

```python
def test_extract_row_records_effective_seed():
    row = m._extract_row(
        {"completed": 1, "total_input_tokens": 5, "total_output_tokens": 8,
         "usage_reported_count": 1, "tokenizer_fallback_count": 0,
         "finish_reason_length": 1, "num_prompts": 1,
         "request_throughput": 1.0, "output_throughput": 8.0, "duration": 1.0},
        in_len=5, out_len=8, parallel_num=1, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True, seed=98765)

    assert row["seed"] == 98765
```

- [ ] **步骤 4：给 parser 增加 seed 参数**

在 `build_arg_parser()` 的测试配置参数组中，`--prefix-ratio` 后加入：

```python
    bench.add_argument('--seed', type=int, default=0,
                       help='随机种子基值。默认每个配置会基于该值派生独立 seed（默认: 0）')
    bench.add_argument('--no-vary-seed-by-config', action='store_true', default=False,
                       help='兼容旧行为：所有配置复用 --seed，不再按配置派生独立 seed')
```

- [ ] **步骤 5：在 `_run_all()` 设置每组有效 seed**

在 `_run_all()` 构建 `base` 前加入：

```python
    validate_seed(our_args.seed)
```

在参数汇总日志中加入：

```python
    vary_seed_by_config = not our_args.no_vary_seed_by_config
    logger.info("  随机种子  : base=%d  vary_by_config=%s",
                our_args.seed, vary_seed_by_config)
```

在每组配置计算 `prefix_tokens` 后、ready check 前加入：

```python
            effective_seed = effective_config_seed(
                base_seed=our_args.seed,
                input_len=in_len,
                output_len=out_len,
                parallel_num=parallel_num,
                prefix_ratio=prefix_ratio,
                config_index=config_count,
                vary_seed_by_config=vary_seed_by_config,
            )
            cfg.seed = effective_seed
```

修改开始测试日志，把 seed 打印出来：

```python
                "\n%s\n[%d/%d] 开始测试: input=%d, output=%d, parallel=%d, "
                "num_prompts=%d (=%d×%d epochs), seed=%d%s\n%s",
```

并在日志参数中加入 `effective_seed`。

调用 `_extract_row()` 时传入：

```python
                               has_tokenizer=bool(our_args.tokenizer),
                               seed=effective_seed)
```

- [ ] **步骤 6：运行受影响测试验证通过**

运行：

```bash
pytest -q \
  vllm_standalone_bench/tests/test_config_seed.py \
  vllm_standalone_bench/tests/test_integration.py \
  vllm_standalone_bench/tests/test_extract_row.py
```

预期：全部 PASS。

- [ ] **步骤 7：Commit runner 接入**

```bash
git add \
  vllm_standalone_bench/run_bench_multi.py \
  vllm_standalone_bench/tests/test_extract_row.py \
  vllm_standalone_bench/tests/test_integration.py
git commit -m "feat: 按配置设置 vllm bench seed"
```

---

### 任务 4：更新 shell wrapper 并做完整验证

**文件：**
- 修改：`vllm_standalone_bench/run_bench.sh`
- 测试：`vllm_standalone_bench/tests`

- [ ] **步骤 1：在 `run_bench.sh` 添加 seed 配置段**

在 `WARMUP_REQUESTS=1` 后加入：

```bash
# =============================================================================
# ▌ 七、随机种子配置
# =============================================================================
#
# SEED 是随机种子基值。默认情况下，每个 (input, output, parallel) 配置会
# 基于该值派生独立 seed，避免高并发档位复用低并发档位的 prompt 序列。
#
# 如需复现旧行为（所有配置复用同一个 seed），设为 false。
SEED=0
VARY_SEED_BY_CONFIG=true
```

把后面的章节编号顺延一位，保证注释结构一致。

- [ ] **步骤 2：把 seed 参数加入 CMD**

在初始 `CMD=(...)` 中 `--sleep-between "${SLEEP_BETWEEN}"` 后加入：

```bash
    --seed "${SEED}"
```

在 `CMD+=(--warmup-requests "${WARMUP_REQUESTS}")` 前加入：

```bash
if [[ "${VARY_SEED_BY_CONFIG}" != "true" ]]; then
    CMD+=(--no-vary-seed-by-config)
fi
```

- [ ] **步骤 3：在摘要中打印 seed 策略**

在摘要打印 `配置间隔` 后加入：

```bash
printf "║  随机种子  : %-48s║\n" "${SEED}"
printf "║  配置派生  : %-48s║\n" "${VARY_SEED_BY_CONFIG}"
```

- [ ] **步骤 4：运行完整单元测试**

运行：

```bash
pytest -q vllm_standalone_bench/tests
```

预期：

```text
passed
```

当前基线是 25 个测试；实现后数量会增加，以 pytest 输出为准。

- [ ] **步骤 5：检查 CLI 解析**

运行：

```bash
python3 vllm_standalone_bench/run_bench_multi.py --help | grep -E -- '--seed|--no-vary-seed-by-config'
```

预期输出包含：

```text
--seed
--no-vary-seed-by-config
```

- [ ] **步骤 6：检查 shell wrapper 命令拼接**

不执行真实 benchmark，只检查脚本文本中包含必需参数：

```bash
grep -n "SEED=0\\|VARY_SEED_BY_CONFIG=true\\|--seed\\|--no-vary-seed-by-config" vllm_standalone_bench/run_bench.sh
```

预期输出至少包含四类匹配：

```text
SEED=0
VARY_SEED_BY_CONFIG=true
--seed
--no-vary-seed-by-config
```

- [ ] **步骤 7：Commit shell wrapper 更新**

```bash
git add vllm_standalone_bench/run_bench.sh
git commit -m "chore: 在 run_bench 暴露 seed 配置"
```

---

### 任务 5：最终验证和交付检查

**文件：**
- 检查：`docs/superpowers/specs/2026-06-29-vllm-bench-config-seed-design.md`
- 检查：`vllm_standalone_bench/run_bench_multi.py`
- 检查：`vllm_standalone_bench/run_bench.sh`
- 检查：`vllm_standalone_bench/tests`

- [ ] **步骤 1：运行完整测试**

运行：

```bash
pytest -q vllm_standalone_bench/tests
```

预期：全部 PASS。

- [ ] **步骤 2：确认 worktree 没有未提交实现文件**

运行：

```bash
git status --short
```

预期：无输出。

- [ ] **步骤 3：查看提交序列**

运行：

```bash
git log --oneline --decorate -5
```

预期：能看到 seed 规格提交、测试提交、实现提交、shell wrapper 提交。

- [ ] **步骤 4：汇总交付说明**

最终回复需要包含：

```text
- worktree 路径
- 分支名
- 关键行为：默认 per-config seed，兼容模式参数
- 验证命令和结果
- 未执行真实 vLLM 服务压测的说明
```
