# DeepSeek v4 flash Ascend 8 卡估算实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 基于 `资源规划工具.xlsx` 的 `671B-P800-8测试数据`，生成 `DeepSeek v4 flash` 在 `910B 8卡` 与 `910C 8卡` 下的公式化估算 Excel。

**架构：** 新增一个独立脚本负责读取 P800 基线、应用模型/硬件假设、生成估算行和输出工作簿。测试只针对新增脚本，不改现有 API，也不修改原始 Excel；`data/910B*` 只进入参考 sheet，不参与估算公式。

**技术栈：** Python 3.11、pandas、openpyxl、pytest。

---

## 文件结构

- 创建：`scripts/estimate_deepseek_v4_flash_ascend.py`
  - 职责：定义假设 dataclass、公式函数、Excel/CSV 读取、联网参考元数据、估算 DataFrame 生成、工作簿输出、CLI。
- 创建：`tests/test_deepseek_v4_flash_ascend_estimation.py`
  - 职责：覆盖公式、基线读取、910B 参考隔离、工作簿结构和 CLI 输出。
- 生成：`outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx`
  - 职责：最终估算产物；实现时由脚本生成，不手工编辑。

不修改：

- `资源规划工具.xlsx`
- `backend/app/services/prediction/*`
- `backend/app/api/*`

## 任务 1：建立估算核心公式

**文件：**
- 创建：`scripts/estimate_deepseek_v4_flash_ascend.py`
- 测试：`tests/test_deepseek_v4_flash_ascend_estimation.py`

- [ ] **步骤 1：编写失败的公式单测**

在 `tests/test_deepseek_v4_flash_ascend_estimation.py` 写入：

```python
import math

import pytest

from scripts.estimate_deepseek_v4_flash_ascend import (
    DEFAULT_GPU_ASSUMPTIONS,
    DEFAULT_MODEL_ASSUMPTION,
    DEFAULT_RUNTIME_SCENARIOS,
    BaselineRow,
    estimate_row,
    required_memory_per_card_gb,
)


def test_required_memory_per_card_changes_with_weight_precision():
    bf16 = required_memory_per_card_gb(
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=2.0),
        gpu_count=8,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        framework_overhead_gb=3.0,
    )
    int8 = required_memory_per_card_gb(
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=1.0),
        gpu_count=8,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        framework_overhead_gb=3.0,
    )
    assert bf16 > int8
    assert int8 > 3.0


def test_910c_compute_scale_uses_target_hardware_ratio():
    baseline = BaselineRow(
        excel_row=2,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        throughput_tokens_s=100.0,
        ttft_p90_ms=400.0,
        ttft_p99_ms=500.0,
        ttft_max_ms=550.0,
        ttft_mean_ms=300.0,
        decode_latency_p90_ms=40.0,
        decode_latency_p99_ms=45.0,
        decode_latency_max_ms=60.0,
        decode_latency_mean_ms=35.0,
    )
    result = estimate_row(
        baseline=baseline,
        target_gpu=DEFAULT_GPU_ASSUMPTIONS["910C"],
        base_gpu=DEFAULT_GPU_ASSUMPTIONS["P800"],
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=1.0),
        runtime=DEFAULT_RUNTIME_SCENARIOS["base"],
    )
    assert result["target_gpu"] == "910C"
    assert result["gpu_count"] == 8
    assert result["compute_scale"] == pytest.approx(780 / 280, rel=1e-6)
    assert result["decode_scale"] == pytest.approx(
        min(result["compute_scale"], result["bandwidth_scale"]),
        rel=1e-6,
    )
    assert result["输出tokens总吞吐"] == pytest.approx(100.0 * result["decode_scale"])
    assert 0.0 <= result["confidence"] <= 1.0
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py::test_required_memory_per_card_changes_with_weight_precision tests/test_deepseek_v4_flash_ascend_estimation.py::test_910c_compute_scale_uses_target_hardware_ratio -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'scripts.estimate_deepseek_v4_flash_ascend'`。

- [ ] **步骤 3：实现最小公式代码**

创建 `scripts/estimate_deepseek_v4_flash_ascend.py`，写入：

```python
"""Estimate DeepSeek v4 flash 8-card Ascend benchmark data from P800 baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_SHEET = "671B-P800-8测试数据"
OUTPUT_FILENAME = "deepseek_v4_flash_910b_910c_8card_estimate.xlsx"
GPU_COUNT = 8
MEMORY_UTILIZATION = 0.90
FRAMEWORK_OVERHEAD_GB = 3.0
ATTENTION_FACTOR = 2.0

SOURCE_COLUMNS = [
    "输入长度",
    "输出长度",
    "并发数",
    "输出tokens总吞吐",
    "首tokens时延TP90（ms）",
    "首tokens时延TP99（ms）",
    "最大首tokens时延（ms）",
    "平均首tokens时延（ms）",
    "增量时延TP90（ms）",
    "增量时延TP99（ms）",
    "最大增量时延（ms）",
    "平均增量时延（ms）",
]


@dataclass(frozen=True)
class GpuAssumption:
    name: str
    memory_gb: float
    bandwidth_gbs: float
    bf16_tflops: float
    spec_source: str


@dataclass(frozen=True)
class ModelAssumption:
    name: str
    total_params_b: float
    active_params_b: float
    layers: int
    hidden_size: int
    num_kv_heads: int
    head_size: int
    weight_bytes: float
    kv_bytes: float
    spec_source: str

    def with_updates(self, **kwargs: Any) -> "ModelAssumption":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RuntimeScenario:
    name: str
    base_efficiency: float
    target_efficiency: float
    label: str


@dataclass(frozen=True)
class BaselineRow:
    excel_row: int
    input_tokens: float
    output_tokens: float
    concurrency: float
    throughput_tokens_s: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    ttft_max_ms: float
    ttft_mean_ms: float
    decode_latency_p90_ms: float
    decode_latency_p99_ms: float
    decode_latency_max_ms: float
    decode_latency_mean_ms: float


DEFAULT_GPU_ASSUMPTIONS = {
    "P800": GpuAssumption("P800", 64.0, 2000.0, 280.0, "project_seed_data"),
    "910B": GpuAssumption("910B", 64.0, 2000.0, 280.0, "p800_equivalent_assumption"),
    "910C": GpuAssumption("910C", 128.0, 4000.0, 780.0, "public_cloudmatrix384_assumption"),
}

DEFAULT_MODEL_ASSUMPTION = ModelAssumption(
    name="DeepSeek v4 flash proxy",
    total_params_b=671.0,
    active_params_b=37.0,
    layers=61,
    hidden_size=7168,
    num_kv_heads=128,
    head_size=128,
    weight_bytes=1.0,
    kv_bytes=2.0,
    spec_source="DeepSeek-V3/R1 proxy",
)

DEFAULT_RUNTIME_SCENARIOS = {
    "conservative": RuntimeScenario("conservative", 0.70, 0.55, "保守"),
    "base": RuntimeScenario("base", 0.70, 0.70, "基准"),
    "optimistic": RuntimeScenario("optimistic", 0.70, 0.85, "乐观"),
}


def required_memory_per_card_gb(
    *,
    model: ModelAssumption,
    gpu_count: int,
    input_tokens: float,
    output_tokens: float,
    concurrency: float,
    framework_overhead_gb: float,
) -> float:
    weights_gb = model.total_params_b * 1e9 * model.weight_bytes / 1024**3
    kv_cache_gb = (
        2
        * model.layers
        * model.num_kv_heads
        * model.head_size
        * model.kv_bytes
        * (input_tokens + output_tokens)
        * concurrency
        / 1024**3
    )
    return weights_gb / gpu_count + kv_cache_gb / gpu_count + framework_overhead_gb


def decode_bytes_per_token(
    *,
    model: ModelAssumption,
    input_tokens: float,
    output_tokens: float,
) -> float:
    generated_tokens_so_far = output_tokens / 2
    weight_bytes = model.active_params_b * 1e9 * model.weight_bytes
    kv_bytes = (
        2
        * model.layers
        * model.num_kv_heads
        * model.head_size
        * model.kv_bytes
        * (input_tokens + generated_tokens_so_far)
    )
    return weight_bytes + kv_bytes


def prefill_flops(*, model: ModelAssumption, input_tokens: float) -> float:
    return (
        2 * model.active_params_b * 1e9 * input_tokens
        + ATTENTION_FACTOR * model.layers * input_tokens**2 * model.hidden_size
    )


def confidence_for(target_gpu: GpuAssumption) -> float:
    hardware_score = 0.70 if target_gpu.name == "910B" else 0.60
    return round(
        0.30 * 1.00
        + 0.25 * hardware_score
        + 0.20 * 0.65
        + 0.15 * 0.60
        + 0.10 * 0.90,
        2,
    )


def estimate_row(
    *,
    baseline: BaselineRow,
    target_gpu: GpuAssumption,
    base_gpu: GpuAssumption,
    model: ModelAssumption,
    runtime: RuntimeScenario,
) -> dict[str, Any]:
    target_model = model
    base_model = model
    required_gb = required_memory_per_card_gb(
        model=target_model,
        gpu_count=GPU_COUNT,
        input_tokens=baseline.input_tokens,
        output_tokens=baseline.output_tokens,
        concurrency=baseline.concurrency,
        framework_overhead_gb=FRAMEWORK_OVERHEAD_GB,
    )
    feasible = required_gb <= target_gpu.memory_gb * MEMORY_UTILIZATION

    compute_scale = (
        (target_gpu.bf16_tflops * runtime.target_efficiency)
        / (base_gpu.bf16_tflops * runtime.base_efficiency)
        * (base_model.active_params_b / target_model.active_params_b)
    )
    bandwidth_scale = (
        (target_gpu.bandwidth_gbs * runtime.target_efficiency)
        / (base_gpu.bandwidth_gbs * runtime.base_efficiency)
        * (
            decode_bytes_per_token(
                model=base_model,
                input_tokens=baseline.input_tokens,
                output_tokens=baseline.output_tokens,
            )
            / decode_bytes_per_token(
                model=target_model,
                input_tokens=baseline.input_tokens,
                output_tokens=baseline.output_tokens,
            )
        )
    )
    decode_scale = min(compute_scale, bandwidth_scale)
    prefill_compute_scale = compute_scale * (
        prefill_flops(model=base_model, input_tokens=baseline.input_tokens)
        / prefill_flops(model=target_model, input_tokens=baseline.input_tokens)
    )
    bottleneck = "compute" if compute_scale <= bandwidth_scale else "memory_bandwidth"
    if not feasible:
        bottleneck = "memory_infeasible"

    def scaled_decode(value: float) -> float | None:
        return round(value / decode_scale, 4) if feasible else None

    def scaled_ttft(value: float) -> float | None:
        return round(value / prefill_compute_scale, 4) if feasible else None

    return {
        "输入长度": baseline.input_tokens,
        "输出长度": baseline.output_tokens,
        "并发数": baseline.concurrency,
        "输出tokens总吞吐": round(baseline.throughput_tokens_s * decode_scale, 4) if feasible else None,
        "首tokens时延TP90（ms）": scaled_ttft(baseline.ttft_p90_ms),
        "首tokens时延TP99（ms）": scaled_ttft(baseline.ttft_p99_ms),
        "最大首tokens时延（ms）": scaled_ttft(baseline.ttft_max_ms),
        "平均首tokens时延（ms）": scaled_ttft(baseline.ttft_mean_ms),
        "增量时延TP90（ms）": scaled_decode(baseline.decode_latency_p90_ms),
        "增量时延TP99（ms）": scaled_decode(baseline.decode_latency_p99_ms),
        "最大增量时延（ms）": scaled_decode(baseline.decode_latency_max_ms),
        "平均增量时延（ms）": scaled_decode(baseline.decode_latency_mean_ms),
        "base_sheet": SOURCE_SHEET,
        "base_row": baseline.excel_row,
        "target_gpu": target_gpu.name,
        "gpu_count": GPU_COUNT,
        "runtime_efficiency": runtime.target_efficiency,
        "compute_scale": round(compute_scale, 6),
        "bandwidth_scale": round(bandwidth_scale, 6),
        "decode_scale": round(decode_scale, 6),
        "bottleneck": bottleneck,
        "required_per_card_gb": round(required_gb, 4),
        "confidence": confidence_for(target_gpu),
        "assumption_notes": f"{model.spec_source}; {target_gpu.spec_source}; runtime={runtime.label}",
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py::test_required_memory_per_card_changes_with_weight_precision tests/test_deepseek_v4_flash_ascend_estimation.py::test_910c_compute_scale_uses_target_hardware_ratio -v
```

预期：`2 passed`。

- [ ] **步骤 5：Commit**

运行：

```bash
git add scripts/estimate_deepseek_v4_flash_ascend.py tests/test_deepseek_v4_flash_ascend_estimation.py
git commit -m "feat: add deepseek ascend estimation formulas"
```

预期：commit 成功，提交只包含新增脚本和新增测试。

## 任务 2：读取 P800 基线并生成 910B/910C 估算表

**文件：**
- 修改：`scripts/estimate_deepseek_v4_flash_ascend.py`
- 测试：`tests/test_deepseek_v4_flash_ascend_estimation.py`

- [ ] **步骤 1：编写失败的基线读取和估算 DataFrame 测试**

追加到 `tests/test_deepseek_v4_flash_ascend_estimation.py`：

```python
from pathlib import Path

import pandas as pd

from scripts.estimate_deepseek_v4_flash_ascend import (
    SOURCE_COLUMNS,
    build_estimates,
    load_p800_baseline,
)


def write_baseline_workbook(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "输入长度": 512,
                "输出长度": 1024,
                "并发数": 5,
                "输出tokens总吞吐": 117.0,
                "首tokens时延TP90（ms）": 476.0,
                "首tokens时延TP99（ms）": 500.0,
                "最大首tokens时延（ms）": 501.0,
                "平均首tokens时延（ms）": 381.0,
                "增量时延TP90（ms）": 43.0,
                "增量时延TP99（ms）": 44.0,
                "最大增量时延（ms）": 46.0,
                "平均增量时延（ms）": 42.0,
            },
            {
                "输入长度": 1024,
                "输出长度": 1024,
                "并发数": 10,
                "输出tokens总吞吐": 150.0,
                "首tokens时延TP90（ms）": 600.0,
                "首tokens时延TP99（ms）": 650.0,
                "最大首tokens时延（ms）": 700.0,
                "平均首tokens时延（ms）": 500.0,
                "增量时延TP90（ms）": 55.0,
                "增量时延TP99（ms）": 58.0,
                "最大增量时延（ms）": 70.0,
                "平均增量时延（ms）": 50.0,
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="671B-P800-8测试数据", index=False)


def test_load_p800_baseline_preserves_excel_row_numbers(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    write_baseline_workbook(workbook)
    rows = load_p800_baseline(workbook)
    assert len(rows) == 2
    assert rows[0].excel_row == 2
    assert rows[1].excel_row == 3
    assert rows[0].throughput_tokens_s == 117.0


def test_build_estimates_creates_910b_and_910c_rows(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    write_baseline_workbook(workbook)
    baseline_rows = load_p800_baseline(workbook)
    estimates = build_estimates(baseline_rows)
    assert set(estimates["target_gpu"]) == {"910B", "910C"}
    assert len(estimates) == 4
    assert list(estimates.columns[: len(SOURCE_COLUMNS)]) == SOURCE_COLUMNS
    assert {"base_row", "compute_scale", "bandwidth_scale", "decode_scale", "confidence"}.issubset(estimates.columns)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py::test_load_p800_baseline_preserves_excel_row_numbers tests/test_deepseek_v4_flash_ascend_estimation.py::test_build_estimates_creates_910b_and_910c_rows -v
```

预期：FAIL，报错包含 `ImportError` 或 `AttributeError`，指出 `load_p800_baseline` / `build_estimates` 未定义。

- [ ] **步骤 3：实现基线读取和估算表生成**

追加到 `scripts/estimate_deepseek_v4_flash_ascend.py`：

```python
AUDIT_COLUMNS = [
    "base_sheet",
    "base_row",
    "target_gpu",
    "gpu_count",
    "runtime_efficiency",
    "compute_scale",
    "bandwidth_scale",
    "decode_scale",
    "bottleneck",
    "required_per_card_gb",
    "confidence",
    "assumption_notes",
]


def load_p800_baseline(workbook_path: Path) -> list[BaselineRow]:
    df = pd.read_excel(workbook_path, sheet_name=SOURCE_SHEET)
    missing = [col for col in SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required baseline columns: {missing}")
    rows: list[BaselineRow] = []
    for index, row in df.dropna(subset=["输入长度", "并发数"]).iterrows():
        rows.append(
            BaselineRow(
                excel_row=int(index) + 2,
                input_tokens=float(row["输入长度"]),
                output_tokens=float(row["输出长度"]),
                concurrency=float(row["并发数"]),
                throughput_tokens_s=float(row["输出tokens总吞吐"]),
                ttft_p90_ms=float(row["首tokens时延TP90（ms）"]),
                ttft_p99_ms=float(row["首tokens时延TP99（ms）"]),
                ttft_max_ms=float(row["最大首tokens时延（ms）"]),
                ttft_mean_ms=float(row["平均首tokens时延（ms）"]),
                decode_latency_p90_ms=float(row["增量时延TP90（ms）"]),
                decode_latency_p99_ms=float(row["增量时延TP99（ms）"]),
                decode_latency_max_ms=float(row["最大增量时延（ms）"]),
                decode_latency_mean_ms=float(row["平均增量时延（ms）"]),
            )
        )
    return rows


def build_estimates(
    baseline_rows: list[BaselineRow],
    *,
    model: ModelAssumption = DEFAULT_MODEL_ASSUMPTION,
    runtime: RuntimeScenario = DEFAULT_RUNTIME_SCENARIOS["base"],
) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []
    base_gpu = DEFAULT_GPU_ASSUMPTIONS["P800"]
    for baseline in baseline_rows:
        for gpu_name in ("910B", "910C"):
            output_rows.append(
                estimate_row(
                    baseline=baseline,
                    target_gpu=DEFAULT_GPU_ASSUMPTIONS[gpu_name],
                    base_gpu=base_gpu,
                    model=model,
                    runtime=runtime,
                )
            )
    return pd.DataFrame(output_rows, columns=SOURCE_COLUMNS + AUDIT_COLUMNS)
```

- [ ] **步骤 4：运行任务 1 和任务 2 测试**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py -v
```

预期：已写的测试全部 PASS。

- [ ] **步骤 5：Commit**

运行：

```bash
git add scripts/estimate_deepseek_v4_flash_ascend.py tests/test_deepseek_v4_flash_ascend_estimation.py
git commit -m "feat: build deepseek ascend estimate rows"
```

预期：commit 成功。

## 任务 3：生成 910B 低利用率参考统计但隔离估算公式

**文件：**
- 修改：`scripts/estimate_deepseek_v4_flash_ascend.py`
- 测试：`tests/test_deepseek_v4_flash_ascend_estimation.py`

- [ ] **步骤 1：编写失败的参考隔离测试**

追加到 `tests/test_deepseek_v4_flash_ascend_estimation.py`：

```python
from scripts.estimate_deepseek_v4_flash_ascend import summarize_910b_reference


def write_reference_csv(path: Path, throughput: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "输入长度,输出长度,并发数,输出tokens总吞吐,首tokens时延TP90（ms）,首tokens时延TP99（ms）,最大首tokens时延（ms）,平均首tokens时延（ms）,增量时延TP90（ms）,增量时延TP99（ms）,最大增量时延（ms）,平均增量时延（ms）",
                f"2048,1200,1,{throughput},320,330,340,310,30,35,55,29",
                f"2048,1200,4,{throughput * 2},1100,1110,1120,1000,32,38,58,30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_910b_reference_summary_does_not_change_estimates(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    write_baseline_workbook(workbook)
    baseline_rows = load_p800_baseline(workbook)
    before = build_estimates(baseline_rows)

    write_reference_csv(tmp_path / "data" / "910B1" / "72B-AWQ" / "8.csv", 10.0)
    first_summary = summarize_910b_reference(tmp_path / "data")
    write_reference_csv(tmp_path / "data" / "910B1" / "72B-AWQ" / "8.csv", 9999.0)
    second_summary = summarize_910b_reference(tmp_path / "data")
    after = build_estimates(baseline_rows)

    assert first_summary.loc[0, "mean_throughput_tokens_s"] != second_summary.loc[0, "mean_throughput_tokens_s"]
    pd.testing.assert_frame_equal(before, after)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py::test_910b_reference_summary_does_not_change_estimates -v
```

预期：FAIL，报错指出 `summarize_910b_reference` 未定义。

- [ ] **步骤 3：实现参考统计**

追加到 `scripts/estimate_deepseek_v4_flash_ascend.py`：

```python
REFERENCE_COLUMNS = [
    "source_file",
    "gpu_family",
    "model",
    "gpu_count",
    "rows",
    "mean_throughput_tokens_s",
    "max_throughput_tokens_s",
    "min_decode_latency_mean_ms",
    "mean_ttft_ms",
    "note",
]


def summarize_910b_reference(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(data_dir.glob("910B*/**/*.csv")):
        rel_parts = csv_path.relative_to(data_dir).parts
        if len(rel_parts) < 3:
            continue
        gpu_family, model_name, filename = rel_parts[0], rel_parts[1], rel_parts[2]
        gpu_count = Path(filename).stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rows.append(
            {
                "source_file": str(csv_path),
                "gpu_family": gpu_family,
                "model": model_name,
                "gpu_count": gpu_count,
                "rows": len(df),
                "mean_throughput_tokens_s": round(float(df["输出tokens总吞吐"].mean()), 4),
                "max_throughput_tokens_s": round(float(df["输出tokens总吞吐"].max()), 4),
                "min_decode_latency_mean_ms": round(float(df["平均增量时延（ms）"].min()), 4),
                "mean_ttft_ms": round(float(df["平均首tokens时延（ms）"].mean()), 4),
                "note": "低利用率参考，不参与估算校准",
            }
        )
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
```

- [ ] **步骤 4：运行全部新增测试**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py -v
```

预期：全部 PASS。

- [ ] **步骤 5：Commit**

运行：

```bash
git add scripts/estimate_deepseek_v4_flash_ascend.py tests/test_deepseek_v4_flash_ascend_estimation.py
git commit -m "feat: summarize low-utilization 910b references"
```

预期：commit 成功。

## 任务 4：输出 Excel 工作簿

**文件：**
- 修改：`scripts/estimate_deepseek_v4_flash_ascend.py`
- 测试：`tests/test_deepseek_v4_flash_ascend_estimation.py`

- [ ] **步骤 1：编写失败的工作簿结构测试**

追加到 `tests/test_deepseek_v4_flash_ascend_estimation.py`：

```python
from openpyxl import load_workbook

from scripts.estimate_deepseek_v4_flash_ascend import build_workbook


def test_build_workbook_writes_expected_sheets(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    output = tmp_path / "estimate.xlsx"
    write_baseline_workbook(workbook)
    write_reference_csv(tmp_path / "data" / "910B3" / "72B-AWQ" / "8.csv", 20.0)

    build_workbook(
        source_workbook=workbook,
        data_dir=tmp_path / "data",
        output_path=output,
    )

    loaded = load_workbook(output, data_only=False)
    assert loaded.sheetnames == [
        "00_估算说明",
        "01_假设表",
        "02_910B_8卡估算",
        "03_910C_8卡估算",
        "04_P800基线",
        "05_910B低利用率参考",
        "06_联网参考数据",
    ]
    assert loaded["00_估算说明"]["A1"].value == "DeepSeek v4 flash 910B/910C 8卡估算"
    assert loaded["02_910B_8卡估算"]["A1"].value == "输入长度"
    assert loaded["03_910C_8卡估算"]["A1"].value == "输入长度"
    assert loaded["05_910B低利用率参考"]["J2"].value == "低利用率参考，不参与估算校准"
    assert loaded["06_联网参考数据"]["A1"].value == "source_title"
    assert "CloudMatrix384" in loaded["06_联网参考数据"]["A4"].value
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py::test_build_workbook_writes_expected_sheets -v
```

预期：FAIL，报错指出 `build_workbook` 未定义。

- [ ] **步骤 3：实现工作簿输出**

追加到 `scripts/estimate_deepseek_v4_flash_ascend.py`：

```python
def assumptions_dataframe() -> pd.DataFrame:
    rows = []
    for gpu in DEFAULT_GPU_ASSUMPTIONS.values():
        rows.append(
            {
                "category": "gpu",
                "name": gpu.name,
                "memory_gb": gpu.memory_gb,
                "bandwidth_gbs": gpu.bandwidth_gbs,
                "bf16_tflops": gpu.bf16_tflops,
                "source": gpu.spec_source,
            }
        )
    for runtime in DEFAULT_RUNTIME_SCENARIOS.values():
        rows.append(
            {
                "category": "runtime",
                "name": runtime.name,
                "memory_gb": None,
                "bandwidth_gbs": None,
                "bf16_tflops": runtime.target_efficiency,
                "source": runtime.label,
            }
        )
    rows.append(
        {
            "category": "model",
            "name": DEFAULT_MODEL_ASSUMPTION.name,
            "memory_gb": DEFAULT_MODEL_ASSUMPTION.total_params_b,
            "bandwidth_gbs": DEFAULT_MODEL_ASSUMPTION.active_params_b,
            "bf16_tflops": DEFAULT_MODEL_ASSUMPTION.weight_bytes,
            "source": DEFAULT_MODEL_ASSUMPTION.spec_source,
        }
    )
    return pd.DataFrame(rows)


def explanation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["版本", "v1"],
            ["主基线", SOURCE_SHEET],
            ["估算口径", "P800实测基线 + 模型差异 + 硬件Roofline比例"],
            ["910B参考数据", "只进入参考sheet，不参与校准"],
            ["适用边界", "估算不能替代实测；实际部署需用910B/910C复测"],
        ],
        columns=["项目", "说明"],
    )


def public_reference_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_title": "A-IO: Adaptive Inference Orchestration for Memory-Bound NPUs",
                "url": "https://arxiv.org/abs/2604.09752",
                "hardware": "Ascend 910B",
                "model_or_scenario": "OpenPangu 1B/7B, HuggingFace + PyTorch",
                "metric_summary": "Reports OpenPangu 1B/7B TPS under 2K workloads and 32K accuracy on Ascend 910B.",
                "comparability_note": "Single-card small-model paper data; not used to calibrate DeepSeek 8-card MoE estimates.",
            },
            {
                "source_title": "An Empirical Study of OpenPangu Quantization on Ascend NPUs",
                "url": "https://arxiv.org/abs/2606.21257",
                "hardware": "Ascend 910B1",
                "model_or_scenario": "OpenPangu 1B/7B post-training quantization",
                "metric_summary": "Documents Ascend 910B1 64GB HBM environment and quantization behavior.",
                "comparability_note": "Quantization and accuracy reference; not a throughput calibration dataset.",
            },
            {
                "source_title": "Serving Large Language Models on Huawei CloudMatrix384",
                "url": "https://arxiv.org/abs/2506.12708",
                "hardware": "384 x Ascend 910C",
                "model_or_scenario": "DeepSeek-R1 on CloudMatrix-Infer",
                "metric_summary": "Reports 6688 prefill tokens/s/NPU, 1943 decode tokens/s/NPU, and 538 tokens/s/NPU at 15 ms TPOT.",
                "comparability_note": "384-card CloudMatrix-Infer result; upper-bound reference, not directly comparable to 8-card deployment.",
            },
            {
                "source_title": "Tom's Hardware CloudMatrix384 report",
                "url": "https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power",
                "hardware": "Ascend 910C",
                "model_or_scenario": "Public hardware/spec reporting",
                "metric_summary": "Reports 910C at 780 BF16 TFLOPS, 128GB HBM, 3.2TB/s HBM bandwidth, and CloudMatrix384 at 300 BF16 PFLOPS.",
                "comparability_note": "Non-vendor public report; used only as a configurable default assumption/reference.",
            },
        ]
    )


def build_workbook(*, source_workbook: Path, data_dir: Path, output_path: Path) -> Path:
    baseline_rows = load_p800_baseline(source_workbook)
    estimates = build_estimates(baseline_rows)
    baseline_df = pd.read_excel(source_workbook, sheet_name=SOURCE_SHEET)
    reference_df = summarize_910b_reference(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        explanation_dataframe().to_excel(writer, sheet_name="00_估算说明", index=False)
        assumptions_dataframe().to_excel(writer, sheet_name="01_假设表", index=False)
        estimates[estimates["target_gpu"] == "910B"].to_excel(writer, sheet_name="02_910B_8卡估算", index=False)
        estimates[estimates["target_gpu"] == "910C"].to_excel(writer, sheet_name="03_910C_8卡估算", index=False)
        baseline_df.to_excel(writer, sheet_name="04_P800基线", index=False)
        reference_df.to_excel(writer, sheet_name="05_910B低利用率参考", index=False)
        public_reference_dataframe().to_excel(writer, sheet_name="06_联网参考数据", index=False)

    return output_path
```

- [ ] **步骤 4：运行工作簿测试和全量新增测试**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py -v
```

预期：全部 PASS。

- [ ] **步骤 5：Commit**

运行：

```bash
git add scripts/estimate_deepseek_v4_flash_ascend.py tests/test_deepseek_v4_flash_ascend_estimation.py
git commit -m "feat: write deepseek ascend estimate workbook"
```

预期：commit 成功。

## 任务 5：添加 CLI 并生成真实估算文件

**文件：**
- 修改：`scripts/estimate_deepseek_v4_flash_ascend.py`
- 测试：`tests/test_deepseek_v4_flash_ascend_estimation.py`
- 生成：`outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx`

- [ ] **步骤 1：编写失败的 CLI 测试**

追加到 `tests/test_deepseek_v4_flash_ascend_estimation.py`：

```python
from scripts.estimate_deepseek_v4_flash_ascend import main


def test_main_generates_output_file(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    output = tmp_path / "estimate.xlsx"
    write_baseline_workbook(workbook)

    exit_code = main(
        [
            "--source-workbook",
            str(workbook),
            "--data-dir",
            str(tmp_path / "data"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()
```

- [ ] **步骤 2：运行 CLI 测试验证失败**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py::test_main_generates_output_file -v
```

预期：FAIL，报错指出 `main` 未定义。

- [ ] **步骤 3：实现 CLI**

追加到 `scripts/estimate_deepseek_v4_flash_ascend.py`：

```python
def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workbook", type=Path, default=Path("资源规划工具.xlsx"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs") / OUTPUT_FILENAME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_workbook(
        source_workbook=args.source_workbook,
        data_dir=args.data_dir,
        output_path=args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 4：运行新增测试**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py -v
```

预期：全部 PASS。

- [ ] **步骤 5：生成真实估算文件**

运行：

```bash
python scripts/estimate_deepseek_v4_flash_ascend.py
```

预期：stdout 输出 `outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx`，且该文件存在。

- [ ] **步骤 6：检查真实工作簿 sheet**

运行：

```bash
python -c "from openpyxl import load_workbook; wb=load_workbook('outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx', read_only=True); print(wb.sheetnames)"
```

预期：输出包含 `00_估算说明`、`02_910B_8卡估算`、`03_910C_8卡估算`、`05_910B低利用率参考`、`06_联网参考数据`。

- [ ] **步骤 7：Commit**

运行：

```bash
git add scripts/estimate_deepseek_v4_flash_ascend.py tests/test_deepseek_v4_flash_ascend_estimation.py outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx
git commit -m "feat: generate deepseek ascend estimate workbook"
```

预期：commit 成功。

## 任务 6：最终验证与审计

**文件：**
- 修改：无代码修改
- 检查：`scripts/estimate_deepseek_v4_flash_ascend.py`
- 检查：`tests/test_deepseek_v4_flash_ascend_estimation.py`
- 检查：`outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx`

- [ ] **步骤 1：运行新增测试文件**

运行：

```bash
pytest tests/test_deepseek_v4_flash_ascend_estimation.py -v
```

预期：全部 PASS。

- [ ] **步骤 2：运行现有报告测试，确认未回归**

运行：

```bash
pytest tests/test_inference_token_factory_report.py -v
```

预期：全部 PASS。

- [ ] **步骤 3：抽查估算文件关键列**

运行：

```bash
python -c "import pandas as pd; p='outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx'; df=pd.read_excel(p, sheet_name='02_910B_8卡估算'); print(df[['target_gpu','base_row','compute_scale','bandwidth_scale','decode_scale','confidence']].head(3).to_string(index=False))"
```

预期：输出 3 行，`target_gpu` 全为 `910B`，`base_row` 非空，比例列和 `confidence` 为数值。

- [ ] **步骤 4：确认 910B 参考隔离说明存在**

运行：

```bash
python -c "import pandas as pd; p='outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx'; df=pd.read_excel(p, sheet_name='05_910B低利用率参考'); print(sorted(df['note'].dropna().unique()))"
```

预期：输出包含 `低利用率参考，不参与估算校准`。

- [ ] **步骤 5：确认联网参考数据存在且不参与估算**

运行：

```bash
python -c "import pandas as pd; p='outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx'; ref=pd.read_excel(p, sheet_name='06_联网参考数据'); est=pd.read_excel(p, sheet_name='03_910C_8卡估算'); print(ref[['hardware','metric_summary']].to_string(index=False)); print(est[['target_gpu','compute_scale']].head(1).to_string(index=False))"
```

预期：联网参考 sheet 中包含 `Ascend 910B`、`384 x Ascend 910C` 等公开参考；估算 sheet 中没有引用联网参考列，只有公式审计列。

- [ ] **步骤 6：检查 git 状态**

运行：

```bash
git status --short
```

预期：只允许存在用户已有的未跟踪 `model/.lock/`、`model/Qwen/`、`model/down.py`，没有未提交的估算脚本、测试或输出文件。

- [ ] **步骤 7：提交最终状态说明**

如果步骤 1-5 全部符合预期，在最终回复中列出：

```text
验证：
- pytest tests/test_deepseek_v4_flash_ascend_estimation.py -v
- pytest tests/test_inference_token_factory_report.py -v
- 工作簿 sheet/关键列抽查命令

产物：
- outputs/deepseek_v4_flash_910b_910c_8card_estimate.xlsx
```
