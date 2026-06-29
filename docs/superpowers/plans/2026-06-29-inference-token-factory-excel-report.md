# 推理 Token 工厂 Excel 汇报初版 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 生成 `inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx`，作为面向领导汇报的多 Sheet 推理 Token 工厂初版工作簿。

**架构：** 新增独立生成脚本 `scripts/build_inference_token_factory_report.py`，读取已有 `outputs/context_analysis_20260609_034248/` 和 `data/H200/` 数据，使用 `openpyxl` 生成 13 个 Sheet、原理简图、图表和数据附录。测试放在 `tests/test_inference_token_factory_report.py`，通过轻量夹具和真实数据路径验证工作簿结构、关键结论、图表和原理简图。

**技术栈：** Python 3.10+，`openpyxl==3.1.5`，`csv`/`json` 标准库，`pytest`。执行生成和最终检查阶段需要使用 `xlsx` 技能辅助验证工作簿内容。

**规格：** `docs/superpowers/specs/2026-06-29-inference-token-factory-excel-report-design.md`

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `scripts/build_inference_token_factory_report.py` | 读取上下文与 H200 测试数据，计算摘要，生成 Excel 工作簿 |
| `tests/test_inference_token_factory_report.py` | 测试 Sheet 结构、状态标签、数据计算、原理简图、图表和 CLI 输出 |
| `inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx` | 生成的汇报工作簿初版 |

设计边界：

- 报告生成逻辑不进入 `backend/`，避免影响业务服务。
- 原理简图使用 Excel 单元格网格、边框、填充色和箭头字符实现；这是“Excel 原生形状或等价图形”的稳定实现方式，便于测试和跨环境打开。
- 图表使用 `openpyxl.chart` 的 `BarChart` 和 `LineChart`，底表全部集中在 `12_数据附录`。
- 缺系统数据的技术专题只写验证方案、资源诉求和指标，不生成虚假数值。

## 任务 1：测试固定 Sheet 结构与状态标签

**文件：**
- 创建：`tests/test_inference_token_factory_report.py`
- 创建：`scripts/build_inference_token_factory_report.py`

- [ ] **步骤 1：编写失败的结构测试**

创建 `tests/test_inference_token_factory_report.py`，先写最小结构测试：

```python
from pathlib import Path

import pytest

from scripts.build_inference_token_factory_report import (
    REPORT_VERSION,
    SHEET_SPECS,
    STATUS_LABELS,
    build_workbook,
)


EXPECTED_SHEETS = [
    "00_目录与版本",
    "01_背景与目标",
    "02_总览结论",
    "03_技术路线地图",
    "04_模型量化",
    "05_KVCache量化",
    "06_Prefix_KV命中",
    "07_投机解码",
    "08_PD分离",
    "09_MOE专家并行",
    "10_连续批处理",
    "11_汇总建议与资源计划",
    "12_数据附录",
]


def test_static_sheet_specs_and_status_labels():
    assert REPORT_VERSION == "v2.0"
    assert [item["title"] for item in SHEET_SPECS] == EXPECTED_SHEETS
    assert STATUS_LABELS == [
        "已验证",
        "有结论但缺系统数据",
        "待验证",
        "已具备无需独立测试",
    ]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_static_sheet_specs_and_status_labels -q`

预期：导入失败，报错包含 `No module named 'scripts.build_inference_token_factory_report'`。

- [ ] **步骤 3：编写最少脚本骨架**

创建 `scripts/build_inference_token_factory_report.py`：

```python
"""Generate the inference token factory leadership report workbook."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook


REPORT_VERSION = "v2.0"
OUTPUT_FILENAME = "推理token工厂汇报大纲_汇报版_v2.0.xlsx"

STATUS_LABELS = [
    "已验证",
    "有结论但缺系统数据",
    "待验证",
    "已具备无需独立测试",
]

SHEET_SPECS = [
    {"title": "00_目录与版本", "owner": "孙力光", "status": "已验证"},
    {"title": "01_背景与目标", "owner": "孙力光", "status": "已验证"},
    {"title": "02_总览结论", "owner": "孙力光", "status": "已验证"},
    {"title": "03_技术路线地图", "owner": "孙力光", "status": "已验证"},
    {"title": "04_模型量化", "owner": "孙力光", "status": "已验证"},
    {"title": "05_KVCache量化", "owner": "胡正升", "status": "有结论但缺系统数据"},
    {"title": "06_Prefix_KV命中", "owner": "胡正升", "status": "有结论但缺系统数据"},
    {"title": "07_投机解码", "owner": "胡正升", "status": "待验证"},
    {"title": "08_PD分离", "owner": "孙力光", "status": "待验证"},
    {"title": "09_MOE专家并行", "owner": "孙力光", "status": "待验证"},
    {"title": "10_连续批处理", "owner": "孙力光", "status": "已具备无需独立测试"},
    {"title": "11_汇总建议与资源计划", "owner": "孙力光", "status": "待验证"},
    {"title": "12_数据附录", "owner": "孙力光", "status": "已验证"},
]


def build_workbook(repo_root: Path) -> Workbook:
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    for spec in SHEET_SPECS:
        wb.create_sheet(spec["title"])
    return wb
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_static_sheet_specs_and_status_labels -q`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_inference_token_factory_report.py tests/test_inference_token_factory_report.py
git commit -m "test: 锁定推理汇报工作簿结构"
```

## 任务 2：读取上下文背景数据与 H200 量化数据

**文件：**
- 修改：`scripts/build_inference_token_factory_report.py`
- 修改：`tests/test_inference_token_factory_report.py`

- [ ] **步骤 1：编写失败的数据计算测试**

追加到 `tests/test_inference_token_factory_report.py`：

```python
from scripts.build_inference_token_factory_report import (
    compute_fp8_summary,
    load_context_summary,
)


def test_load_context_summary_from_existing_outputs():
    summary = load_context_summary(Path("."))
    assert summary["total_requests"] == 1_008_098
    assert summary["total_tokens_billion"] == pytest.approx(59.36, rel=0.01)
    assert summary["input_token_ratio"] == pytest.approx(0.989, rel=0.01)
    assert summary["long_context_ratio"] == pytest.approx(0.623, rel=0.02)


def test_compute_h200_fp8_summary_from_existing_csvs():
    summary = compute_fp8_summary(Path("."))
    assert summary["72B"]["matched_rows"] >= 30
    assert summary["72B"]["avg_throughput_gain_pct"] == pytest.approx(32.4, rel=0.08)
    assert summary["72B"]["avg_tpot_ratio_pct"] == pytest.approx(77.5, rel=0.08)
    assert summary["32B"]["matched_rows"] >= 30
    assert summary["32B"]["avg_throughput_gain_pct"] == pytest.approx(30.3, rel=0.08)
    assert summary["32B"]["avg_tpot_ratio_pct"] == pytest.approx(77.8, rel=0.08)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_load_context_summary_from_existing_outputs tests/test_inference_token_factory_report.py::test_compute_h200_fp8_summary_from_existing_csvs -q`

预期：导入失败，报错包含 `cannot import name 'load_context_summary'`。

- [ ] **步骤 3：实现数据读取与计算函数**

在 `scripts/build_inference_token_factory_report.py` 中补充：

```python
import csv
import json
from statistics import mean


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_context_summary(repo_root: Path) -> dict[str, float | int]:
    out_dir = repo_root / "outputs" / "context_analysis_20260609_034248"
    overview = json.loads((out_dir / "01_overview.json").read_text(encoding="utf-8"))
    input_rows = _read_csv_rows(out_dir / "02_input_buckets.csv")
    total_requests = int(overview["total_requests"])
    total_tokens = float(overview["total_tokens"])
    input_tokens = float(overview["total_input_tokens"])
    long_requests = sum(
        int(row["request_count"])
        for row in input_rows
        if row["range_label"] in {"32K-64K", "64K-128K", "128K+"}
    )
    return {
        "total_requests": total_requests,
        "total_tokens_billion": total_tokens / 1_000_000_000,
        "input_token_ratio": input_tokens / total_tokens,
        "long_context_ratio": long_requests / total_requests,
    }


def _load_h200_csv(path: Path) -> dict[tuple[float, float, int], dict[str, float]]:
    rows = {}
    for row in _read_csv_rows(path):
        key = (
            float(row["输入长度"]),
            float(row["输出长度"]),
            int(float(row["并发数"])),
        )
        rows[key] = {
            "throughput": float(row["输出tokens总吞吐"]),
            "avg_tpot_ms": float(row["平均增量时延（ms）"]),
            "avg_ttft_ms": float(row["平均首tokens时延（ms）"]),
        }
    return rows


def _compare_h200_pair(base_path: Path, fp8_path: Path) -> dict[str, float | int]:
    base = _load_h200_csv(base_path)
    fp8 = _load_h200_csv(fp8_path)
    keys = sorted(set(base) & set(fp8))
    throughput_ratios = [fp8[k]["throughput"] / base[k]["throughput"] for k in keys]
    tpot_ratios = [fp8[k]["avg_tpot_ms"] / base[k]["avg_tpot_ms"] for k in keys]
    return {
        "matched_rows": len(keys),
        "avg_throughput_gain_pct": (mean(throughput_ratios) - 1.0) * 100.0,
        "avg_tpot_ratio_pct": mean(tpot_ratios) * 100.0,
    }


def compute_fp8_summary(repo_root: Path) -> dict[str, dict[str, float | int]]:
    h200 = repo_root / "data" / "H200"
    return {
        "32B": _compare_h200_pair(h200 / "32B" / "1.csv", h200 / "32B-FP8" / "1.csv"),
        "72B": _compare_h200_pair(h200 / "72B" / "2.csv", h200 / "72B-FP8" / "2.csv"),
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_load_context_summary_from_existing_outputs tests/test_inference_token_factory_report.py::test_compute_h200_fp8_summary_from_existing_csvs -q`

预期：`2 passed`。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_inference_token_factory_report.py tests/test_inference_token_factory_report.py
git commit -m "feat: 汇总上下文与 H200 FP8 数据"
```

## 任务 3：生成 13 个 Sheet 的正文骨架

**文件：**
- 修改：`scripts/build_inference_token_factory_report.py`
- 修改：`tests/test_inference_token_factory_report.py`

- [ ] **步骤 1：编写失败的工作簿内容测试**

追加测试：

```python
from openpyxl import load_workbook


def test_workbook_contains_required_sheet_headers(tmp_path):
    output = tmp_path / "report.xlsx"
    wb = build_workbook(Path("."))
    wb.save(output)
    loaded = load_workbook(output)
    assert loaded.sheetnames == EXPECTED_SHEETS
    for sheet_name in EXPECTED_SHEETS:
        ws = loaded[sheet_name]
        assert ws["A1"].value == "推理 Token 工厂汇报"
        assert ws["B2"].value == REPORT_VERSION
        assert ws["A4"].value == "当前状态"


def test_summary_sheet_has_expected_technology_states(tmp_path):
    output = tmp_path / "report.xlsx"
    wb = build_workbook(Path("."))
    wb.save(output)
    ws = load_workbook(output)["02_总览结论"]
    values = [cell.value for row in ws.iter_rows() for cell in row if cell.value]
    assert "模型量化" in values
    assert "MOE 专家并行" in values
    assert "待验证" in values
    assert "连续批处理" in values
    assert "已具备无需独立测试" in values
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_workbook_contains_required_sheet_headers tests/test_inference_token_factory_report.py::test_summary_sheet_has_expected_technology_states -q`

预期：失败，第一项报 `None != '推理 Token 工厂汇报'`。

- [ ] **步骤 3：实现通用标题、目录和总览表**

在脚本中添加样式和写入函数：

```python
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SUB_FILL = PatternFill("solid", fgColor="D9EAF7")
NOTE_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN_BORDER = Border(
    left=Side(style="thin", color="D9E2F3"),
    right=Side(style="thin", color="D9E2F3"),
    top=Side(style="thin", color="D9E2F3"),
    bottom=Side(style="thin", color="D9E2F3"),
)


TECH_ROWS = [
    ("模型量化", "已验证", "H200 FP8/BF16 已有数据，重点展示吞吐和 TPOT 收益"),
    ("KV Cache 量化", "有结论但缺系统数据", "fp8 可节省 KV 显存，fp4 因 H200 不支持暂不测"),
    ("Prefix/KV 命中", "有结论但缺系统数据", "需补测 prefix_ratio、命中率和 TTFT 下降比例"),
    ("投机解码", "待验证", "候选 token 数 1/3/5，关注 TPOT、接受率和高并发衰减"),
    ("PD 分离", "待验证", "覆盖 H200 P+D、H200 P/H200 D、H200 P/H20 D"),
    ("MOE 专家并行", "待验证", "按 GLM5.1/GLM5.2 FP8 部署口径申请资源"),
    ("连续批处理", "已具备无需独立测试", "框架原生支持，作为能力现状说明"),
]


def _style_range(ws, min_row: int, max_row: int, min_col: int, max_col: int) -> None:
    for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
        for cell in row:
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical="center", wrap_text=True)


def _write_header(ws, spec: dict[str, str]) -> None:
    ws["A1"] = "推理 Token 工厂汇报"
    ws["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    ws["A1"].fill = HEADER_FILL
    ws.merge_cells("A1:H1")
    fields = [
        ("报告版本", REPORT_VERSION),
        ("责任人", spec["owner"]),
        ("当前状态", spec["status"]),
        ("数据来源", "见 12_数据附录"),
    ]
    for idx, (label, value) in enumerate(fields, start=2):
        ws.cell(row=idx, column=1, value=label)
        ws.cell(row=idx, column=2, value=value)
    ws["A4"] = "当前状态"
    ws.freeze_panes = "A6"
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 22
    for col in "CDEFGH":
        ws.column_dimensions[col].width = 18


def _write_sheet_index(ws) -> None:
    ws.append([])
    ws.append(["Sheet", "名称", "责任人", "状态", "说明"])
    for index, spec in enumerate(SHEET_SPECS):
        ws.append([index, spec["title"], spec["owner"], spec["status"], "按规格生成"])
    _style_range(ws, 6, 6 + len(SHEET_SPECS), 1, 5)


def _write_summary(ws) -> None:
    ws.append([])
    ws.append(["技术点", "状态", "汇报口径"])
    for row in TECH_ROWS:
        ws.append(list(row))
    _style_range(ws, 6, 6 + len(TECH_ROWS), 1, 3)
```

修改 `build_workbook`：

```python
def build_workbook(repo_root: Path) -> Workbook:
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    for spec in SHEET_SPECS:
        ws = wb.create_sheet(spec["title"])
        _write_header(ws, spec)
        if spec["title"] == "00_目录与版本":
            _write_sheet_index(ws)
        if spec["title"] == "02_总览结论":
            _write_summary(ws)
    return wb
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_workbook_contains_required_sheet_headers tests/test_inference_token_factory_report.py::test_summary_sheet_has_expected_technology_states -q`

预期：`2 passed`。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_inference_token_factory_report.py tests/test_inference_token_factory_report.py
git commit -m "feat: 生成汇报工作簿基础结构"
```

## 任务 4：为技术专题写入原理简图

**文件：**
- 修改：`scripts/build_inference_token_factory_report.py`
- 修改：`tests/test_inference_token_factory_report.py`

- [ ] **步骤 1：编写失败的原理简图测试**

追加测试：

```python
PRINCIPLE_EXPECTATIONS = {
    "04_模型量化": ["BF16/FP16", "FP8", "吞吐提升"],
    "05_KVCache量化": ["KV Cache Block", "fp8 KV", "fp4 H200 不支持"],
    "06_Prefix_KV命中": ["共享前缀", "Prefix Hash", "复用 KV"],
    "07_投机解码": ["Draft", "Verify", "Accept/Reject"],
    "08_PD分离": ["Prefill", "KV 传输", "Decode", "H200 P + H20 D"],
    "09_MOE专家并行": ["Router", "Experts", "All2All", "GLM5.1/GLM5.2 FP8"],
    "10_连续批处理": ["Scheduler", "Prefill", "Decode", "框架原生支持"],
}


def test_each_technology_sheet_has_principle_diagram(tmp_path):
    output = tmp_path / "report.xlsx"
    wb = build_workbook(Path("."))
    wb.save(output)
    loaded = load_workbook(output)
    for sheet_name, expected_labels in PRINCIPLE_EXPECTATIONS.items():
        ws = loaded[sheet_name]
        values = [cell.value for row in ws.iter_rows(min_row=6, max_row=18) for cell in row if cell.value]
        text = " ".join(str(v) for v in values)
        assert "V2.2 方案：原理简图" in text
        for label in expected_labels:
            assert label in text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_each_technology_sheet_has_principle_diagram -q`

预期：失败，报 `V2.2 方案：原理简图` 不存在。

- [ ] **步骤 3：实现单元格原理简图**

在脚本中增加：

```python
PRINCIPLE_DIAGRAMS = {
    "04_模型量化": ["BF16/FP16 权重与激活", "FP8/量化模型", "显存下降", "吞吐提升"],
    "05_KVCache量化": ["KV Cache Block", "fp16/bf16 KV", "fp8 KV", "fp4 H200 不支持"],
    "06_Prefix_KV命中": ["共享前缀", "Prefix Hash", "复用 KV", "TTFT 降低"],
    "07_投机解码": ["Draft 候选 token", "Verify", "Accept/Reject", "TPOT 降低"],
    "08_PD分离": ["Prefill(H200)", "KV 传输", "Decode(H200/H20)", "H200 P + H20 D"],
    "09_MOE专家并行": ["Router", "多 GPU Experts", "All2All 聚合", "GLM5.1/GLM5.2 FP8"],
    "10_连续批处理": ["Scheduler", "新请求 Prefill", "旧请求 Decode", "框架原生支持"],
}


def _write_principle_diagram(ws, labels: list[str]) -> None:
    ws["A6"] = "V2.2 方案：原理简图"
    ws["A6"].font = Font(bold=True, color="1F4E78")
    start_row = 8
    start_col = 1
    for idx, label in enumerate(labels):
        col = start_col + idx * 2
        ws.cell(start_row, col, label)
        ws.cell(start_row, col).fill = SUB_FILL
        ws.cell(start_row, col).font = Font(bold=True)
        ws.cell(start_row, col).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.cell(start_row, col).border = THIN_BORDER
        ws.column_dimensions[ws.cell(start_row, col).column_letter].width = 20
        if idx < len(labels) - 1:
            ws.cell(start_row, col + 1, "→")
            ws.cell(start_row, col + 1).alignment = Alignment(horizontal="center", vertical="center")
    ws["A10"] = "说明"
    ws["B10"] = "该图用于快速解释技术机制，详细参数与测试设计见下方表格。"
```

修改 `build_workbook` 循环，在创建每个技术专题 Sheet 后调用：

```python
        if spec["title"] in PRINCIPLE_DIAGRAMS:
            _write_principle_diagram(ws, PRINCIPLE_DIAGRAMS[spec["title"]])
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_each_technology_sheet_has_principle_diagram -q`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_inference_token_factory_report.py tests/test_inference_token_factory_report.py
git commit -m "feat: 增加专题原理简图"
```

## 任务 5：写入背景、模型量化、资源计划和数据附录内容

**文件：**
- 修改：`scripts/build_inference_token_factory_report.py`
- 修改：`tests/test_inference_token_factory_report.py`

- [ ] **步骤 1：编写失败的内容测试**

追加测试：

```python
def test_report_contains_key_business_and_quantization_conclusions(tmp_path):
    output = tmp_path / "report.xlsx"
    wb = build_workbook(Path("."))
    wb.save(output)
    loaded = load_workbook(output, data_only=False)
    background_text = " ".join(
        str(cell.value)
        for row in loaded["01_背景与目标"].iter_rows()
        for cell in row
        if cell.value
    )
    quant_text = " ".join(
        str(cell.value)
        for row in loaded["04_模型量化"].iter_rows()
        for cell in row
        if cell.value
    )
    pd_text = " ".join(
        str(cell.value)
        for row in loaded["08_PD分离"].iter_rows()
        for cell in row
        if cell.value
    )
    appendix_text = " ".join(
        str(cell.value)
        for row in loaded["12_数据附录"].iter_rows()
        for cell in row
        if cell.value
    )
    assert "100.8 万" in background_text
    assert "593.6 亿" in background_text
    assert "32K 以上" in background_text
    assert "72B" in quant_text and "吞吐约提升" in quant_text
    assert "H200 做 P、H20 做 D" in pd_text
    assert "data/H200/72B-FP8/2.csv" in appendix_text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_report_contains_key_business_and_quantization_conclusions -q`

预期：失败，报关键文本不存在。

- [ ] **步骤 3：实现正文写入函数**

在脚本中新增：

```python
def _append_table(ws, start_row: int, headers: list[str], rows: list[list[object]]) -> int:
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(start_row, col, header)
        cell.fill = SUB_FILL
        cell.font = Font(bold=True)
    for row_idx, row in enumerate(rows, start=start_row + 1):
        for col_idx, value in enumerate(row, start=1):
            ws.cell(row_idx, col_idx, value)
    _style_range(ws, start_row, start_row + len(rows), 1, len(headers))
    return start_row + len(rows) + 2


def _write_background(ws, context: dict[str, float | int]) -> None:
    ws["A6"] = "背景摘要"
    ws["A7"] = f"最近 90 天约 {context['total_requests'] / 10000:.1f} 万请求，总 token 约 {context['total_tokens_billion'] * 10:.1f} 亿。"
    ws["A8"] = f"输入 token 占比约 {context['input_token_ratio']:.1%}，说明成本和时延压力主要来自长输入。"
    ws["A9"] = f"32K 以上长上下文请求占比约 {context['long_context_ratio']:.1%}，推理优化需重点关注 prefill、KV Cache 和长上下文并发。"


def _write_model_quantization(ws, fp8_summary: dict[str, dict[str, float | int]]) -> None:
    row = 13
    rows = []
    for model in ["32B", "72B"]:
        item = fp8_summary[model]
        rows.append([
            model,
            int(item["matched_rows"]),
            round(float(item["avg_throughput_gain_pct"]), 1),
            round(float(item["avg_tpot_ratio_pct"]), 1),
            "H200 FP8 vs BF16",
        ])
    _append_table(ws, row, ["模型", "匹配行数", "吞吐约提升", "TPOT 为 BF16 比例", "数据口径"], rows)


def _write_pd_plan(ws) -> None:
    row = 13
    rows = [
        ["基线", "H200 单体 P+D", "对照收益", "TTFT/TPOT/吞吐/GPU 利用率"],
        ["同构 PD", "H200 做 P、H200 做 D", "验证收益上限", "KV 传输延迟/网络带宽"],
        ["异构 PD", "H200 做 P、H20 做 D", "验证成本更低 Decode 组合", "H20 Decode 稳定性/显存"],
    ]
    _append_table(ws, row, ["类型", "资源组合", "验证目的", "监控指标"], rows)


def _write_appendix(ws) -> None:
    rows = [
        ["生产上下文概览", "outputs/context_analysis_20260609_034248/01_overview.json"],
        ["输入长度分布", "outputs/context_analysis_20260609_034248/02_input_buckets.csv"],
        ["H200 32B BF16", "data/H200/32B/1.csv"],
        ["H200 32B FP8", "data/H200/32B-FP8/1.csv"],
        ["H200 72B BF16", "data/H200/72B/2.csv"],
        ["H200 72B FP8", "data/H200/72B-FP8/2.csv"],
        ["会议资料目录", "inference-report/"],
    ]
    _append_table(ws, 6, ["数据项", "路径"], rows)
```

修改 `build_workbook`：先计算 `context = load_context_summary(repo_root)`、`fp8_summary = compute_fp8_summary(repo_root)`，然后按 Sheet 调用正文写入函数。

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_report_contains_key_business_and_quantization_conclusions -q`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_inference_token_factory_report.py tests/test_inference_token_factory_report.py
git commit -m "feat: 写入汇报正文与数据附录"
```

## 任务 6：增加 Excel 图表

**文件：**
- 修改：`scripts/build_inference_token_factory_report.py`
- 修改：`tests/test_inference_token_factory_report.py`

- [ ] **步骤 1：编写失败的图表测试**

追加测试：

```python
def test_workbook_contains_required_charts(tmp_path):
    output = tmp_path / "report.xlsx"
    wb = build_workbook(Path("."))
    wb.save(output)
    loaded = load_workbook(output)
    assert len(loaded["01_背景与目标"]._charts) >= 1
    assert len(loaded["04_模型量化"]._charts) >= 1
    assert len(loaded["12_数据附录"]._charts) == 0
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_workbook_contains_required_charts -q`

预期：失败，报 `0 >= 1`。

- [ ] **步骤 3：实现背景和量化图表**

在脚本中新增：

```python
from openpyxl.chart import BarChart, LineChart, Reference


def _add_context_chart(ws) -> None:
    row = 12
    ws.cell(row, 1, "区间")
    ws.cell(row, 2, "占比")
    buckets = [
        ["0-8K", 0.1482],
        ["8K-32K", 0.2291],
        ["32K-64K", 0.2346],
        ["64K-128K", 0.2933],
        ["128K+", 0.0948],
    ]
    for offset, item in enumerate(buckets, start=1):
        ws.cell(row + offset, 1, item[0])
        ws.cell(row + offset, 2, item[1])
    chart = BarChart()
    chart.title = "输入 Token 区间占比"
    chart.y_axis.title = "占比"
    chart.x_axis.title = "区间"
    chart.add_data(Reference(ws, min_col=2, min_row=row, max_row=row + len(buckets)), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=row + 1, max_row=row + len(buckets)))
    ws.add_chart(chart, "D12")


def _add_quant_chart(ws, start_row: int) -> None:
    chart = BarChart()
    chart.title = "H200 FP8 吞吐提升"
    chart.y_axis.title = "提升比例"
    chart.x_axis.title = "模型"
    chart.add_data(Reference(ws, min_col=3, min_row=start_row, max_row=start_row + 2), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=start_row + 1, max_row=start_row + 2))
    ws.add_chart(chart, "G13")
```

在 `_write_background` 末尾调用 `_add_context_chart(ws)`。在 `_write_model_quantization` 中确保吞吐提升列写入数字百分比而非字符串，并调用 `_add_quant_chart(ws, row)`。

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_workbook_contains_required_charts -q`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_inference_token_factory_report.py tests/test_inference_token_factory_report.py
git commit -m "feat: 增加汇报工作簿图表"
```

## 任务 7：增加 CLI 输出并生成工作簿初版

**文件：**
- 修改：`scripts/build_inference_token_factory_report.py`
- 修改：`tests/test_inference_token_factory_report.py`
- 创建：`inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx`

- [ ] **步骤 1：编写失败的 CLI 测试**

追加测试：

```python
import subprocess
import sys


def test_cli_writes_report_to_requested_path(tmp_path):
    output = tmp_path / "report.xlsx"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_inference_token_factory_report.py",
            "--repo-root",
            ".",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.exists()
    assert "Wrote" in result.stdout
    loaded = load_workbook(output)
    assert loaded.sheetnames == EXPECTED_SHEETS
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_cli_writes_report_to_requested_path -q`

预期：失败，脚本没有 CLI 参数解析或没有写入文件。

- [ ] **步骤 3：实现 CLI**

在脚本末尾添加：

```python
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument(
        "--output",
        default=str(Path("inference-report") / OUTPUT_FILENAME),
        help="Output xlsx path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wb = build_workbook(repo_root)
    wb.save(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行 CLI 测试验证通过**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py::test_cli_writes_report_to_requested_path -q`

预期：`1 passed`。

- [ ] **步骤 5：生成实际报告文件**

运行：

```bash
python3 scripts/build_inference_token_factory_report.py \
  --repo-root . \
  --output inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx
```

预期输出包含：`Wrote inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx`。

- [ ] **步骤 6：Commit**

```bash
git add scripts/build_inference_token_factory_report.py tests/test_inference_token_factory_report.py inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx
git commit -m "feat: 生成推理 token 工厂汇报初版"
```

## 任务 8：最终验证与 xlsx 检查

**文件：**
- 修改：无，除非验证发现问题

- [ ] **步骤 1：运行单元测试**

运行：`python3 -m pytest tests/test_inference_token_factory_report.py -q`

预期：所有测试通过，输出形如 `8 passed`。

- [ ] **步骤 2：运行工作簿结构检查**

运行：

```bash
python3 - <<'PY'
from openpyxl import load_workbook
p = "inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx"
wb = load_workbook(p)
print(len(wb.sheetnames), wb.sheetnames)
print("charts", len(wb["01_背景与目标"]._charts), len(wb["04_模型量化"]._charts))
print("principle", wb["08_PD分离"]["A6"].value, wb["09_MOE专家并行"]["A6"].value)
PY
```

预期：

- 第一行以 `13 [` 开头。
- 第二行显示背景和模型量化 Sheet 都至少有 1 个 chart。
- 第三行包含两处 `V2.2 方案：原理简图`。

- [ ] **步骤 3：使用 xlsx 技能检查输出文件**

在当前会话中加载 `xlsx` 技能，检查：

- 工作簿能正常打开。
- 13 个 Sheet 名称正确。
- 技术专题 Sheet 顶部存在原理简图。
- 图表存在于背景和模型量化 Sheet。
- 数据附录包含实际路径。

- [ ] **步骤 4：最终 git 状态检查**

运行：`git status --short`

预期：只存在用户已有的无关未跟踪/修改文件；本计划相关文件没有未提交变化。

- [ ] **步骤 5：汇报验证结果**

最终回复需要列出：

- 输出文件路径。
- 关键 Sheet 数和图表/原理简图验证证据。
- 测试命令和结果。
- 若 `xlsx` 检查发现问题，列出修复或剩余风险。

## 自检清单

- 规格覆盖：13 个 Sheet、版本号、状态标签、已有数据可视化、原理简图、PD 异构方案、MOE GLM5.1/GLM5.2 FP8 资源口径、数据附录均有任务覆盖。
- 空洞内容扫描：计划中没有未定义动作、含糊步骤或跨任务省略说明。
- 类型一致性：核心入口固定为 `build_workbook(repo_root: Path) -> Workbook`，CLI 固定为 `--repo-root` 和 `--output`。
- 验证路径：单元测试、CLI 生成、openpyxl 结构检查、xlsx 技能检查四层验证。
