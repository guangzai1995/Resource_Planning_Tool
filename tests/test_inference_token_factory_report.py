from pathlib import Path

from openpyxl import load_workbook
import pytest

from scripts.build_inference_token_factory_report import (
    REPORT_VERSION,
    SHEET_SPECS,
    STATUS_LABELS,
    build_workbook,
    compute_fp8_summary,
    load_context_summary,
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

PRINCIPLE_EXPECTATIONS = {
    "04_模型量化": ["BF16/FP16", "FP8", "吞吐提升"],
    "05_KVCache量化": ["KV Cache Block", "fp8 KV", "fp4 H200 不支持"],
    "06_Prefix_KV命中": ["共享前缀", "Prefix Hash", "复用 KV"],
    "07_投机解码": ["Draft", "Verify", "Accept/Reject"],
    "08_PD分离": ["Prefill", "KV 传输", "Decode", "H200 P + H20 D"],
    "09_MOE专家并行": ["Router", "Experts", "All2All", "GLM5.1/GLM5.2 FP8"],
    "10_连续批处理": ["Scheduler", "Prefill", "Decode", "框架原生支持"],
}


def assert_cell_has_table_style(cell):
    assert cell.alignment.wrap_text is True
    assert cell.border.left.style == "thin"


def assert_cell_has_table_header_style(cell):
    assert cell.fill.fgColor.rgb in {"00D9EAF7", "D9EAF7"}
    assert cell.font.bold is True


def test_static_sheet_specs_and_status_labels():
    assert REPORT_VERSION == "v2.0"
    assert [item["title"] for item in SHEET_SPECS] == EXPECTED_SHEETS
    assert STATUS_LABELS == [
        "已验证",
        "有结论但缺系统数据",
        "待验证",
        "已具备无需独立测试",
    ]


def test_build_workbook_creates_expected_sheets():
    wb = build_workbook(Path("."))
    assert wb.sheetnames == EXPECTED_SHEETS


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

    index_ws = loaded["00_目录与版本"]
    assert index_ws["A7"].value == "Sheet"
    assert_cell_has_table_header_style(index_ws["A7"])
    assert index_ws.cell(row=index_ws.max_row, column=2).value == "12_数据附录"
    assert_cell_has_table_style(index_ws.cell(row=index_ws.max_row, column=2))


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
    assert ws["A7"].value == "技术点"
    assert_cell_has_table_header_style(ws["A7"])
    assert ws.cell(row=ws.max_row, column=1).value == "连续批处理"
    assert_cell_has_table_style(ws.cell(row=ws.max_row, column=1))


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
