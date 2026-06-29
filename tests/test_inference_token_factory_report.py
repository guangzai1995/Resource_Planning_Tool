from pathlib import Path

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


def test_build_workbook_creates_expected_sheets():
    wb = build_workbook(Path("."))
    assert wb.sheetnames == EXPECTED_SHEETS
