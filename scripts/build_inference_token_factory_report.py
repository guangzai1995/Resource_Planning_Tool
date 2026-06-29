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
