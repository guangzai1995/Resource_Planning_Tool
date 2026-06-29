"""Generate the inference token factory leadership report workbook."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


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

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SUB_FILL = PatternFill("solid", fgColor="D9EAF7")
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

PRINCIPLE_DIAGRAMS = {
    "04_模型量化": ["BF16/FP16 权重与激活", "FP8/量化模型", "显存下降", "吞吐提升"],
    "05_KVCache量化": ["KV Cache Block", "fp16/bf16 KV", "fp8 KV", "fp4 H200 不支持"],
    "06_Prefix_KV命中": ["共享前缀", "Prefix Hash", "复用 KV", "TTFT 降低"],
    "07_投机解码": ["Draft 候选 token", "Verify", "Accept/Reject", "TPOT 降低"],
    "08_PD分离": ["Prefill(H200)", "KV 传输", "Decode(H200/H20)", "H200 P + H20 D"],
    "09_MOE专家并行": ["Router", "多 GPU Experts", "All2All 聚合", "GLM5.1/GLM5.2 FP8"],
    "10_连续批处理": ["Scheduler", "新请求 Prefill", "旧请求 Decode", "框架原生支持"],
}


def _style_range(ws, min_row: int, max_row: int, min_col: int, max_col: int) -> None:
    for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
        for cell in row:
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical="center", wrap_text=True)


def _style_table_header(ws, row: int, max_col: int) -> None:
    for row_cells in ws.iter_rows(min_row=row, max_row=row, min_col=1, max_col=max_col):
        for cell in row_cells:
            cell.fill = SUB_FILL
            cell.font = Font(bold=True)


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
    header_row = ws.max_row
    for index, spec in enumerate(SHEET_SPECS):
        ws.append([index, spec["title"], spec["owner"], spec["status"], "按规格生成"])
    _style_range(ws, header_row, ws.max_row, 1, 5)
    _style_table_header(ws, header_row, 5)


def _write_summary(ws) -> None:
    ws.append([])
    ws.append(["技术点", "状态", "汇报口径"])
    header_row = ws.max_row
    for row in TECH_ROWS:
        ws.append(list(row))
    _style_range(ws, header_row, ws.max_row, 1, 3)
    _style_table_header(ws, header_row, 3)


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
            ws.cell(start_row, col + 1).border = THIN_BORDER
            ws.column_dimensions[ws.cell(start_row, col + 1).column_letter].width = 6
    ws["A10"] = "说明"
    ws["A10"].font = Font(bold=True)
    ws["A10"].alignment = Alignment(vertical="center", wrap_text=True)
    ws["A10"].border = THIN_BORDER
    ws["B10"] = "该图用于快速解释技术机制，详细参数与测试设计见下方表格。"
    ws["B10"].alignment = Alignment(vertical="center", wrap_text=True)
    ws["B10"].border = THIN_BORDER
    ws.merge_cells("B10:G10")


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


def build_workbook(repo_root: Path) -> Workbook:
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    for spec in SHEET_SPECS:
        ws = wb.create_sheet(spec["title"])
        _write_header(ws, spec)
        if spec["title"] in PRINCIPLE_DIAGRAMS:
            _write_principle_diagram(ws, PRINCIPLE_DIAGRAMS[spec["title"]])
        if spec["title"] == "00_目录与版本":
            _write_sheet_index(ws)
        if spec["title"] == "02_总览结论":
            _write_summary(ws)
    return wb
