"""Generate the inference token factory leadership report workbook."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

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
        wb.create_sheet(spec["title"])
    return wb
