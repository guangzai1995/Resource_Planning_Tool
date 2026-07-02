"""多引擎结果对比聚合。

读取各 serve_profile 的 result.csv，按 (bench_profile, input_len, output_len,
parallel_num) 对齐多引擎，产出 compare.csv / compare.xlsx 与图表。

铁律：原始 result.csv 只读，本模块永不修改或删除它们。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 参与对比的指标（须为 result.csv 真实列名）
COMPARE_METRICS = ("throughput_tok_s", "ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms", "cache_hit_rate")
PLOT_METRICS = ("throughput_tok_s", "ttft_p50_ms")
_PLOT_YLABEL = {
    "throughput_tok_s": "输出吞吐 (tok/s)",
    "ttft_p50_ms": "TTFT p50 (ms)",
}


def _engine_by_serve_profile(config: Any) -> dict[str, str]:
    return {profile.name: profile.engine for profile in config.serve_profiles}


def _read_result_rows(path: Path, bench_profile: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["bench_profile"] = bench_profile  # csv 内无此列，由目录名注入
    return rows


def _collect_aligned(
    config: Any, run_dir: Path
) -> dict[tuple, dict[str, dict[str, str]]]:
    aligned: dict[tuple, dict[str, dict[str, str]]] = {}
    for serve_name, engine in _engine_by_serve_profile(config).items():
        for model in config.models:
            for bench in config.bench_profiles:
                csv_path = run_dir / model.name / serve_name / bench.name / "result.csv"
                if not csv_path.exists():
                    logger.warning("对比缺失结果文件，跳过：%s", csv_path)
                    continue
                for row in _read_result_rows(csv_path, bench.name):
                    key = (
                        bench.name,
                        int(row["input_len"]),
                        int(row["output_len"]),
                        int(row["parallel_num"]),
                    )
                    aligned.setdefault(key, {})[engine] = row
    return aligned


def _ordered_engines(config: Any) -> list[str]:
    """参与对比的全部引擎，顺序由 config.serve_profiles 定义（去重）。

    使用配置全集而非 aligned 中实际出现的引擎，确保缺失引擎的列以 N/A 填充，
    且列顺序稳定（不随数据增减而抖动）。
    """
    seen: list[str] = []
    for profile in config.serve_profiles:
        if profile.engine not in seen:
            seen.append(profile.engine)
    return seen


def _compare_fieldnames(engines: list[str]) -> list[str]:
    cols = ["bench_profile", "input_len", "output_len", "parallel_num"]
    for engine in engines:
        for metric in COMPARE_METRICS:
            cols.append(f"{engine}__{metric}")
    return cols


def _build_compare_rows(
    aligned: dict[tuple, dict[str, dict[str, str]]], engines: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(aligned):
        bench_profile, in_len, out_len, parallel = key
        row: dict[str, Any] = {
            "bench_profile": bench_profile,
            "input_len": in_len,
            "output_len": out_len,
            "parallel_num": parallel,
        }
        engine_map = aligned[key]
        for engine in engines:
            present = engine in engine_map
            for metric in COMPARE_METRICS:
                col = f"{engine}__{metric}"
                row[col] = engine_map[engine].get(metric, "") if present else "N/A"
        rows.append(row)
    return rows


def _write_compare_csv(
    rows: list[dict[str, Any]], engines: list[str], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_compare_fieldnames(engines), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_compare_xlsx(
    rows: list[dict[str, Any]], engines: list[str], path: Path
) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("openpyxl 不可用，跳过 compare.xlsx")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "compare"
    fieldnames = _compare_fieldnames(engines)
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(col, "") for col in fieldnames])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _plot(
    run_dir: Path,
    aligned: dict[tuple, dict[str, dict[str, str]]],
    engines: list[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 不可用，跳过绘图")
        return
    # 默认 DejaVu Sans 缺中文字形会刷屏 UserWarning，静默该已知告警
    import warnings
    warnings.filterwarnings(
        "ignore", message="Glyph .* missing from font", category=UserWarning
    )
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # 按 (bench_profile, input_len, output_len) 聚点
    grouped: dict[tuple, dict[str, list[tuple[int, float]]]] = {}
    for key, engine_map in aligned.items():
        bench_profile, in_len, out_len, parallel = key
        gkey = (bench_profile, in_len, out_len)
        series = grouped.setdefault(gkey, {})
        for engine, row in engine_map.items():
            for metric in PLOT_METRICS:
                series.setdefault(f"{engine}__{metric}", []).append(
                    (parallel, _to_float(row.get(metric)))
                )
    for gkey, series in grouped.items():
        bench_profile, in_len, out_len = gkey
        for metric in PLOT_METRICS:
            plt.figure(figsize=(7, 4))
            for label, points in sorted(series.items()):
                if not label.endswith(f"__{metric}"):
                    continue
                points_sorted = sorted(points, key=lambda p: p[0])
                xs = [p[0] for p in points_sorted]
                ys = [p[1] for p in points_sorted]
                engine = label.split("__", 1)[0]
                plt.plot(xs, ys, marker="o", label=engine)
            plt.xlabel("并发数 (parallel_num)")
            plt.ylabel(_PLOT_YLABEL[metric])
            plt.title(f"{bench_profile} in={in_len} out={out_len} · {metric}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{bench_profile}__{in_len}x{out_len}__{metric}.png")
            plt.close()


def aggregate_compare(config: Any, run_dir: Path) -> Path | None:
    """聚合多引擎结果。返回 compare.csv 路径；无任何结果时返回 None。"""
    run_dir = Path(run_dir)
    aligned = _collect_aligned(config, run_dir)
    if not aligned:
        logger.info("无可对比结果，跳过聚合")
        return None
    engines = _ordered_engines(config)
    rows = _build_compare_rows(aligned, engines)
    compare_csv = run_dir / "compare.csv"
    _write_compare_csv(rows, engines, compare_csv)
    _write_compare_xlsx(rows, engines, run_dir / "compare.xlsx")
    _plot(run_dir, aligned, engines)
    logger.info("对比聚合完成：%s（引擎：%s）", compare_csv, engines)
    return compare_csv
