"""多引擎结果对比聚合。

读取各 serve_profile/topology_profile 的 result.csv，按
(model, bench_profile, input_len, output_len, parallel_num) 对齐多引擎，产出
compare.csv / compare.xlsx 与图表。

铁律：原始 result.csv 只读，本模块永不修改或删除它们。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, NamedTuple

logger = logging.getLogger(__name__)

# 参与对比的指标（须为 result.csv 真实列名）
COMPARE_METRICS = ("throughput_tok_s", "ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms", "cache_hit_rate")
PLOT_METRICS = ("throughput_tok_s", "ttft_p50_ms")
_PLOT_YLABEL = {
    "throughput_tok_s": "输出吞吐 (tok/s)",
    "ttft_p50_ms": "TTFT p50 (ms)",
}

# CJK 字体 fallback：bench-runner 镜像装 fonts-wqy-microhei 命中首个；
# 兼容装了 Noto Sans CJK 的环境；都没有则退回 DejaVu Sans（不崩，但中文仍为方框）
_CJK_FONT_SANS_SERIF = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']


def _apply_cjk_font():
    """配置 matplotlib 中文字体，避免中文标签（如"并发数""输出吞吐"）渲染成方框。

    bench-runner 镜像装了 fonts-wqy-microhei，命中 fallback 首项，正常渲染且无告警。
    若运行环境没有任何 CJK 字体（如未装该包的开发机），会退回 DejaVu Sans，
    中文为方框——此时静默已知 Glyph missing 告警，避免刷屏；装了字体的环境不触发。
    """
    import matplotlib
    from matplotlib import font_manager
    matplotlib.rcParams['font.sans-serif'] = _CJK_FONT_SANS_SERIF
    matplotlib.rcParams['axes.unicode_minus'] = False
    available = {f.name for f in font_manager.fontManager.ttflist}
    if not any(name in available for name in _CJK_FONT_SANS_SERIF[:-1]):
        import warnings
        warnings.filterwarnings(
            "ignore", message="Glyph .* missing from font", category=UserWarning
        )


class ServingDimension(NamedTuple):
    name: str
    engine: str
    field: str
    label: str


def _serving_profiles(config: Any) -> Iterable[tuple[str, str, str]]:
    for profile in config.serve_profiles:
        yield profile.name, profile.engine, "serve_profile"
    for profile in getattr(config, "topology_profiles", ()):
        yield profile.name, profile.engine, "topology_profile"


def _serving_dimensions(config: Any) -> list[ServingDimension]:
    profiles = list(_serving_profiles(config))
    engine_counts: dict[str, int] = {}
    for _, engine, _ in profiles:
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
    labels = [
        name if engine_counts[engine] > 1 else engine
        for name, engine, _ in profiles
    ]
    if len(set(labels)) != len(labels):
        labels = [name for name, _, _ in profiles]
    return [
        ServingDimension(
            name=name,
            engine=engine,
            field=field,
            label=label,
        )
        for (name, engine, field), label in zip(profiles, labels)
    ]


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
    for serving in _serving_dimensions(config):
        for model in config.models:
            for bench in config.bench_profiles:
                csv_path = run_dir / model.name / serving.name / bench.name / "result.csv"
                if not csv_path.exists():
                    logger.warning("对比缺失结果文件，跳过：%s", csv_path)
                    continue
                for row in _read_result_rows(csv_path, bench.name):
                    key = (
                        model.name,
                        bench.name,
                        int(row["input_len"]),
                        int(row["output_len"]),
                        int(row["parallel_num"]),
                    )
                    aligned.setdefault(key, {})[serving.label] = row
    return aligned


def _ordered_labels(config: Any) -> list[str]:
    """参与对比的列前缀，顺序由 serve/topology 配置定义。

    当每个 engine 只出现一次时，列前缀保持旧格式（engine 名）。同一 engine
    出现多次时，使用 serving profile/topology profile 名称作为列前缀以避免覆盖。
    使用配置全集而非 aligned 中实际出现的列，确保缺失结果以 N/A 填充。
    """
    seen: list[str] = []
    for serving in _serving_dimensions(config):
        if serving.label not in seen:
            seen.append(serving.label)
    return seen


def _compare_fieldnames(labels: list[str]) -> list[str]:
    cols = ["model", "bench_profile", "input_len", "output_len", "parallel_num"]
    for label in labels:
        for metric in COMPARE_METRICS:
            cols.append(f"{label}__{metric}")
    return cols


def _build_compare_rows(
    aligned: dict[tuple, dict[str, dict[str, str]]], labels: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(aligned):
        model, bench_profile, in_len, out_len, parallel = key
        row: dict[str, Any] = {
            "model": model,
            "bench_profile": bench_profile,
            "input_len": in_len,
            "output_len": out_len,
            "parallel_num": parallel,
        }
        label_map = aligned[key]
        for label in labels:
            present = label in label_map
            for metric in COMPARE_METRICS:
                col = f"{label}__{metric}"
                row[col] = label_map[label].get(metric, "") if present else "N/A"
        rows.append(row)
    return rows


def _write_compare_csv(
    rows: list[dict[str, Any]], labels: list[str], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_compare_fieldnames(labels), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_compare_xlsx(
    rows: list[dict[str, Any]], labels: list[str], path: Path
) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("openpyxl 不可用，跳过 compare.xlsx")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "compare"
    fieldnames = _compare_fieldnames(labels)
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


def _plot_file_part(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _plot(
    run_dir: Path,
    aligned: dict[tuple, dict[str, dict[str, str]]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 不可用，跳过绘图")
        return
    # 设中文字体（镜像内 fonts-wqy-microhei 命中），不再压制缺字告警——
    # 设了真字体后 Glyph missing 本就不会触发；保留压制反而掩盖真实缺字问题
    _apply_cjk_font()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    include_model = len({key[0] for key in aligned}) > 1
    # 按 (model, bench_profile, input_len, output_len) 聚点
    grouped: dict[tuple, dict[str, list[tuple[int, float]]]] = {}
    for key, label_map in aligned.items():
        model, bench_profile, in_len, out_len, parallel = key
        gkey = (model, bench_profile, in_len, out_len)
        series = grouped.setdefault(gkey, {})
        for label, row in label_map.items():
            for metric in PLOT_METRICS:
                series.setdefault(f"{label}__{metric}", []).append(
                    (parallel, _to_float(row.get(metric)))
                )
    for gkey, series in grouped.items():
        model, bench_profile, in_len, out_len = gkey
        for metric in PLOT_METRICS:
            plt.figure(figsize=(7, 4))
            for label, points in sorted(series.items()):
                if not label.endswith(f"__{metric}"):
                    continue
                points_sorted = sorted(points, key=lambda p: p[0])
                xs = [p[0] for p in points_sorted]
                ys = [p[1] for p in points_sorted]
                serving_label = label.split("__", 1)[0]
                plt.plot(xs, ys, marker="o", label=serving_label)
            plt.xlabel("并发数 (parallel_num)")
            plt.ylabel(_PLOT_YLABEL[metric])
            model_prefix = f"{model} · " if include_model else ""
            plt.title(
                f"{model_prefix}{bench_profile} in={in_len} out={out_len} · {metric}"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f"{bench_profile}__{in_len}x{out_len}__{metric}.png"
            if include_model:
                filename = f"{_plot_file_part(model)}__{filename}"
            plt.savefig(plots_dir / filename)
            plt.close()


def aggregate_compare(config: Any, run_dir: Path) -> Path | None:
    """聚合多引擎结果。返回 compare.csv 路径；无任何结果时返回 None。"""
    run_dir = Path(run_dir)
    aligned = _collect_aligned(config, run_dir)
    if not aligned:
        logger.info("无可对比结果，跳过聚合")
        return None
    labels = _ordered_labels(config)
    rows = _build_compare_rows(aligned, labels)
    compare_csv = run_dir / "compare.csv"
    _write_compare_csv(rows, labels, compare_csv)
    _write_compare_xlsx(rows, labels, run_dir / "compare.xlsx")
    _plot(run_dir, aligned)
    logger.info("对比聚合完成：%s（列前缀：%s）", compare_csv, labels)
    return compare_csv
