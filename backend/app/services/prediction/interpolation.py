"""
插值预测引擎：基于实测数据的 3D 插值（input_tokens × concurrency × output_tokens → 各指标）
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sqlalchemy.orm import Session

from app.core.cache import cache_key, get_interp, set_interp
from app.core.logging import get_logger

logger = get_logger(__name__)

METRICS = [
    "throughput_tokens_s",
    "ttft_mean_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "decode_latency_mean_ms",
    "decode_latency_p90_ms",
]


def _build_interp_data(db: Session, gpu_name: str, model_name: str, gpu_count: int):
    """从 SQLite 构建插值数据集（带缓存）"""
    key = cache_key("interp", gpu_name, model_name, gpu_count)
    cached = get_interp(key)
    if cached is not None:
        return cached

    from app.models.benchmark_data import BenchmarkData

    rows = (
        db.query(BenchmarkData)
        .filter_by(gpu_name=gpu_name, model_name=model_name, gpu_count=gpu_count)
        .all()
    )

    if len(rows) < 4:
        return None

    # 提取坐标轴唯一值
    input_vals = sorted({r.input_tokens for r in rows})
    conc_vals = sorted({r.concurrency for r in rows})
    output_vals = sorted({r.output_tokens for r in rows})

    if len(input_vals) < 2 or len(conc_vals) < 2:
        return None

    # 如果 output_tokens 只有 1 个值，降为 2D
    use_output_dim = len(output_vals) >= 2

    axes = (np.array(input_vals), np.array(conc_vals))
    if use_output_dim:
        axes = (np.array(input_vals), np.array(conc_vals), np.array(output_vals))

    # 构建网格数据
    interp_map: dict[str, RegularGridInterpolator | None] = {}
    for metric in METRICS:
        if use_output_dim:
            grid = np.full((len(input_vals), len(conc_vals), len(output_vals)), np.nan)
        else:
            grid = np.full((len(input_vals), len(conc_vals)), np.nan)

        for row in rows:
            i = input_vals.index(row.input_tokens)
            j = conc_vals.index(row.concurrency)
            val = getattr(row, metric, None)
            if val is None:
                continue
            if use_output_dim:
                try:
                    k = output_vals.index(row.output_tokens)
                    grid[i, j, k] = val
                except ValueError:
                    pass
            else:
                grid[i, j] = val

        # 用 nanmean 填充孤立 NaN
        nan_mask = np.isnan(grid)
        if nan_mask.all():
            interp_map[metric] = None
            continue

        grid_filled = _fill_nan(grid)
        try:
            interp_map[metric] = RegularGridInterpolator(
                axes, grid_filled,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,  # 超出范围时返回 nan（由上层 clamp 保证不触达）
            )
        except Exception as e:
            logger.warning("interp_build_error", metric=metric, error=str(e))
            interp_map[metric] = None

    result = {
        "interp_map": interp_map,
        "axes": axes,
        "use_output_dim": use_output_dim,
        "data_count": len(rows),
        "input_range": (min(input_vals), max(input_vals)),
        "conc_range": (min(conc_vals), max(conc_vals)),
        "output_range": (min(output_vals), max(output_vals)) if use_output_dim else None,
    }
    set_interp(key, result)
    return result


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    """
    修补网格 NaN：沿并发数轴（axis=1）做前向 + 后向填充。

    背景：网格的并发轴取所有 input_tokens 切片的并集，某些高 input_tokens 切片
    在高并发区没有实测数据（NaN）。若用全局均值填充，会产生物理上不可能的平台值，
    使 calcK 误判出"假有效窗口"。

    改用"最后已知值"前向填充：即高并发区没数据时，沿用同切片中最后一个有效并发
    点的值（相当于假设性能已到顶 / 不再继续优化），TTFT 会保持在高并发测量值，
    自然超出约束阈值，calcK 不会把这些区域纳入有效范围。
    """
    filled = grid.copy()

    def _ff_bf_1d(arr: np.ndarray) -> np.ndarray:
        """对一维数组先前向填充再后向填充"""
        out = arr.copy()
        last = np.nan
        for i in range(len(out)):
            if not np.isnan(out[i]):
                last = out[i]
            elif not np.isnan(last):
                out[i] = last
        first = np.nan
        for i in range(len(out) - 1, -1, -1):
            if not np.isnan(out[i]):
                first = out[i]
            elif not np.isnan(first):
                out[i] = first
        return out

    if filled.ndim == 2:
        n_input, _ = filled.shape
        for i in range(n_input):
            filled[i, :] = _ff_bf_1d(filled[i, :])
    elif filled.ndim == 3:
        n_input, _, n_output = filled.shape
        for i in range(n_input):
            for k in range(n_output):
                filled[i, :, k] = _ff_bf_1d(filled[i, :, k])

    # 全局均值兜底（极少情况：某切片整列都是 NaN）
    flat = filled.reshape(-1)
    if np.isnan(flat).any():
        valid_mean = float(np.nanmean(flat)) if not np.isnan(flat).all() else 0.0
        flat[np.isnan(flat)] = valid_mean
    return filled


def predict(
    db: Session,
    gpu_name: str,
    model_name: str,
    gpu_count: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> dict | None:
    """
    返回插值预测结果，数据不足时返回 None
    """
    interp_data = _build_interp_data(db, gpu_name, model_name, gpu_count)
    if interp_data is None:
        return None

    interp_map = interp_data["interp_map"]
    use_output_dim = interp_data["use_output_dim"]

    # 将查询点 clamp 到数据网格边界，防止线性外推产生荒谬值
    axes = interp_data["axes"]
    in_tokens_c   = float(np.clip(input_tokens,  axes[0][0], axes[0][-1]))
    concurrency_c = float(np.clip(concurrency,   axes[1][0], axes[1][-1]))
    if use_output_dim:
        out_tokens_c = float(np.clip(output_tokens, axes[2][0], axes[2][-1]))
        point = [[in_tokens_c, concurrency_c, out_tokens_c]]
    else:
        point = [[in_tokens_c, concurrency_c]]

    results: dict = {}
    for metric, interp in interp_map.items():
        if interp is None:
            results[metric] = None
            continue
        try:
            val = float(interp(point)[0])
            results[metric] = max(0.0, val)
        except Exception:
            results[metric] = None

    # 置信度：在原始数据范围内为高置信，超出范围（使用边界clamp）时降低
    in_input_range = interp_data["input_range"][0] <= input_tokens <= interp_data["input_range"][1]
    in_conc_range  = interp_data["conc_range"][0]  <= concurrency   <= interp_data["conc_range"][1]
    in_output_range = (
        not use_output_dim
        or (interp_data["output_range"] is not None
            and interp_data["output_range"][0] <= output_tokens <= interp_data["output_range"][1])
    )
    in_range = in_input_range and in_conc_range and in_output_range
    base_confidence = 0.90 if in_range else 0.55
    data_factor = min(1.0, interp_data["data_count"] / 20)
    confidence = round(base_confidence * data_factor, 2)

    results["confidence"] = confidence
    results["data_count"] = interp_data["data_count"]
    results["is_extrapolation"] = not in_range
    return results
