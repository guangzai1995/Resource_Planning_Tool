#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""对比：原始数据 vs 增加网络延时后的并发推荐。

需求对应：
- 网络延时设为 20ms（可通过参数调整）
- 仍满足两个约束：
  1) 吞吐量（per_user_tps = 1000/平均增量时延ms） >= user_throughput
  2) 首 token 延时（严格小于 max_delay）
- 计算仍考虑插值：边界点使用 delay=max_delay-eps 做线性插值

实现假设（按你的最新要求）：
- 网络延时同时作用于：
    - 首 token 延时：delay_s' = delay_s + net_ms/1000
    - 平均增量时延：inc_ms' = inc_ms + net_ms
    因此网络情形下吞吐也会下降：per_user_tps' = 1000 / inc_ms'

输出：
  outputs/32B-P800/compare_net{net_ms}ms_input{input}_output{output}_delay{max_delay}_tps{user_throughput}.png
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch


def _load_base_plot_module():
    """动态加载同目录下的 `plot_32b_p800_sheet.py`，避免脚本式运行时的包导入问题。"""

    base_path = os.path.join(os.path.dirname(__file__), "plot_32b_p800_sheet.py")
    spec = importlib.util.spec_from_file_location("plot_32b_p800_sheet", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base_plot_module()
_set_cn_font = _base._set_cn_font
choose_optimal_test_point = _base.choose_optimal_test_point
load_group = _base.load_group


def compute_feasible_boundary(
    conc: np.ndarray,
    per_user_tps: np.ndarray,
    first_token_delay_s: np.ndarray,
    *,
    max_delay_s: float,
    user_throughput_req: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    """计算“同时满足两约束”的最大可行并发边界，并在边界处做线性插值。

    约束：
    - delay < max_delay_s（严格小于，因此使用 max_delay_s - eps 插值）
    - per_user_tps >= user_throughput_req

    返回：(boundary_conc, boundary_tps, boundary_delay_s, interpolated)
    """

    if len(conc) == 0:
        return None, None, None, False

    eps = 0.001
    ok = (
        np.isfinite(per_user_tps)
        & (per_user_tps >= float(user_throughput_req))
        & (first_token_delay_s < float(max_delay_s))
    )
    ok_idx = np.where(ok)[0]
    if len(ok_idx) == 0:
        return None, None, None, False

    last = int(ok_idx[-1])
    if last == len(conc) - 1:
        return float(conc[last]), float(per_user_tps[last]), float(first_token_delay_s[last]), False

    i0 = last
    i1 = last + 1
    x0, x1 = float(conc[i0]), float(conc[i1])
    t0, t1 = float(per_user_tps[i0]), float(per_user_tps[i1])
    d0, d1 = float(first_token_delay_s[i0]), float(first_token_delay_s[i1])

    candidates: list[float] = []

    # 若下一个点延时超限：找 delay=max_delay-eps 的交点
    if d1 >= float(max_delay_s):
        target_d = float(max_delay_s) - eps
        if d1 != d0:
            x_delay = x0 + (target_d - d0) * (x1 - x0) / (d1 - d0)
            if x0 <= x_delay <= x1:
                candidates.append(float(x_delay))

    # 若下一个点吞吐不达标：找 tps=req 的交点
    if (not np.isfinite(t1)) or (t1 < float(user_throughput_req)):
        target_t = float(user_throughput_req)
        if t1 != t0:
            x_tps = x0 + (target_t - t0) * (x1 - x0) / (t1 - t0)
            if x0 <= x_tps <= x1:
                candidates.append(float(x_tps))

    if not candidates:
        return float(x0), float(t0), float(d0), False

    xb = min(candidates)
    # 在边界并发 xb 处插值回 tps/delay
    if x1 != x0:
        tb = t0 + (xb - x0) * (t1 - t0) / (x1 - x0)
        db = d0 + (xb - x0) * (d1 - d0) / (x1 - x0)
    else:
        tb, db = t0, d0
    return float(xb), float(tb), float(db), True


def choose_optimal_test_point_with_net_latency(
    group,
    *,
    max_delay_s: float,
    user_throughput_req: float,
    net_latency_ms: float,
) -> Optional[Tuple[int, float, float]]:
    """严格从真实测试点里选最优测试点。

    网络情形下：
    - 首token延时 + net_latency_ms
    - 平均增量时延 + net_latency_ms（因此 per_user_tps 下降）
    """

    conc = group["并发数"].to_numpy(dtype=int)
    delay_s = group["平均首tokens时延（ms）"].to_numpy(dtype=float) / 1000.0
    delay_s = delay_s + (float(net_latency_ms) / 1000.0)

    inc_ms = group["平均增量时延（ms）"].to_numpy(dtype=float)
    inc_ms = inc_ms + float(net_latency_ms)
    per_user_tps = np.where(inc_ms > 0, 1000.0 / inc_ms, np.nan)

    best: Optional[Tuple[int, float, float]] = None
    for c, d, pu in zip(conc, delay_s, per_user_tps):
        if (d < max_delay_s) and np.isfinite(pu) and (float(pu) >= user_throughput_req):
            if best is None or c > best[0]:
                best = (int(c), float(pu), float(d))
    return best


def plot_compare(
    *,
    xlsx: str,
    sheet: str,
    input_len: int,
    output_len: int,
    max_delay_s: float,
    user_throughput_req: float,
    net_latency_ms: float,
    out_path: str,
) -> None:
    _set_cn_font()
    plt.close("all")

    group = load_group(xlsx, sheet, input_len, output_len)

    conc = group["并发数"].to_numpy(dtype=float)
    base_delay_s = group["平均首tokens时延（ms）"].to_numpy(dtype=float) / 1000.0
    net_delay_s = base_delay_s + (float(net_latency_ms) / 1000.0)

    inc_ms = group["平均增量时延（ms）"].to_numpy(dtype=float)
    base_per_user_tps = np.where(inc_ms > 0, 1000.0 / inc_ms, np.nan)

    net_inc_ms = inc_ms + float(net_latency_ms)
    net_per_user_tps = np.where(net_inc_ms > 0, 1000.0 / net_inc_ms, np.nan)

    # 计算“同时满足两约束”的可行边界（含插值）
    b0_x, b0_tps, b0_delay, _ = compute_feasible_boundary(
        conc,
        base_per_user_tps,
        base_delay_s,
        max_delay_s=max_delay_s,
        user_throughput_req=user_throughput_req,
    )
    b1_x, b1_tps, b1_delay, _ = compute_feasible_boundary(
        conc,
        net_per_user_tps,
        net_delay_s,
        max_delay_s=max_delay_s,
        user_throughput_req=user_throughput_req,
    )

    # 最优测试点（必须是实测并发）
    best0 = choose_optimal_test_point(group, max_delay_s, user_throughput_req)
    best1 = choose_optimal_test_point_with_net_latency(
        group,
        max_delay_s=max_delay_s,
        user_throughput_req=user_throughput_req,
        net_latency_ms=net_latency_ms,
    )

    # 画图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：吞吐（使用更深配色，风格对齐单图脚本）
    ax1.plot(conc, base_per_user_tps, color="#6272A4", linewidth=2.2, alpha=0.75, label="原始吞吐")
    ax1.plot(conc, net_per_user_tps, color="#FF79C6", linewidth=2.2, alpha=0.75, label=f"吞吐 +{net_latency_ms:.0f}ms")
    ax1.scatter(conc, base_per_user_tps, c="#6272A4", s=60, alpha=0.75)
    ax1.scatter(conc, net_per_user_tps, c="#FF79C6", s=60, alpha=0.55)
    ax1.axhline(
        user_throughput_req,
        color="#FFB86C",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"吞吐要求: ≥ {user_throughput_req}",
    )

    # 边界插值点（原始 vs net）
    if b0_x is not None and b0_tps is not None:
        ax1.scatter(b0_x, b0_tps, c="#8BE9FD", s=160, marker="s", alpha=0.9, edgecolors="#6272A4", linewidth=2, label="原始-可行插值边界", zorder=4)
        ax1.annotate(
            f"原始可行边界: {b0_x:.1f}",
            (b0_x, b0_tps),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#8BE9FD", alpha=0.12),
            arrowprops=dict(arrowstyle="->", color="#8BE9FD", alpha=0.6, lw=1.0),
        )

    if b1_x is not None and b1_tps is not None:
        ax1.scatter(b1_x, b1_tps, c="#F1FA8C", s=160, marker="s", alpha=0.9, edgecolors="#6272A4", linewidth=2, label=f"+{net_latency_ms:.0f}ms-可行插值边界", zorder=4)
        ax1.annotate(
            f"+{net_latency_ms:.0f}ms可行边界: {b1_x:.1f}",
            (b1_x, b1_tps),
            xytext=(10, -18),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F1FA8C", alpha=0.12),
            arrowprops=dict(arrowstyle="->", color="#F1FA8C", alpha=0.6, lw=1.0),
        )

    # 最优测试点（原始 vs net）
    opt0_x = opt0_tps = opt0_delay = None
    if best0 is not None:
        opt0_x, opt0_tps, opt0_delay = best0
        ax1.scatter(opt0_x, opt0_tps, c="#FF79C6", s=320, marker="*", edgecolors="#BD93F9", linewidth=2, label="原始-最优测试点", zorder=5)
        ax1.annotate(
            f"原始最优: {opt0_x}",
            (opt0_x, opt0_tps),
            xytext=(-120, 12),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FF79C6", alpha=0.12),
            arrowprops=dict(arrowstyle="->", color="#FF79C6", alpha=0.55, lw=1.0),
        )

    opt1_x = opt1_tps = opt1_delay = None
    if best1 is not None:
        opt1_x, opt1_tps, opt1_delay = best1
        ax1.scatter(opt1_x, opt1_tps, c="#50FA7B", s=320, marker="*", edgecolors="#BD93F9", linewidth=2, label=f"+{net_latency_ms:.0f}ms-最优测试点", zorder=5)
        ax1.annotate(
            f"+{net_latency_ms:.0f}ms最优: {opt1_x}",
            (opt1_x, opt1_tps),
            xytext=(-120, -18),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#50FA7B", alpha=0.12),
            arrowprops=dict(arrowstyle="->", color="#50FA7B", alpha=0.55, lw=1.0),
        )

    # 汇总：同时给出“可行插值边界”和“最优测试点”的并发变化，避免混淆
    summary_lines = []
    if b0_x is not None and b1_x is not None:
        down = float(b0_x) - float(b1_x)
        summary_lines.append(f"可行插值边界: {b0_x:.1f} → {b1_x:.1f}  (下降{down:.1f})")
    if opt0_x is not None and opt1_x is not None:
        down_tp = float(opt0_x) - float(opt1_x)
        summary_lines.append(f"最优测试点: {opt0_x} → {opt1_x}  (下降{down_tp:.0f})")
    if summary_lines:
        ax1.text(
            0.98,
            0.98,
            "\n".join(summary_lines),
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F8F8F2", alpha=0.65),
        )

    ax1.set_xlabel("并发数")
    ax1.set_ylabel("吞吐量 (tokens/s)")
    ax1.set_title(
        f"P800 - 32B - 输入{input_len} 输出{output_len} (吞吐=1000/平均增量时延ms)\n"
        f"对比：原始 vs +{net_latency_ms:.0f}ms(首token+增量时延)"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # 子图2：首 token 延时曲线对比（使用更深配色，风格对齐单图脚本）
    ax2.plot(conc, base_delay_s, color="#6272A4", linewidth=2.2, alpha=0.75, label="原始首token延时")
    ax2.plot(conc, net_delay_s, color="#FF79C6", linewidth=2.2, alpha=0.75, label=f"首token延时 +{net_latency_ms:.0f}ms")
    ax2.scatter(conc, base_delay_s, c="#6272A4", s=60, alpha=0.75)
    ax2.scatter(conc, net_delay_s, c="#FF79C6", s=60, alpha=0.55)

    ax2.axhline(
        y=max_delay_s,
        color="#FF6B6B",
        linestyle="--",
        linewidth=2,
        label=f"延时要求: < {max_delay_s}s (严格小于)",
        alpha=0.8,
    )
    ax2.axhspan(0, max_delay_s, alpha=0.08, color="#50FA7B")

    if b0_x is not None and b0_delay is not None:
        ax2.scatter(b0_x, b0_delay, c="#8BE9FD", s=160, marker="s", alpha=0.9, edgecolors="#6272A4", linewidth=2, zorder=4)
    if b1_x is not None and b1_delay is not None:
        ax2.scatter(b1_x, b1_delay, c="#F1FA8C", s=160, marker="s", alpha=0.9, edgecolors="#6272A4", linewidth=2, zorder=4)

    if opt0_x is not None and opt0_delay is not None:
        ax2.scatter(opt0_x, opt0_delay, c="#FF79C6", s=320, marker="*", edgecolors="#BD93F9", linewidth=2, zorder=5)
    if opt1_x is not None and opt1_delay is not None:
        ax2.scatter(opt1_x, opt1_delay, c="#50FA7B", s=320, marker="*", edgecolors="#BD93F9", linewidth=2, zorder=5)

    # 跨子图虚线连接（便于一眼看到“同一个点”的吞吐-延时对应）
    if b0_x is not None and b0_tps is not None and b0_delay is not None:
        fig.add_artist(ConnectionPatch(xyA=(b0_x, b0_tps), coordsA=ax1.transData, xyB=(b0_x, b0_delay), coordsB=ax2.transData, linestyle="--", linewidth=1.2, color="#8BE9FD", alpha=0.6))
    if b1_x is not None and b1_tps is not None and b1_delay is not None:
        fig.add_artist(ConnectionPatch(xyA=(b1_x, b1_tps), coordsA=ax1.transData, xyB=(b1_x, b1_delay), coordsB=ax2.transData, linestyle="--", linewidth=1.2, color="#F1FA8C", alpha=0.6))
    if opt0_x is not None and opt0_tps is not None and opt0_delay is not None:
        fig.add_artist(ConnectionPatch(xyA=(opt0_x, opt0_tps), coordsA=ax1.transData, xyB=(opt0_x, opt0_delay), coordsB=ax2.transData, linestyle="--", linewidth=1.2, color="#FF79C6", alpha=0.55))
    if opt1_x is not None and opt1_tps is not None and opt1_delay is not None:
        fig.add_artist(ConnectionPatch(xyA=(opt1_x, opt1_tps), coordsA=ax1.transData, xyB=(opt1_x, opt1_delay), coordsB=ax2.transData, linestyle="--", linewidth=1.2, color="#50FA7B", alpha=0.55))

    ax2.set_xlabel("并发数")
    ax2.set_ylabel("平均首tokens时延 (秒)")
    ax2.set_title("首token延时随并发数变化（对比网络延时影响）")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    base_opt = best0[0] if best0 is not None else None
    net_opt = best1[0] if best1 is not None else None
    print(f"Baseline optimal test concurrency: {base_opt}")
    print(f"+{net_latency_ms:.0f}ms optimal test concurrency: {net_opt}")
    if (base_opt is not None) and (net_opt is not None):
        print(f"Concurrency decrease: {base_opt} -> {net_opt} (delta={net_opt - base_opt})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="/work/dev-code/Resource_Planning_Tool/高价值场景-66@算力和部署评估表@260112.xlsx")
    ap.add_argument("--sheet", default="32B-P800测试数据")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--max-delay", type=float, default=3.0, help="首token延时约束(秒)，严格小于")
    ap.add_argument("--user-throughput", type=float, default=15.0, help="单用户吞吐约束(tokens/s)")
    ap.add_argument("--network-latency-ms", type=float, default=20.0, help="额外网络延时(ms)，加到首token延时上")
    ap.add_argument("--output-dir", default="/work/dev-code/Resource_Planning_Tool/outputs/32B-P800")
    args = ap.parse_args()

    out_path = os.path.join(
        args.output_dir,
        f"compare_net{int(args.network_latency_ms)}ms_"
        f"input{args.input_len}_output{args.output_len}_delay{args.max_delay}_tps{args.user_throughput}.png",
    )

    plot_compare(
        xlsx=args.xlsx,
        sheet=args.sheet,
        input_len=args.input_len,
        output_len=args.output_len,
        max_delay_s=args.max_delay,
        user_throughput_req=args.user_throughput,
        net_latency_ms=args.network_latency_ms,
        out_path=out_path,
    )
    print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
