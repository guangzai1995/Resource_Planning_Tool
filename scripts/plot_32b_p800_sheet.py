#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""只绘制一张图：input=1024/output=1024 的并发选择可视化。

- 子图1纵轴：per_user_tps = 1000 / 平均增量时延(ms)（单位仍为 tokens/s）
- “最优测试点”必须来自真实测试并发点（不会出现 182 这种数据里不存在的并发）

输出文件名：
    outputs/32B-P800/input{input}_output{output}_delay{max_delay}_tps{user_throughput}.png
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch

DEFAULT_XLSX = "/work/dev-code/Resource_Planning_Tool/高价值场景-66@算力和部署评估表@260112.xlsx"
DEFAULT_SHEET = "32B-P800测试数据"


def _set_cn_font() -> None:
    def _try_add_font_file(font_file: str) -> Optional[str]:
        try:
            if (not font_file) or (not os.path.exists(font_file)):
                return None
            font_manager.fontManager.addfont(font_file)
            return FontProperties(fname=font_file).get_name()
        except Exception:
            return None

    preferred_files: List[str] = []
    if os.environ.get("CJK_FONT_PATH"):
        preferred_files.append(os.environ["CJK_FONT_PATH"])

    try:
        out = subprocess.check_output(
            ["fc-match", "-f", "%{file}\\n", "Noto Sans CJK SC"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            preferred_files.append(out)
    except Exception:
        pass

    preferred_files.extend(
        [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ]
    )

    chosen_family: Optional[str] = None
    for ff in preferred_files:
        chosen_family = _try_add_font_file(ff)
        if chosen_family:
            break

    plt.rcParams["font.family"] = "sans-serif"
    if chosen_family:
        plt.rcParams["font.sans-serif"] = [chosen_family, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "WenQuanYi Micro Hei",
            "SimHei",
            "Microsoft YaHei",
            "DejaVu Sans",
        ]
    plt.rcParams["axes.unicode_minus"] = False


def load_group(xlsx_path: str, sheet_name: str, input_len: int, output_len: int) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]

    required = [
        "输入长度",
        "输出长度",
        "并发数",
        "平均首tokens时延（ms）",
        "平均增量时延（ms）",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sheet列缺失: {missing}. 实际列: {list(df.columns)}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required, how="any")
    df = df[(df["输入长度"] == input_len) & (df["输出长度"] == output_len)].copy()
    if df.empty:
        raise ValueError(f"找不到 input={input_len}, output={output_len} 的数据")

    df = df.sort_values(["并发数"], ascending=True).reset_index(drop=True)
    return df


def compute_boundary(
    conc: np.ndarray,
    per_user_tps: np.ndarray,
    first_token_delay_s: np.ndarray,
    max_delay_s: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    """token延时阈值找边界，并在边界处返回(并发, per_user_tps, delay_s)。


    - 返回的并发可能是浮点（用于展示“插值点”）
    """

    eps = 0.001
    target_delay = max_delay_s - eps

    for i in range(len(first_token_delay_s)):
        if first_token_delay_s[i] >= max_delay_s:
            if i == 0:
                return None, None, None, False

            x1, y1 = conc[i - 1], first_token_delay_s[i - 1]
            x2, y2 = conc[i], first_token_delay_s[i]

            if y2 != y1:
                est_conc = x1 + (target_delay - y1) * (x2 - x1) / (y2 - y1)
                if x1 <= est_conc <= x2:
                    p1, p2 = per_user_tps[i - 1], per_user_tps[i]
                    est_tps = p1 + (est_conc - x1) * (p2 - p1) / (x2 - x1)
                    return float(est_conc), float(est_tps), float(target_delay), True

            return float(x1), float(per_user_tps[i - 1]), float(first_token_delay_s[i - 1]), False

    if len(conc) == 0:
        return None, None, None, False
    return float(conc[-1]), float(per_user_tps[-1]), float(first_token_delay_s[-1]), False


def choose_optimal_test_point(
    group: pd.DataFrame, max_delay_s: float, user_throughput_req: float
) -> Optional[Tuple[int, float, float]]:
    """严格从真实测试点里选“最优测试点”（并发最大且满足两条约束）。"""

    conc = group["并发数"].to_numpy(dtype=int)
    delay_s = group["平均首tokens时延（ms）"].to_numpy(dtype=float) / 1000.0
    inc_ms = group["平均增量时延（ms）"].to_numpy(dtype=float)
    per_user_tps = np.where(inc_ms > 0, 1000.0 / inc_ms, np.nan)

    best: Optional[Tuple[int, float, float]] = None
    for c, d, pu in zip(conc, delay_s, per_user_tps):
        if (d < max_delay_s) and np.isfinite(pu) and (float(pu) >= user_throughput_req):
            if best is None or c > best[0]:
                best = (int(c), float(pu), float(d))
    return best


def plot_one(
    group: pd.DataFrame,
    *,
    input_len: int,
    output_len: int,
    max_delay_s: float,
    user_throughput_req: float,
    out_path: str,
) -> None:
    _set_cn_font()
    plt.close("all")

    conc = group["并发数"].to_numpy(dtype=float)
    delay_s = group["平均首tokens时延（ms）"].to_numpy(dtype=float) / 1000.0
    inc_ms = group["平均增量时延（ms）"].to_numpy(dtype=float)
    per_user_tps = np.where(inc_ms > 0, 1000.0 / inc_ms, np.nan)

    colors: List[str] = []
    for pu, d in zip(per_user_tps, delay_s):
        if d >= max_delay_s:
            colors.append("#FF6B6B")
        elif (not np.isfinite(pu)) or float(pu) < user_throughput_req:
            colors.append("#FFB86C")
        else:
            colors.append("#50FA7B")

    b_x, b_tps, b_delay, _ = compute_boundary(conc, per_user_tps, delay_s, max_delay_s)
    best = choose_optimal_test_point(group, max_delay_s, user_throughput_req)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：per-user throughput
    ax1.scatter(conc, per_user_tps, c=colors, s=60, alpha=0.75, label="原始数据点")
    ax1.plot(conc, per_user_tps, "#6272A4", alpha=0.3, linewidth=1, label="")
    ax1.axhline(
        user_throughput_req,
        color="#FFB86C",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"吞吐要求: ≥ {user_throughput_req}",
    )

    # 固定参考线：15 tokens/s
    if abs(float(user_throughput_req) - 15.0) > 1e-9:
        ax1.axhline(
            15.0,
            color="#FFB86C",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label="参考吞吐: 15",
        )

    if b_x is not None and b_tps is not None:
        ax1.scatter(
            b_x,
            b_tps,
            c="#8BE9FD",
            s=150,
            marker="s",
            alpha=0.9,
            edgecolors="#6272A4",
            linewidth=2,
            label="最优插值点",
            zorder=4,
        )
        ax1.annotate(
            f"最优插值点: 并发{b_x:.1f}\n吞吐{b_tps:.1f}",
            (b_x, b_tps),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#8BE9FD", alpha=0.15),
            arrowprops=dict(arrowstyle="->", color="#8BE9FD", alpha=0.6, lw=1.2),
            fontsize=9,
            weight="bold",
        )

    opt_x = opt_tps = opt_delay = None
    if best is not None:
        opt_x, opt_tps, opt_delay = best
        ax1.scatter(
            opt_x,
            opt_tps,
            c="#FF79C6",
            s=300,
            marker="*",
            edgecolors="#BD93F9",
            linewidth=2,
            label="最优测试点",
            zorder=5,
        )
        ax1.annotate(
            f"最优测试点: 并发{opt_x}\n{opt_tps:.1f} tokens/s",
            (opt_x, opt_tps),
            xytext=(-140, 14),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FF79C6", alpha=0.15),
            arrowprops=dict(arrowstyle="->", color="#FF79C6", alpha=0.55, lw=1.2),
            fontsize=9,
            weight="bold",
        )

    ax1.set_xlabel("并发数")
    ax1.set_ylabel("吞吐量 (tokens/s)")
    ax1.set_title(f"P800 - 32B - 输入{input_len} 输出{output_len} (吞吐=1000/平均增量时延ms)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # 子图2：首 token 延时
    ax2.scatter(conc, delay_s, c=colors, s=60, alpha=0.75, label="原始数据点")
    ax2.plot(conc, delay_s, "#6272A4", alpha=0.3, linewidth=1, label="趋势线")
    ax2.axhline(
        y=max_delay_s,
        color="#FF6B6B",
        linestyle="--",
        linewidth=2,
        label=f"延时要求: < {max_delay_s}s (严格小于)",
        alpha=0.8,
    )
    ax2.axhspan(0, max_delay_s, alpha=0.1, color="#50FA7B")

    if b_x is not None and b_delay is not None:
        ax2.scatter(
            b_x,
            b_delay,
            c="#8BE9FD",
            s=150,
            marker="s",
            alpha=0.9,
            edgecolors="#6272A4",
            linewidth=2,
            label="最优插值点",
            zorder=4,
        )
        ax2.annotate(
            f"最优插值点: 并发{b_x:.1f}\n延时{b_delay:.3f}s",
            (b_x, b_delay),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#8BE9FD", alpha=0.15),
            arrowprops=dict(arrowstyle="->", color="#8BE9FD", alpha=0.6, lw=1.2),
            fontsize=9,
            weight="bold",
        )

    if best is not None and opt_x is not None and opt_delay is not None:
        ax2.scatter(
            opt_x,
            opt_delay,
            c="#FF79C6",
            s=300,
            marker="*",
            edgecolors="#BD93F9",
            linewidth=2,
            label="最优测试点",
            zorder=5,
        )
        ax2.annotate(
            f"最优测试点: 并发{opt_x}\n延时{opt_delay:.3f}s",
            (opt_x, opt_delay),
            xytext=(-140, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FF79C6", alpha=0.15),
            arrowprops=dict(arrowstyle="->", color="#FF79C6", alpha=0.55, lw=1.2),
            fontsize=9,
            weight="bold",
        )

    # 跨子图虚线连接
    if b_x is not None and b_tps is not None and b_delay is not None:
        fig.add_artist(
            ConnectionPatch(
                xyA=(b_x, b_tps),
                coordsA=ax1.transData,
                xyB=(b_x, b_delay),
                coordsB=ax2.transData,
                linestyle="--",
                linewidth=1.5,
                color="#8BE9FD",
                alpha=0.65,
                zorder=3,
            )
        )
    if best is not None and opt_x is not None and opt_tps is not None and opt_delay is not None:
        fig.add_artist(
            ConnectionPatch(
                xyA=(opt_x, opt_tps),
                coordsA=ax1.transData,
                xyB=(opt_x, opt_delay),
                coordsB=ax2.transData,
                linestyle="--",
                linewidth=1.5,
                color="#FF79C6",
                alpha=0.55,
                zorder=3,
            )
        )

    ax2.set_xlabel("并发数")
    ax2.set_ylabel("平均首tokens时延 (秒)")
    ax2.set_title("首token延时随并发数变化")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--sheet", default=DEFAULT_SHEET)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--max-delay", type=float, default=3.0, help="首token延时约束(秒)，严格小于")
    ap.add_argument("--user-throughput", type=float, default=15.0, help="单用户吞吐约束(tokens/s)")
    ap.add_argument("--output-dir", default="/work/dev-code/Resource_Planning_Tool/outputs/32B-P800")
    args = ap.parse_args()

    group = load_group(args.xlsx, args.sheet, args.input_len, args.output_len)

    out_path = os.path.join(
        args.output_dir,
        f"input{args.input_len}_output{args.output_len}_delay{args.max_delay}_tps{args.user_throughput}.png",
    )
    plot_one(
        group,
        input_len=args.input_len,
        output_len=args.output_len,
        max_delay_s=args.max_delay,
        user_throughput_req=args.user_throughput,
        out_path=out_path,
    )
    print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
