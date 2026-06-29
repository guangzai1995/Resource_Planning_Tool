#!/usr/bin/env python3
"""
上下文分布全面分析工具
======================
基于 PostgreSQL usage_records 表的 token 统计数据，
进行全面的上下文使用分布分析，包含：
  1. 总体概览（请求量/Token/费用/延迟）
  2. 上下文长度分布（桶状分布 + 百分位）
  3. 按模型分析（上下文窗口使用率）
  4. 缓存命中率分析（cache_read/write vs input）
  5. 每日趋势
  6. 用户维度分析
  7. 可视化图表输出

输出:
  - 终端彩色输出（进度反馈）
  - Markdown 报告文件（完整分析结果）
  - CSV / JSON 数据文件
  - PNG 可视化图表

无需 psql 客户端，直接使用 psycopg2 连接数据库。

用法:
  python3 analyze_context.py                       # 默认: 全部分析
  python3 analyze_context.py --days 7              # 分析最近 7 天
  python3 analyze_context.py --module overview     # 只运行某个模块
  python3 analyze_context.py --no-plot             # 不生成图表
"""
import argparse
import os
import sys
import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("❌ 需要安装 psycopg2-binary: pip install psycopg2-binary")
    sys.exit(1)

# ─────────────────────────────────────────────
# 数据库配置
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host": os.environ.get("PG_HOST", "10.88.202.214"),
    "port": int(os.environ.get("PG_PORT", "5432")),
    "dbname": os.environ.get("PG_DATABASE", "appdb"),
    "user": os.environ.get("PG_USER", "appuser"),
    "password": os.environ.get("PG_PASSWORD", "op3mF+-LKez3AN-mQP9D"),
}

# ─────────────────────────────────────────────
# 颜色输出
# ─────────────────────────────────────────────
class C:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def colored(text, color):
    return f"{color}{text}{C.NC}"


def section(title):
    print(f"\n{colored('=' * 70, C.CYAN)}")
    print(colored(f"  {title}", C.BOLD))
    print(colored('=' * 70, C.CYAN))


def subsection(title):
    print(f"\n  {colored('▸ ' + title, C.YELLOW)}")


def fmt_num(n):
    """格式化数字，添加千分位"""
    if n is None:
        return "N/A"
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def fmt_pct(n):
    """格式化百分比"""
    if n is None:
        return "N/A"
    return f"{n:.2f}%"


# ─────────────────────────────────────────────
# 数据库连接
# ─────────────────────────────────────────────
class DB:
    def __init__(self):
        self.conn = None

    def connect(self):
        print(colored("正在连接数据库...", C.BLUE))
        self.conn = psycopg2.connect(**DB_CONFIG)
        print(colored(f"✓ 已连接到 {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}", C.GREEN))

    def query(self, sql, params=None):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    def query_scalar(self, sql, params=None):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None

    def close(self):
        if self.conn:
            self.conn.close()
            print(colored("✓ 数据库连接已关闭", C.GREEN))


# ─────────────────────────────────────────────
# Markdown 报告生成器
# ─────────────────────────────────────────────
class Report:
    """Markdown 报告累加器，所有分析结果同时写入此对象"""

    def __init__(self, title="上下文分布分析报告", output_dir="."):
        self.lines = []
        self.title = title
        self.output_dir = output_dir
        self._write_header()

    def _write_header(self):
        self.lines.append(f"# {self.title}\n")
        self.lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.lines.append("")

    def h1(self, text):
        self.lines.append(f"# {text}\n")

    def h2(self, text):
        self.lines.append(f"## {text}\n")

    def h3(self, text):
        self.lines.append(f"### {text}\n")

    def text(self, s=""):
        self.lines.append(s)

    def blank(self):
        self.lines.append("")

    def kv(self, key, value):
        """键值对，用于概览统计"""
        self.lines.append(f"- **{key}**: {value}")

    def table(self, headers, rows, align=None):
        """生成 Markdown 表格"""
        if not rows:
            self.lines.append("_(无数据)_\n")
            return
        # 表头
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        self.lines.append(header_line)
        # 对齐行
        if align is None:
            align = ["---"] * len(headers)
        self.lines.append("| " + " | ".join(align) + " |")
        # 数据行
        for row in rows:
            cells = [str(v) for v in row]
            self.lines.append("| " + " | ".join(cells) + " |")
        self.lines.append("")

    def image(self, filename, alt_text="图表"):
        """引用图片（相对路径）"""
        self.lines.append(f"![{alt_text}]({filename})\n")

    def hr(self):
        self.lines.append("---\n")

    def code(self, text, lang=""):
        self.lines.append(f"```{lang}")
        self.lines.append(text)
        self.lines.append("```\n")

    def save(self, filename="report.md"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(colored(f"  ✓ Markdown 报告已保存: {path}", C.GREEN))
        return path


# ─────────────────────────────────────────────
# 1. 总体概览
# ─────────────────────────────────────────────
def analyze_overview(db, days, output_dir, report=None):
    section("1. 总体概览（最近 {} 天）".format(days))

    rows = db.query("""
        SELECT
            COUNT(*)                                           AS total_requests,
            COUNT(*) FILTER (WHERE status = 'success')        AS success_count,
            COUNT(*) FILTER (WHERE status != 'success')       AS fail_count,
            -- Token 统计
            SUM(input_tokens)                                  AS total_input_tokens,
            SUM(output_tokens)                                 AS total_output_tokens,
            SUM(cache_read_tokens)                             AS total_cache_read,
            SUM(cache_write_tokens)                            AS total_cache_write,
            SUM(total_tokens)                                  AS total_tokens,
            AVG(input_tokens)::bigint                          AS avg_input,
            AVG(output_tokens)::bigint                         AS avg_output,
            AVG(total_tokens)::bigint                          AS avg_total,
            -- 中位数 & 极值
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_tokens)::bigint  AS median_total,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_tokens)::bigint  AS p95_total,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_tokens)::bigint  AS p99_total,
            MAX(total_tokens)                                  AS max_total,
            -- 费用
            SUM(total_cost)                                    AS total_cost,
            SUM(input_cost)                                    AS total_input_cost,
            SUM(output_cost)                                   AS total_output_cost,
            SUM(cache_read_cost)                               AS total_cache_read_cost,
            SUM(cache_write_cost)                              AS total_cache_write_cost,
            AVG(total_cost)                                    AS avg_cost,
            -- 延迟
            AVG(response_time)::bigint                         AS avg_latency_ms,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time)::bigint AS median_latency_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time)::bigint AS p95_latency_ms,
            MAX(response_time)                                 AS max_latency_ms,
            -- 缓存命中率
            CASE WHEN SUM(input_tokens) > 0
                THEN ROUND(100.0 * SUM(cache_read_tokens) / SUM(input_tokens), 2)
                ELSE 0
            END AS cache_hit_rate_pct,
            -- 流式比例
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_stream) / COUNT(*), 2) AS stream_pct
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '%s days'
    """ % days)

    r = rows[0]

    print(f"""
  📊 请求统计
     总请求数:       {fmt_num(r['total_requests'])}
     成功:           {colored(fmt_num(r['success_count']), C.GREEN)}
     失败:           {colored(fmt_num(r['fail_count']), C.RED) if r['fail_count'] and r['fail_count'] > 0 else fmt_num(r['fail_count'])}
     流式比例:       {fmt_pct(r['stream_pct'])}

  🔢 Token 统计
     总 Token:       {fmt_num(r['total_tokens'])}
     总输入 Token:   {fmt_num(r['total_input_tokens'])}
     总输出 Token:   {fmt_num(r['total_output_tokens'])}
     缓存读取 Token: {fmt_num(r['total_cache_read'])}
     缓存写入 Token: {fmt_num(r['total_cache_write'])}

  📈 Token 分布
     平均 Token:     {fmt_num(r['avg_total'])}
     中位数 Token:   {fmt_num(r['median_total'])}
     P95 Token:      {fmt_num(r['p95_total'])}
     P99 Token:      {fmt_num(r['p99_total'])}
     最大 Token:     {fmt_num(r['max_total'])}

  💰 费用统计
     总费用:         ${fmt_num(r['total_cost'])}
     平均费用:       ${fmt_num(r['avg_cost'])}
     输入费用:       ${fmt_num(r['total_input_cost'])}
     输出费用:       ${fmt_num(r['total_output_cost'])}
     缓存读取费用:   ${fmt_num(r['total_cache_read_cost'])}
     缓存写入费用:   ${fmt_num(r['total_cache_write_cost'])}

  ⚡ 延迟统计
     平均延迟:       {fmt_num(r['avg_latency_ms'])} ms
     中位数延迟:     {fmt_num(r['median_latency_ms'])} ms
     P95 延迟:       {fmt_num(r['p95_latency_ms'])} ms
     最大延迟:       {fmt_num(r['max_latency_ms'])} ms

  🔄 缓存命中率:   {colored(fmt_pct(r['cache_hit_rate_pct']), C.GREEN)}
""")

    # 保存到 JSON
    with open(os.path.join(output_dir, "01_overview.json"), "w") as f:
        json.dump({k: (str(v) if v is not None else None) for k, v in r.items()}, f, indent=2, ensure_ascii=False)

    # 写入 Markdown 报告
    if report:
        report.h2("1. 总体概览")
        report.kv("分析周期", f"最近 {days} 天")
        report.blank()
        report.h3("请求统计")
        report.kv("总请求数", fmt_num(r['total_requests']))
        report.kv("成功", fmt_num(r['success_count']))
        report.kv("失败", fmt_num(r['fail_count']))
        report.kv("流式比例", fmt_pct(r['stream_pct']))
        report.blank()
        report.h3("Token 统计")
        report.kv("总 Token", fmt_num(r['total_tokens']))
        report.kv("总输入 Token", fmt_num(r['total_input_tokens']))
        report.kv("总输出 Token", fmt_num(r['total_output_tokens']))
        report.kv("缓存读取 Token", fmt_num(r['total_cache_read']))
        report.kv("缓存写入 Token", fmt_num(r['total_cache_write']))
        report.blank()
        report.h3("Token 分布")
        report.table(
            ["指标", "平均", "中位数", "P95", "P99", "最大值"],
            [["Token", fmt_num(r['avg_total']), fmt_num(r['median_total']),
              fmt_num(r['p95_total']), fmt_num(r['p99_total']), fmt_num(r['max_total'])]],
            ["---", ":---:", ":---:", ":---:", ":---:", ":---:"]
        )
        report.h3("费用统计")
        report.kv("总费用", f"${fmt_num(r['total_cost'])}")
        report.kv("平均费用", f"${fmt_num(r['avg_cost'])}")
        report.kv("输入费用", f"${fmt_num(r['total_input_cost'])}")
        report.kv("输出费用", f"${fmt_num(r['total_output_cost'])}")
        report.kv("缓存读取费用", f"${fmt_num(r['total_cache_read_cost'])}")
        report.kv("缓存写入费用", f"${fmt_num(r['total_cache_write_cost'])}")
        report.blank()
        report.h3("延迟统计")
        report.table(
            ["指标", "平均", "中位数", "P95", "最大值"],
            [["延迟(ms)", fmt_num(r['avg_latency_ms']), fmt_num(r['median_latency_ms']),
              fmt_num(r['p95_latency_ms']), fmt_num(r['max_latency_ms'])]],
            ["---", ":---:", ":---:", ":---:", ":---:"]
        )
        report.kv("缓存命中率", fmt_pct(r['cache_hit_rate_pct']))
        report.blank()
        if float(r['cache_hit_rate_pct'] or 0) == 0:
            report.text("> ⚠️ **缓存命中率为 0 的原因**: 当前请求量 Top 模型（如 minimax-m2.7、glm-5.1）的 `cache_pricing` 配置为 `None`，即这些模型不支持上下文缓存。支持缓存的模型（如 claude-*、gpt-5.4、glm-4.7）请求量极少。")
        report.hr()

    return r


# ─────────────────────────────────────────────
# 2. 上下文长度分布（桶状分布 + 百分位）
# ─────────────────────────────────────────────
def analyze_context_distribution(db, days, output_dir, report=None):
    section("2. 上下文长度分布")

    # 2a. 桶状分布
    subsection("输入 Token 区间分布")
    input_buckets = db.query("""
        SELECT
            CASE
                WHEN input_tokens < 512      THEN '0-512'
                WHEN input_tokens < 1024     THEN '512-1K'
                WHEN input_tokens < 2048     THEN '1K-2K'
                WHEN input_tokens < 4096     THEN '2K-4K'
                WHEN input_tokens < 8192     THEN '4K-8K'
                WHEN input_tokens < 16384    THEN '8K-16K'
                WHEN input_tokens < 32768    THEN '16K-32K'
                WHEN input_tokens < 65536    THEN '32K-64K'
                WHEN input_tokens < 131072   THEN '64K-128K'
                ELSE '128K+'
            END AS range_label,
            MIN(input_tokens) AS sort_key,
            COUNT(*) AS request_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct,
            SUM(SUM(input_tokens)) OVER() AS total_in_all
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY 1
        ORDER BY sort_key
    """ % days)

    print(f"  {'区间':<15} {'请求数':>12} {'占比':>8} {'累计占比':>8}")
    print(f"  {'-'*50}")
    cumulative = 0.0
    for row in input_buckets:
        cumulative += float(row['pct'])
        bar = "█" * max(1, int(float(row['pct']) / 2))
        print(f"  {row['range_label']:<15} {fmt_num(row['request_count']):>12} {fmt_pct(float(row['pct'])):>8} {fmt_pct(cumulative):>8}  {colored(bar, C.BLUE)}")

    # 2b. 总 Token 区间分布
    subsection("总 Token 区间分布")
    total_buckets = db.query("""
        SELECT
            CASE
                WHEN total_tokens < 1024     THEN '0-1K'
                WHEN total_tokens < 4096     THEN '1K-4K'
                WHEN total_tokens < 8192     THEN '4K-8K'
                WHEN total_tokens < 16384    THEN '8K-16K'
                WHEN total_tokens < 32768    THEN '16K-32K'
                WHEN total_tokens < 65536    THEN '32K-64K'
                WHEN total_tokens < 131072   THEN '64K-128K'
                ELSE '128K+'
            END AS range_label,
            MIN(total_tokens) AS sort_key,
            COUNT(*) AS request_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY 1
        ORDER BY sort_key
    """ % days)

    print(f"  {'区间':<15} {'请求数':>12} {'占比':>8} {'累计占比':>8}")
    print(f"  {'-'*50}")
    cumulative = 0.0
    for row in total_buckets:
        cumulative += float(row['pct'])
        bar = "█" * max(1, int(float(row['pct']) / 2))
        print(f"  {row['range_label']:<15} {fmt_num(row['request_count']):>12} {fmt_pct(float(row['pct'])):>8} {fmt_pct(cumulative):>8}  {colored(bar, C.GREEN)}")

    # 2c. 百分位详细分析
    subsection("百分位详细分析")
    percentiles = db.query("""
        SELECT
            'input_tokens' AS metric,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY input_tokens)::bigint AS p10,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY input_tokens)::bigint AS p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY input_tokens)::bigint AS p50,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY input_tokens)::bigint AS p75,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY input_tokens)::bigint AS p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY input_tokens)::bigint AS p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY input_tokens)::bigint AS p99,
            MAX(input_tokens) AS max_val,
            AVG(input_tokens)::bigint AS avg_val
        FROM usage_records WHERE created_at >= NOW() - INTERVAL '%s days'
        UNION ALL
        SELECT
            'output_tokens' AS metric,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY output_tokens)::bigint,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY output_tokens)::bigint,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY output_tokens)::bigint,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY output_tokens)::bigint,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY output_tokens)::bigint,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY output_tokens)::bigint,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY output_tokens)::bigint,
            MAX(output_tokens),
            AVG(output_tokens)::bigint
        FROM usage_records WHERE created_at >= NOW() - INTERVAL '%s days'
        UNION ALL
        SELECT
            'total_tokens' AS metric,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY total_tokens)::bigint,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_tokens)::bigint,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_tokens)::bigint,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_tokens)::bigint,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_tokens)::bigint,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_tokens)::bigint,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_tokens)::bigint,
            MAX(total_tokens),
            AVG(total_tokens)::bigint
        FROM usage_records WHERE created_at >= NOW() - INTERVAL '%s days'
    """ % (days, days, days))

    print(f"  {'指标':<15} {'P10':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P90':>10} {'P95':>10} {'P99':>10} {'Max':>12} {'Avg':>10}")
    print(f"  {'-'*107}")
    for row in percentiles:
        print(f"  {row['metric']:<15} {fmt_num(row['p10']):>10} {fmt_num(row['p25']):>10} {fmt_num(row['p50']):>10} {fmt_num(row['p75']):>10} {fmt_num(row['p90']):>10} {fmt_num(row['p95']):>10} {fmt_num(row['p99']):>10} {fmt_num(row['max_val']):>12} {fmt_num(row['avg_val']):>10}")

    # 保存 CSV
    with open(os.path.join(output_dir, "02_input_buckets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=input_buckets[0].keys())
        w.writeheader()
        w.writerows(input_buckets)

    with open(os.path.join(output_dir, "02_total_buckets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=total_buckets[0].keys())
        w.writeheader()
        w.writerows(total_buckets)

    # 写入 Markdown 报告
    if report:
        report.h2("2. 上下文长度分布")
        report.h3("输入 Token 区间分布")
        cum = 0.0
        ib_rows = []
        for row in input_buckets:
            cum += float(row['pct'])
            ib_rows.append([row['range_label'], fmt_num(row['request_count']),
                           fmt_pct(float(row['pct'])), fmt_pct(cum)])
        report.table(["区间", "请求数", "占比", "累计占比"], ib_rows,
                     ["---", ":---:", ":---:", ":---:"])

        report.h3("总 Token 区间分布")
        cum = 0.0
        tb_rows = []
        for row in total_buckets:
            cum += float(row['pct'])
            tb_rows.append([row['range_label'], fmt_num(row['request_count']),
                           fmt_pct(float(row['pct'])), fmt_pct(cum)])
        report.table(["区间", "请求数", "占比", "累计占比"], tb_rows,
                     ["---", ":---:", ":---:", ":---:"])

        report.h3("百分位详细分析")
        pct_rows = []
        for row in percentiles:
            pct_rows.append([row['metric'], fmt_num(row['p10']), fmt_num(row['p25']),
                            fmt_num(row['p50']), fmt_num(row['p75']), fmt_num(row['p90']),
                            fmt_num(row['p95']), fmt_num(row['p99']),
                            fmt_num(row['max_val']), fmt_num(row['avg_val'])])
        report.table(
            ["指标", "P10", "P25", "P50", "P75", "P90", "P95", "P99", "Max", "Avg"],
            pct_rows,
            ["---"] + [":---:"] * 9
        )
        report.image("08a_input_token_dist.png", "输入 Token 分布")
        report.image("08b_total_token_dist.png", "总 Token 分布")
        report.hr()

    return input_buckets, total_buckets, percentiles


# ─────────────────────────────────────────────
# 3. 按模型分析
# ─────────────────────────────────────────────
def analyze_by_model(db, days, output_dir, report=None):
    section("3. 按模型上下文分析")

    rows = db.query("""
        SELECT
            ur.model,
            m.context_window,
            m.max_input_tokens,
            m.max_output_tokens,
            m.cache_pricing IS NOT NULL AS has_cache_pricing,
            COUNT(*)                                                    AS request_count,
            AVG(ur.input_tokens)::bigint                                AS avg_input,
            AVG(ur.output_tokens)::bigint                               AS avg_output,
            AVG(ur.total_tokens)::bigint                                AS avg_total,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ur.total_tokens)::bigint  AS median_total,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ur.total_tokens)::bigint  AS p95_total,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ur.total_tokens)::bigint  AS p99_total,
            MAX(ur.total_tokens)                                        AS max_total,
            -- 上下文窗口使用率
            CASE WHEN m.context_window > 0
                THEN ROUND(100.0 * AVG(ur.total_tokens) / m.context_window, 2)
                ELSE NULL
            END AS avg_context_usage_pct,
            CASE WHEN m.context_window > 0
                THEN ROUND(100.0 * MAX(ur.total_tokens) / m.context_window, 2)
                ELSE NULL
            END AS max_context_usage_pct,
            -- 缓存
            SUM(ur.cache_read_tokens)                                   AS total_cache_read,
            SUM(ur.cache_write_tokens)                                  AS total_cache_write,
            SUM(ur.input_tokens)                                        AS total_input_sum,
            CASE WHEN SUM(ur.input_tokens) > 0
                THEN ROUND(100.0 * SUM(ur.cache_read_tokens) / SUM(ur.input_tokens), 2)
                ELSE 0
            END AS cache_hit_rate_pct,
            -- 费用
            SUM(ur.total_cost)::numeric(12,4)                           AS total_cost,
            AVG(ur.total_cost)::numeric(10,6)                           AS avg_cost,
            -- 延迟
            AVG(ur.response_time)::bigint                               AS avg_latency_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ur.response_time)::bigint AS p95_latency_ms
        FROM usage_records ur
        LEFT JOIN models m ON ur.model = m.model_id
        WHERE ur.created_at >= NOW() - INTERVAL '%s days'
        GROUP BY ur.model, m.context_window, m.max_input_tokens, m.max_output_tokens, m.cache_pricing
        ORDER BY request_count DESC
    """ % days)

    header = f"  {'模型':<30} {'请求数':>8} {'上下文窗口':>10} {'平均Token':>10} {'窗口使用%':>10} {'缓存命中%':>10} {'平均延迟':>8} {'总费用$':>12}"
    print(header)
    print(f"  {'-' * len(header)}")

    for row in rows:
        ctx_window = fmt_num(row['context_window']) if row['context_window'] else 'N/A'
        usage_pct = fmt_pct(float(row['avg_context_usage_pct'])) if row['avg_context_usage_pct'] is not None else 'N/A'
        cache_pct = fmt_pct(float(row['cache_hit_rate_pct'])) if row['cache_hit_rate_pct'] is not None else '0.00%'

        # 颜色标记上下文使用率
        if row['avg_context_usage_pct'] is not None:
            up = float(row['avg_context_usage_pct'])
            if up > 80:
                usage_pct = colored(usage_pct, C.RED)
            elif up > 50:
                usage_pct = colored(usage_pct, C.YELLOW)

        print(f"  {row['model']:<30} {fmt_num(row['request_count']):>8} {ctx_window:>10} {fmt_num(row['avg_total']):>10} {usage_pct:>20} {cache_pct:>10} {fmt_num(row['avg_latency_ms']):>8} {fmt_num(row['total_cost']):>12}")

    # 保存 CSV
    with open(os.path.join(output_dir, "03_by_model.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for row in rows:
            w.writerow({k: str(v) if v is not None else '' for k, v in row.items()})

    # 写入 Markdown 报告
    if report:
        report.h2("3. 按模型上下文分析")
        md_rows = []
        for row in rows:
            ctx_window = fmt_num(row['context_window']) if row['context_window'] else 'N/A'
            usage_pct = fmt_pct(float(row['avg_context_usage_pct'])) if row['avg_context_usage_pct'] is not None else 'N/A'
            cache_pct = fmt_pct(float(row['cache_hit_rate_pct'])) if row['cache_hit_rate_pct'] is not None else '0.00%'
            md_rows.append([row['model'], fmt_num(row['request_count']), ctx_window,
                           fmt_num(row['avg_total']), usage_pct, cache_pct,
                           fmt_num(row['avg_latency_ms']), f"${fmt_num(row['total_cost'])}"])
        report.table(
            ["模型", "请求数", "上下文窗口", "平均Token", "窗口使用%", "缓存命中%", "平均延迟(ms)", "总费用$"],
            md_rows,
            ["---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"]
        )
        report.hr()

    return rows


# ─────────────────────────────────────────────
# 4. 缓存命中率分析
# ─────────────────────────────────────────────
def analyze_cache(db, days, output_dir, report=None):
    section("4. 缓存命中率分析")

    # 4a. 按模型
    subsection("按模型缓存统计")
    cache_by_model = db.query("""
        SELECT
            model,
            COUNT(*)                                                    AS total_requests,
            COUNT(*) FILTER (WHERE cache_read_tokens > 0)              AS cache_hit_requests,
            COUNT(*) FILTER (WHERE cache_write_tokens > 0)             AS cache_write_requests,
            SUM(input_tokens)                                           AS total_input,
            SUM(cache_read_tokens)                                      AS total_cache_read,
            SUM(cache_write_tokens)                                     AS total_cache_write,
            CASE WHEN SUM(input_tokens) > 0
                THEN ROUND(100.0 * SUM(cache_read_tokens) / SUM(input_tokens), 2)
                ELSE 0
            END AS token_cache_hit_rate,
            CASE WHEN COUNT(*) > 0
                THEN ROUND(100.0 * COUNT(*) FILTER (WHERE cache_read_tokens > 0) / COUNT(*), 2)
                ELSE 0
            END AS request_cache_hit_rate,
            SUM(cache_read_cost)                                        AS total_cache_read_cost,
            SUM(cache_write_cost)                                       AS total_cache_write_cost,
            SUM(input_cost)                                             AS total_input_cost,
            CASE WHEN SUM(input_cost) > 0
                THEN ROUND(100.0 * SUM(cache_read_cost) / SUM(input_cost), 2)
                ELSE 0
            END AS cost_savings_pct
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY model
        ORDER BY total_requests DESC
    """ % days)

    print(f"  {'模型':<30} {'总请求':>8} {'缓存命中请求':>12} {'请求命中率':>10} {'Token命中率':>11} {'节省费用%':>10}")
    print(f"  {'-' * 90}")
    for row in cache_by_model:
        req_rate = float(row['request_cache_hit_rate'])
        tok_rate = float(row['token_cache_hit_rate'])
        cost_rate = float(row['cost_savings_pct']) if row['cost_savings_pct'] else 0

        req_str = fmt_pct(req_rate)
        tok_str = fmt_pct(tok_rate)
        cost_str = fmt_pct(cost_rate)

        # 颜色
        if tok_rate > 30:
            tok_str = colored(tok_str, C.GREEN)
        elif tok_rate > 10:
            tok_str = colored(tok_str, C.YELLOW)

        print(f"  {row['model']:<30} {fmt_num(row['total_requests']):>8} {fmt_num(row['cache_hit_requests']):>12} {req_str:>10} {tok_str:>21} {cost_str:>10}")

    # 4b. 每日缓存趋势
    subsection("每日缓存命中趋势")
    cache_daily = db.query("""
        SELECT
            DATE(created_at) AS date,
            COUNT(*) AS total_requests,
            SUM(input_tokens) AS total_input,
            SUM(cache_read_tokens) AS total_cache_read,
            CASE WHEN SUM(input_tokens) > 0
                THEN ROUND(100.0 * SUM(cache_read_tokens) / SUM(input_tokens), 2)
                ELSE 0
            END AS cache_hit_rate
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(created_at)
        ORDER BY date
    """ % days)

    print(f"  {'日期':<12} {'请求数':>10} {'输入Token':>14} {'缓存读取Token':>14} {'命中率':>8}")
    print(f"  {'-' * 65}")
    for row in cache_daily:
        rate = float(row['cache_hit_rate'])
        rate_str = fmt_pct(rate)
        if rate > 30:
            rate_str = colored(rate_str, C.GREEN)
        print(f"  {str(row['date']):<12} {fmt_num(row['total_requests']):>10} {fmt_num(row['total_input']):>14} {fmt_num(row['total_cache_read']):>14} {rate_str:>8}")

    # 保存
    with open(os.path.join(output_dir, "04_cache_by_model.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cache_by_model[0].keys())
        w.writeheader()
        for row in cache_by_model:
            w.writerow({k: str(v) if v is not None else '' for k, v in row.items()})

    # 写入 Markdown 报告
    if report:
        report.h2("4. 缓存命中率分析")
        report.h3("按模型缓存统计")
        cache_rows = []
        for row in cache_by_model:
            cache_rows.append([
                row['model'], fmt_num(row['total_requests']),
                fmt_num(row['cache_hit_requests']),
                fmt_pct(float(row['request_cache_hit_rate'])),
                fmt_pct(float(row['token_cache_hit_rate'])),
                fmt_pct(float(row['cost_savings_pct'])) if row['cost_savings_pct'] else '0.00%'
            ])
        report.table(
            ["模型", "总请求", "缓存命中请求", "请求命中率", "Token命中率", "节省费用%"],
            cache_rows,
            ["---", ":---:", ":---:", ":---:", ":---:", ":---:"]
        )

        report.h3("每日缓存命中趋势")
        daily_rows = []
        for row in cache_daily:
            daily_rows.append([
                str(row['date']), fmt_num(row['total_requests']),
                fmt_num(row['total_input']), fmt_num(row['total_cache_read']),
                fmt_pct(float(row['cache_hit_rate']))
            ])
        report.table(
            ["日期", "请求数", "输入Token", "缓存读取Token", "命中率"],
            daily_rows,
            ["---", ":---:", ":---:", ":---:", ":---:"]
        )
        report.image("08d_cache_trend.png", "缓存命中率趋势")
        report.hr()

    return cache_by_model, cache_daily


# ─────────────────────────────────────────────
# 5. 每日趋势
# ─────────────────────────────────────────────
def analyze_daily_trend(db, days, output_dir, report=None):
    section("5. 每日趋势")

    rows = db.query("""
        SELECT
            DATE(created_at) AS date,
            COUNT(*) AS request_count,
            COUNT(*) FILTER (WHERE status = 'success') AS success_count,
            AVG(input_tokens)::bigint AS avg_input,
            AVG(output_tokens)::bigint AS avg_output,
            AVG(total_tokens)::bigint AS avg_total,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_tokens)::bigint AS median_total,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_tokens)::bigint AS p95_total,
            MAX(total_tokens) AS max_total,
            SUM(total_cost)::numeric(12,4) AS daily_cost,
            AVG(response_time)::bigint AS avg_latency_ms
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(created_at)
        ORDER BY date
    """ % days)

    print(f"  {'日期':<12} {'请求数':>8} {'成功率':>8} {'平均Token':>10} {'中位数':>10} {'P95':>10} {'最大':>12} {'日费用$':>10} {'平均延迟':>8}")
    print(f"  {'-' * 95}")
    for row in rows:
        success_rate = round(100.0 * row['success_count'] / row['request_count'], 1) if row['request_count'] > 0 else 0
        sr_str = fmt_pct(success_rate)
        if success_rate >= 99:
            sr_str = colored(sr_str, C.GREEN)
        elif success_rate < 95:
            sr_str = colored(sr_str, C.RED)

        print(f"  {str(row['date']):<12} {fmt_num(row['request_count']):>8} {sr_str:>8} {fmt_num(row['avg_total']):>10} {fmt_num(row['median_total']):>10} {fmt_num(row['p95_total']):>10} {fmt_num(row['max_total']):>12} {fmt_num(row['daily_cost']):>10} {fmt_num(row['avg_latency_ms']):>8}")

    # 保存
    with open(os.path.join(output_dir, "05_daily_trend.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for row in rows:
            w.writerow({k: str(v) if v is not None else '' for k, v in row.items()})

    # 写入 Markdown 报告
    if report:
        report.h2("5. 每日趋势")
        trend_rows = []
        for row in rows:
            success_rate = round(100.0 * row['success_count'] / row['request_count'], 1) if row['request_count'] > 0 else 0
            trend_rows.append([
                str(row['date']), fmt_num(row['request_count']),
                fmt_pct(success_rate), fmt_num(row['avg_total']),
                fmt_num(row['median_total']), fmt_num(row['p95_total']),
                fmt_num(row['max_total']), f"${fmt_num(row['daily_cost'])}",
                fmt_num(row['avg_latency_ms'])
            ])
        report.table(
            ["日期", "请求数", "成功率", "平均Token", "中位数", "P95", "最大", "日费用$", "平均延迟(ms)"],
            trend_rows,
            ["---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"]
        )
        report.image("08c_daily_trend.png", "每日趋势")
        report.hr()

    return rows


# ─────────────────────────────────────────────
# 6. 用户维度分析
# ─────────────────────────────────────────────
def analyze_users(db, days, output_dir, report=None):
    section("6. 用户维度分析（Top 30）")

    rows = db.query("""
        SELECT
            user_id,
            COUNT(*) AS request_count,
            AVG(total_tokens)::bigint AS avg_tokens,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_tokens)::bigint AS median_tokens,
            MAX(total_tokens) AS max_tokens,
            SUM(total_tokens) AS sum_tokens,
            SUM(total_cost)::numeric(12,4) AS total_cost,
            COUNT(DISTINCT model) AS models_used,
            COUNT(*) FILTER (WHERE cache_read_tokens > 0) AS cache_hit_count,
            CASE WHEN COUNT(*) > 0
                THEN ROUND(100.0 * COUNT(*) FILTER (WHERE cache_read_tokens > 0) / COUNT(*), 2)
                ELSE 0
            END AS cache_hit_rate,
            COUNT(*) FILTER (WHERE total_tokens > 32000) AS long_ctx_count,
            CASE WHEN COUNT(*) > 0
                THEN ROUND(100.0 * COUNT(*) FILTER (WHERE total_tokens > 32000) / COUNT(*), 2)
                ELSE 0
            END AS long_ctx_pct
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY user_id
        HAVING COUNT(*) >= 5
        ORDER BY sum_tokens DESC
        LIMIT 30
    """ % days)

    print(f"  {'用户ID':<20} {'请求数':>8} {'平均Token':>10} {'总Token':>12} {'使用模型':>8} {'缓存命中率':>10} {'长上下文%':>10} {'总费用$':>10}")
    print(f"  {'-' * 100}")
    for row in rows:
        print(f"  {row['user_id']:<20} {fmt_num(row['request_count']):>8} {fmt_num(row['avg_tokens']):>10} {fmt_num(row['sum_tokens']):>12} {row['models_used']:>8} {fmt_pct(float(row['cache_hit_rate'])):>10} {fmt_pct(float(row['long_ctx_pct'])):>10} {fmt_num(row['total_cost']):>10}")

    with open(os.path.join(output_dir, "06_user_analysis.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for row in rows:
            w.writerow({k: str(v) if v is not None else '' for k, v in row.items()})

    # 写入 Markdown 报告
    if report:
        report.h2("6. 用户维度分析 (Top 30)")
        user_rows = []
        for row in rows:
            user_rows.append([
                row['user_id'], fmt_num(row['request_count']),
                fmt_num(row['avg_tokens']), fmt_num(row['sum_tokens']),
                row['models_used'], fmt_pct(float(row['cache_hit_rate'])),
                fmt_pct(float(row['long_ctx_pct'])), f"${fmt_num(row['total_cost'])}"
            ])
        report.table(
            ["用户ID", "请求数", "平均Token", "总Token", "使用模型", "缓存命中率", "长上下文%", "总费用$"],
            user_rows,
            ["---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"]
        )
        report.hr()

    return rows


# ─────────────────────────────────────────────
# 7. 上下文使用率分布（相对于模型窗口大小）
# ─────────────────────────────────────────────
def analyze_context_usage_ratio(db, days, output_dir, report=None):
    section("7. 上下文窗口使用率分布")

    rows = db.query("""
        SELECT
            ur.model,
            m.context_window,
            CASE
                WHEN m.context_window IS NULL OR m.context_window = 0 THEN 'unknown'
                WHEN 100.0 * ur.total_tokens / m.context_window < 1    THEN 'lt1pct'
                WHEN 100.0 * ur.total_tokens / m.context_window < 5    THEN '1-5pct'
                WHEN 100.0 * ur.total_tokens / m.context_window < 10   THEN '5-10pct'
                WHEN 100.0 * ur.total_tokens / m.context_window < 25   THEN '10-25pct'
                WHEN 100.0 * ur.total_tokens / m.context_window < 50   THEN '25-50pct'
                WHEN 100.0 * ur.total_tokens / m.context_window < 75   THEN '50-75pct'
                WHEN 100.0 * ur.total_tokens / m.context_window < 90   THEN '75-90pct'
                ELSE '90-100pct'
            END AS usage_bucket,
            COUNT(*) AS cnt
        FROM usage_records ur
        LEFT JOIN models m ON ur.model = m.model_id
        WHERE ur.created_at >= NOW() - INTERVAL '%s days'
          AND m.context_window IS NOT NULL AND m.context_window > 0
        GROUP BY ur.model, m.context_window, 3
        ORDER BY ur.model, MIN(100.0 * ur.total_tokens / m.context_window)
    """ % days)

    # 聚合每个模型
    model_buckets = defaultdict(lambda: defaultdict(int))
    model_totals = defaultdict(int)
    for row in rows:
        model_buckets[row['model']][row['usage_bucket']] += row['cnt']
        model_totals[row['model']] += row['cnt']

    bucket_order = ['lt1pct', '1-5pct', '5-10pct', '10-25pct', '25-50pct', '50-75pct', '75-90pct', '90-100pct']
    bucket_display = ['<1%', '1-5%', '5-10%', '10-25%', '25-50%', '50-75%', '75-90%', '90-100%']

    print(f"  {'模型':<30}", end="")
    for b in bucket_display:
        print(f" {b:>8}", end="")
    print(f" {'总数':>8}")
    print(f"  {'-' * (30 + 9 * (len(bucket_display) + 1))}")

    for model in sorted(model_totals.keys(), key=lambda m: model_totals[m], reverse=True):
        total = model_totals[model]
        print(f"  {model:<30}", end="")
        for b in bucket_order:
            cnt = model_buckets[model].get(b, 0)
            pct = 100.0 * cnt / total if total > 0 else 0
            if pct > 50:
                print(f" {colored(f'{pct:.1f}%', C.RED):>8}", end="")
            elif pct > 25:
                print(f" {colored(f'{pct:.1f}%', C.YELLOW):>8}", end="")
            else:
                print(f" {pct:.1f}%".rjust(8), end="")
        print(f" {fmt_num(total):>8}")

    # 写入 Markdown 报告
    if report:
        report.h2("7. 上下文窗口使用率分布")
        ratio_rows = []
        for model in sorted(model_totals.keys(), key=lambda m: model_totals[m], reverse=True):
            total = model_totals[model]
            cells = [model]
            for b in bucket_order:
                cnt = model_buckets[model].get(b, 0)
                pct = 100.0 * cnt / total if total > 0 else 0
                cells.append(f"{pct:.1f}%")
            cells.append(fmt_num(total))
            ratio_rows.append(cells)
        report.table(
            ["模型"] + bucket_display + ["总数"],
            ratio_rows,
            ["---"] + [":---:"] * (len(bucket_display) + 1)
        )
        report.image("08e_model_overview.png", "模型概览")
        report.hr()

    return rows


# ─────────────────────────────────────────────
# 8. 可视化图表
# ─────────────────────────────────────────────
def generate_plots(output_dir, input_buckets, total_buckets, cache_daily, daily_trend, model_data):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print(colored("\n⚠️  matplotlib 未安装，跳过图表生成。安装: pip install matplotlib", C.YELLOW))
        return

    section("8. 生成可视化图表")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['figure.dpi'] = 150

    # ---- 8a. 输入 Token 分布柱状图 ----
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [r['range_label'] for r in input_buckets]
    counts = [r['request_count'] for r in input_buckets]
    colors = plt.cm.Blues([0.3 + 0.6 * i / len(labels) for i in range(len(labels))])
    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Input Token Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Token Range', fontsize=12)
    ax.set_ylabel('Request Count', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_dir, "08a_input_token_dist.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(colored("  ✓ 08a_input_token_dist.png", C.GREEN))

    # ---- 8b. 总 Token 分布柱状图 ----
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [r['range_label'] for r in total_buckets]
    counts = [r['request_count'] for r in total_buckets]
    colors = plt.cm.Greens([0.3 + 0.6 * i / len(labels) for i in range(len(labels))])
    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Total Token Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Token Range', fontsize=12)
    ax.set_ylabel('Request Count', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_dir, "08b_total_token_dist.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(colored("  ✓ 08b_total_token_dist.png", C.GREEN))

    # ---- 8c. 每日请求量 + 平均 Token 趋势 ----
    if daily_trend:
        fig, ax1 = plt.subplots(figsize=(18, 7))
        dates = [str(r['date']) for r in daily_trend]
        req_counts = [r['request_count'] for r in daily_trend]
        avg_tokens = [r['avg_total'] for r in daily_trend]

        color1 = '#2196F3'
        color2 = '#FF5722'
        x = range(len(dates))
        ax1.bar(x, req_counts, color=color1, alpha=0.7, label='Request Count', width=0.8)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Request Count', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))

        ax2 = ax1.twinx()
        ax2.plot(x, avg_tokens, color=color2, linewidth=2, markersize=3, marker='o', label='Avg Total Tokens')
        ax2.set_ylabel('Avg Total Tokens', color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))

        # 只显示部分日期标签避免重叠
        step = max(1, len(dates) // 15)
        ax1.set_xticks([i for i in x if i % step == 0])
        ax1.set_xticklabels([dates[i] for i in x if i % step == 0], rotation=45, ha='right', fontsize=9)

        ax1.set_title('Daily Request Count & Average Tokens', fontsize=16, fontweight='bold', pad=15)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(output_dir, "08c_daily_trend.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(colored("  ✓ 08c_daily_trend.png", C.GREEN))

    # ---- 8d. 缓存命中率趋势 ----
    if cache_daily:
        fig, ax = plt.subplots(figsize=(18, 7))
        dates = [str(r['date']) for r in cache_daily]
        hit_rates = [float(r['cache_hit_rate']) for r in cache_daily]
        cache_reads = [r['total_cache_read'] for r in cache_daily]
        x = range(len(dates))

        ax.bar(x, cache_reads, color='#4CAF50', alpha=0.6, label='Cache Read Tokens', width=0.8)
        ax.set_ylabel('Cache Read Tokens', fontsize=12)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000000:.1f}M' if x >= 1000000 else f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))

        ax2 = ax.twinx()
        ax2.plot(x, hit_rates, color='#E91E63', linewidth=2, markersize=4, marker='s', label='Cache Hit Rate %')
        ax2.set_ylabel('Cache Hit Rate (%)', color='#E91E63', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#E91E63')
        ax2.set_ylim(0, max(max(hit_rates) * 1.2 + 1, 5))

        step = max(1, len(dates) // 15)
        ax.set_xticks([i for i in x if i % step == 0])
        ax.set_xticklabels([dates[i] for i in x if i % step == 0], rotation=45, ha='right', fontsize=9)

        ax.set_title('Daily Cache Hit Rate Trend', fontsize=16, fontweight='bold', pad=15)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(output_dir, "08d_cache_trend.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(colored("  ✓ 08d_cache_trend.png", C.GREEN))

    # ---- 8e. 各模型请求量 + 平均Token ----
    if model_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        models = [r['model'] for r in model_data[:10]]
        req_counts = [r['request_count'] for r in model_data[:10]]
        avg_tokens_list = [r['avg_total'] for r in model_data[:10]]

        # 饼图: 请求占比 - 使用 legend 替代标签避免重叠
        colors_pie = plt.cm.Set3([i/len(models) for i in range(len(models))])
        wedges, texts, autotexts = ax1.pie(
            req_counts, labels=None, autopct='%1.1f%%',
            colors=colors_pie, pctdistance=0.8, startangle=90,
            textprops={'fontsize': 9}
        )
        # 放到侧边图例
        ax1.legend(wedges, [f'{m} ({c:,})' for m, c in zip(models, req_counts)],
                   title='Models', loc='center left', bbox_to_anchor=(1.0, 0.5),
                   fontsize=9, title_fontsize=10)
        ax1.set_title('Request Share by Model', fontsize=14, fontweight='bold', pad=15)

        # 横向柱状图: 平均 Token
        y_pos = range(len(models))
        bars = ax2.barh(y_pos, avg_tokens_list, color=colors_pie, edgecolor='white', linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(models, fontsize=10)
        ax2.set_xlabel('Average Total Tokens', fontsize=12)
        ax2.set_title('Average Tokens per Request by Model', fontsize=14, fontweight='bold', pad=15)
        ax2.invert_yaxis()
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
        for bar, val in zip(bars, avg_tokens_list):
            ax2.text(bar.get_width() + max(avg_tokens_list)*0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:,}', va='center', fontsize=9)

        plt.subplots_adjust(wspace=0.4)
        plt.savefig(os.path.join(output_dir, "08e_model_overview.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(colored("  ✓ 08e_model_overview.png", C.GREEN))

    print(colored(f"\n  所有图表已保存到: {output_dir}", C.GREEN))


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="上下文分布全面分析工具")
    parser.add_argument("--days", type=int, default=90, help="分析最近 N 天数据（默认 90）")
    parser.add_argument("--module", type=str, default="all",
                        choices=["all", "overview", "distribution", "model", "cache", "trend", "users", "ratio"],
                        help="只运行指定模块")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    args = parser.parse_args()

    # 输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "outputs", f"context_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    print(colored(f"\n🚀 上下文分布全面分析工具", C.BOLD))
    print(colored(f"   分析天数: {args.days} 天", C.BLUE))
    print(colored(f"   输出目录: {output_dir}", C.BLUE))
    print(colored(f"   分析模块: {args.module}", C.BLUE))

    # 创建 Markdown 报告
    report = Report(title=f"上下文分布分析报告（最近 {args.days} 天）", output_dir=output_dir)
    report.kv("分析天数", f"{args.days} 天")
    report.kv("输出目录", output_dir)
    report.kv("分析模块", args.module)
    report.hr()

    db = DB()
    db.connect()

    try:
        # 运行各模块
        overview_data = None
        input_buckets = total_buckets = percentiles = None
        model_data = None
        cache_by_model = cache_daily = None
        daily_trend = None

        if args.module in ("all", "overview"):
            overview_data = analyze_overview(db, args.days, output_dir, report)

        if args.module in ("all", "distribution"):
            input_buckets, total_buckets, percentiles = analyze_context_distribution(db, args.days, output_dir, report)

        if args.module in ("all", "model"):
            model_data = analyze_by_model(db, args.days, output_dir, report)

        if args.module in ("all", "cache"):
            cache_by_model, cache_daily = analyze_cache(db, args.days, output_dir, report)

        if args.module in ("all", "trend"):
            daily_trend = analyze_daily_trend(db, args.days, output_dir, report)

        if args.module in ("all", "users"):
            analyze_users(db, args.days, output_dir, report)

        if args.module in ("all", "ratio"):
            analyze_context_usage_ratio(db, args.days, output_dir, report)

        # 生成图表
        if not args.no_plot and args.module == "all":
            # 需要重新查询绘图数据（如果模块被跳过）
            if input_buckets is None:
                input_buckets, total_buckets, _ = analyze_context_distribution(db, args.days, output_dir)
            if cache_daily is None:
                _, cache_daily = analyze_cache(db, args.days, output_dir)
            if daily_trend is None:
                daily_trend = analyze_daily_trend(db, args.days, output_dir)
            if model_data is None:
                model_data = analyze_by_model(db, args.days, output_dir)
            generate_plots(output_dir, input_buckets, total_buckets, cache_daily, daily_trend, model_data)

        # 保存 Markdown 报告
        report_path = report.save("context_analysis_report.md")

        section("✅ 分析完成")
        print(f"\n  📁 所有结果已保存到: {colored(output_dir, C.GREEN)}")
        print(f"\n  文件列表:")
        for f in sorted(os.listdir(output_dir)):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"    {f:<45} ({size:,} bytes)")

    except Exception as e:
        print(colored(f"\n❌ 错误: {e}", C.RED))
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()
