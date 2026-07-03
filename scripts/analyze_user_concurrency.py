#!/usr/bin/env python3
"""
用户数 & 并发使用统计工具
========================
复用 analyze_context.py 的数据库连接配置，基于 usage_records 表统计：
  1. 总用户数（全量去重 + 各时间窗口活跃用户）
  2. 当前正在使用的用户数（最近若干分钟有请求）
  3. 历史并发峰值（按 1 分钟 / 5 分钟滑动窗口，同一窗口内去重用户数最大值）
  4. 最近 24 小时每小时活跃用户数（日内并发规律）
  5. 最近 7 天每日活跃用户数（DAU）

用法:
  python3 analyze_user_concurrency.py            # 默认统计
  python3 analyze_user_concurrency.py --save     # 额外保存 JSON 结果
"""
import os
import sys
import json
import argparse
from datetime import timedelta

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("❌ 需要安装 psycopg2-binary: pip install psycopg2-binary")
    sys.exit(1)

# ─────────────────────────────────────────────
# 数据库配置（与 analyze_context.py 一致）
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
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def colored(text, color):
    return f"{color}{text}{C.NC}"


def section(title):
    print(f"\n{colored('=' * 72, C.CYAN)}")
    print(colored(f"  {title}", C.BOLD))
    print(colored('=' * 72, C.CYAN))


def subsection(title):
    print(f"\n  {colored('▸ ' + title, C.YELLOW)}")


def fmt(n):
    return f"{n:,}" if n is not None else "N/A"


def to_cn(ts):
    """数据库时间(timestamptz, UTC) 转北京时间字符串 (UTC+8)"""
    if ts is None:
        return "N/A"
    return (ts + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────
# 查询
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="用户数 & 并发使用统计")
    parser.add_argument("--save", action="store_true", help="保存 JSON 结果到 scripts 目录")
    args = parser.parse_args()

    print(colored(f"\n🚀 用户数 & 并发使用统计", C.BOLD))
    print(colored(f"   数据库: {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}", C.BLUE))
    print(colored(f"   注: 以下时间均为北京时间 (UTC+8)", C.BLUE))

    result = {"db_now": None}

    conn = psycopg2.connect(**DB_CONFIG)
    print(colored("✓ 数据库已连接", C.GREEN))
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ── 数据概况 ──
    section("0. 数据概况")
    cur.execute("""
        SELECT COUNT(*) AS total_rows,
               COUNT(DISTINCT user_id) AS distinct_users,
               MIN(created_at) AS min_ts,
               MAX(created_at) AS max_ts,
               NOW() AS db_now
        FROM usage_records
    """)
    o = cur.fetchone()
    result["db_now"] = to_cn(o["db_now"])
    print(f"  总请求数:        {fmt(o['total_rows'])}")
    print(f"  全量去重用户数:  {colored(fmt(o['distinct_users']), C.GREEN)}")
    print(f"  数据起始:        {to_cn(o['min_ts'])}")
    print(f"  最新请求:        {to_cn(o['max_ts'])}")
    print(f"  数据库当前时间:  {to_cn(o['db_now'])}")
    result["overview"] = {
        "total_rows": o["total_rows"],
        "distinct_users_all_time": o["distinct_users"],
        "min_ts": to_cn(o["min_ts"]),
        "max_ts": to_cn(o["max_ts"]),
    }

    # ── 1. 各时间窗口活跃用户数 ──
    section("1. 用户数统计（各时间窗口活跃用户 / 去重）")
    cur.execute("""
        SELECT
            COUNT(DISTINCT user_id) AS all_users,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '5 minutes')  AS m5,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '10 minutes') AS m10,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '30 minutes') AS m30,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '1 hour')     AS h1,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '6 hours')    AS h6,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '1 day')      AS d1,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days')     AS d7,
            COUNT(DISTINCT user_id) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days')    AS d30
        FROM usage_records
    """)
    w = cur.fetchone()
    windows = [
        ("最近  5 分钟 (当前正在使用)", w["m5"]),
        ("最近 10 分钟",                w["m10"]),
        ("最近 30 分钟",                w["m30"]),
        ("最近  1 小时",                w["h1"]),
        ("最近  6 小时",                w["h6"]),
        ("最近  1 天",                  w["d1"]),
        ("最近  7 天",                  w["d7"]),
        ("最近 30 天",                  w["d30"]),
        ("全量(历史累计去重)",          w["all_users"]),
    ]
    print(f"  {'时间窗口':<32} {'去重用户数':>12}")
    print(f"  {'-' * 48}")
    for label, val in windows:
        marker = colored("  ◀ 当前并发", C.RED) if "当前正在使用" in label else ""
        print(f"  {label:<32} {colored(fmt(val), C.GREEN):>12}{marker}")
    result["active_users_by_window"] = {label: val for label, val in windows}

    # ── 2. 历史并发峰值（按时间桶去重用户数）──
    section("2. 历史并发峰值（同一时间窗口内「同时在用」的最大用户数）")

    def peak_by_bucket(seconds, label):
        """按 N 秒对齐的时间桶统计去重用户数，取 Top 峰值"""
        cur.execute("""
            WITH buckets AS (
                SELECT to_timestamp(floor(extract(epoch from created_at) / %s) * %s) AS bucket_start,
                       COUNT(DISTINCT user_id) AS users
                FROM usage_records
                GROUP BY 1
            )
            SELECT bucket_start, users
            FROM buckets
            ORDER BY users DESC, bucket_start DESC
            LIMIT 10
        """, (seconds, seconds))
        rows = cur.fetchall()
        print(f"\n  {colored('▸ ' + label, C.YELLOW)}")
        print(f"    {'排名':<6} {'并发用户数':>10}  {'发生时间(北京时间)':<22}")
        print(f"    {'-' * 44}")
        for i, r in enumerate(rows, 1):
            tag = colored(" ← 峰值", C.RED) if i == 1 else ""
            print(f"    #{i:<5} {colored(fmt(r['users']), C.GREEN):>10}  {to_cn(r['bucket_start']):<22}{tag}")
        return [{"rank": i, "users": r["users"], "ts": to_cn(r["bucket_start"])} for i, r in enumerate(rows, 1)]

    result["peak_1min"] = peak_by_bucket(60, "1 分钟窗口（细粒度，瞬时并发）")
    result["peak_5min"] = peak_by_bucket(300, "5 分钟窗口（粗粒度，会话级并发）")

    # ── 3. 最近 24 小时每小时活跃用户数（日内并发曲线）──
    section("3. 最近 24 小时 — 每小时活跃用户数（日内并发规律）")
    cur.execute("""
        SELECT date_trunc('hour', created_at) AS hr,
               COUNT(DISTINCT user_id) AS users,
               COUNT(*) AS reqs
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        GROUP BY 1
        ORDER BY hr
    """)
    hourly = cur.fetchall()
    max_u = max((h["users"] for h in hourly), default=0) or 1
    print(f"  {'小时(北京)':<22} {'活跃用户':>8} {'请求数':>9}  分布")
    print(f"  {'-' * 60}")
    for h in hourly:
        bar_len = int(30 * h["users"] / max_u)
        print(f"  {to_cn(h['hr'])[:16]:<22} {fmt(h['users']):>8} {fmt(h['reqs']):>9}  {colored('█' * max(1, bar_len), C.MAGENTA)}")
    result["hourly_24h"] = [{"hour": to_cn(h["hr"]), "users": h["users"], "reqs": h["reqs"]} for h in hourly]

    # ── 4. 最近 7 天每日活跃用户数（DAU）──
    section("4. 最近 7 天 — 每日活跃用户数 (DAU)")
    cur.execute("""
        SELECT DATE(created_at) AS d,
               COUNT(DISTINCT user_id) AS users,
               COUNT(*) AS reqs
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '7 days'
        GROUP BY 1
        ORDER BY d
    """)
    daily = cur.fetchall()
    max_d = max((d["users"] for d in daily), default=0) or 1
    print(f"  {'日期':<14} {'活跃用户(DAU)':>14} {'请求数':>9}  分布")
    print(f"  {'-' * 60}")
    for d in daily:
        bar_len = int(30 * d["users"] / max_d)
        print(f"  {to_cn(d['d'])[:10]:<14} {fmt(d['users']):>14} {fmt(d['reqs']):>9}  {colored('█' * max(1, bar_len), C.CYAN)}")
    result["dau_7d"] = [{"date": to_cn(d["d"]), "users": d["users"], "reqs": d["reqs"]} for d in daily]

    cur.close()
    conn.close()
    print(colored("\n✓ 数据库连接已关闭", C.GREEN))

    # ── 汇总 ──
    section("✅ 汇总")
    print(f"  • {colored('总用户数(历史去重)', C.BOLD)}: {colored(fmt(w['all_users']), C.GREEN)} 人")
    print(f"  • {colored('当前正在使用(最近5分钟)', C.BOLD)}: {colored(fmt(w['m5']), C.RED)} 人")
    print(f"  • {colored('最近1天活跃(DAU)', C.BOLD)}: {fmt(w['d1'])} 人")
    print(f"  • {colored('历史1分钟并发峰值', C.BOLD)}: {colored(fmt(result['peak_1min'][0]['users']), C.RED)} 人"
          f"  @ {result['peak_1min'][0]['ts']}")
    print(f"  • {colored('历史5分钟并发峰值', C.BOLD)}: {colored(fmt(result['peak_5min'][0]['users']), C.RED)} 人"
          f"  @ {result['peak_5min'][0]['ts']}")
    print(colored("\n  说明: 由于 LLM 推理请求是异步的，没有严格的「在线会话」概念，", C.BLUE))
    print(colored("        这里用「同一时间窗口内有请求的去重用户数」近似并发使用人数。", C.BLUE))
    print(colored("        1 分钟窗口 ≈ 瞬时并发;5 分钟窗口 ≈ 一次会话级的并发。", C.BLUE))

    if args.save:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(script_dir, "user_concurrency_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(colored(f"\n  ✓ JSON 报告已保存: {out}", C.GREEN))


if __name__ == "__main__":
    main()
