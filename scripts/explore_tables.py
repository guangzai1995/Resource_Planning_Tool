#!/usr/bin/env python3
"""
探索数据库表结构和样本数据
在有数据库连接的服务器上运行:
  python3 explore_tables.py

结果会打印到终端，把输出贴回给我即可。
"""
import psycopg2
import json
from datetime import datetime

DB_CONFIG = {
    "host": "10.88.202.214",
    "port": 5432,
    "dbname": "appdb",
    "user": "appuser",
    "password": "op3mF+-LKez3AN-mQP9D",
}

def query(conn, sql, desc=""):
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    cur.close()
    return cols, rows

def print_table(title, cols, rows, max_rows=5):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    if not cols:
        print("  (无结果)")
        return
    # 打印列头
    header = " | ".join(f"{c:<25}" for c in cols)
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for i, row in enumerate(rows[:max_rows]):
        line = " | ".join(f"{str(v)[:25]:<25}" for v in row)
        print(f"  {line}")
    if len(rows) > max_rows:
        print(f"  ... 共 {len(rows)} 行，仅显示前 {max_rows} 行")
    print()

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    print(f"✓ 已连接到数据库 {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"  查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =====================================================
    # 1. usage_records 完整表结构
    # =====================================================
    cols, rows = query(conn, """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'usage_records'
        ORDER BY ordinal_position
    """)
    print_table("usage_records 表结构", cols, rows, max_rows=50)

    # =====================================================
    # 2. usage_records 样本数据 (最近 5 条)
    # =====================================================
    cols, rows = query(conn, """
        SELECT * FROM usage_records ORDER BY created_at DESC LIMIT 5
    """)
    print_table("usage_records 样本数据 (最近5条)", cols, rows, max_rows=5)

    # =====================================================
    # 3. models 完整表结构
    # =====================================================
    cols, rows = query(conn, """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'models'
        ORDER BY ordinal_position
    """)
    print_table("models 表结构", cols, rows, max_rows=30)

    # =====================================================
    # 4. models 全部数据
    # =====================================================
    cols, rows = query(conn, """
        SELECT * FROM models ORDER BY model_id
    """)
    print_table("models 全部数据", cols, rows, max_rows=50)

    # =====================================================
    # 5. api_request_performance_traces 表结构
    # =====================================================
    cols, rows = query(conn, """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'api_request_performance_traces'
        ORDER BY ordinal_position
    """)
    print_table("api_request_performance_traces 表结构", cols, rows, max_rows=80)

    # =====================================================
    # 6. api_request_performance_traces 样本 (最近 3 条)
    # =====================================================
    cols, rows = query(conn, """
        SELECT * FROM api_request_performance_traces ORDER BY created_at DESC LIMIT 3
    """)
    print_table("api_request_performance_traces 样本数据", cols, rows, max_rows=3)

    # =====================================================
    # 7. api_request_performance_attempts 表结构
    # =====================================================
    cols, rows = query(conn, """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'api_request_performance_attempts'
        ORDER BY ordinal_position
    """)
    print_table("api_request_performance_attempts 表结构", cols, rows, max_rows=30)

    # =====================================================
    # 8. 查看 usage_records 中是否有缓存相关字段
    # =====================================================
    # 搜索所有表中与 cache 相关的列
    cols, rows = query(conn, """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE column_name ILIKE '%cache%'
           OR column_name ILIKE '%cached%'
           OR column_name ILIKE '%prompt_tokens%'
           OR column_name ILIKE '%reasoning%'
           OR column_name ILIKE '%completion_tokens%'
        ORDER BY table_name, column_name
    """)
    print_table("所有与 cache/token 明细相关的列", cols, rows, max_rows=50)

    # =====================================================
    # 9. usage_records 中 model 分布统计
    # =====================================================
    cols, rows = query(conn, """
        SELECT model, COUNT(*) as cnt,
               AVG(input_tokens)::int as avg_in,
               AVG(output_tokens)::int as avg_out,
               AVG(total_tokens)::int as avg_total
        FROM usage_records
        WHERE created_at >= NOW() - INTERVAL '30 days'
        GROUP BY model
        ORDER BY cnt DESC
        LIMIT 20
    """)
    print_table("usage_records 最近30天 model 分布", cols, rows, max_rows=20)

    # =====================================================
    # 10. 检查 usage_records 中是否有 JSONB / JSON 字段
    # =====================================================
    cols, rows = query(conn, """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'usage_records'
          AND data_type IN ('json', 'jsonb', 'text')
        ORDER BY ordinal_position
    """)
    print_table("usage_records 中 JSON/TEXT 类型字段", cols, rows, max_rows=20)

    # =====================================================
    # 11. 如果有 JSONB 字段，取一条样本看结构
    # =====================================================
    jsonb_cols = [r[0] for r in rows if r[1] in ('json', 'jsonb')]
    if jsonb_cols:
        for col in jsonb_cols[:3]:
            _, sample = query(conn, f"""
                SELECT {col} FROM usage_records
                WHERE {col} IS NOT NULL
                ORDER BY created_at DESC LIMIT 1
            """)
            if sample and sample[0][0]:
                print(f"\n{'='*80}")
                print(f"  usage_records.{col} 样本值")
                print(f"{'='*80}")
                val = sample[0][0]
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except:
                        pass
                print(json.dumps(val, indent=2, ensure_ascii=False, default=str)[:3000])
                print()

    # =====================================================
    # 12. 检查 usage_event_consumptions 表 (有缓存信息)
    # =====================================================
    cols, rows = query(conn, """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'usage_event_consumptions'
        ORDER BY ordinal_position
    """)
    print_table("usage_event_consumptions 表结构", cols, rows, max_rows=30)

    cols, rows = query(conn, """
        SELECT * FROM usage_event_consumptions
        ORDER BY created_at DESC LIMIT 3
    """)
    print_table("usage_event_consumptions 样本数据", cols, rows, max_rows=3)

    # =====================================================
    # 13. 检查 account_health_snapshots 表
    # =====================================================
    cols, rows = query(conn, """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'account_health_snapshots'
        ORDER BY ordinal_position
    """)
    print_table("account_health_snapshots 表结构", cols, rows, max_rows=40)

    conn.close()
    print("\n✓ 查询完成，请将以上输出贴回给我！")

if __name__ == "__main__":
    main()
