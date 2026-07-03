#!/usr/bin/env python3
"""
导出 usage_records 表结构 + 10 条样例数据
==========================================
用法:
  python3 scripts/export_sample_data.py
  python3 scripts/export_sample_data.py --table models        # 导出其他表
  python3 scripts/export_sample_data.py --limit 20            # 导出 20 条
  python3 scripts/export_sample_data.py --output my_dump.sql  # 指定输出文件
"""
import argparse
import os
import sys
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("❌ 需要安装 psycopg2-binary: pip install psycopg2-binary")
    sys.exit(1)

# 数据库配置（与 analyze_context.py 保持一致）
DB_CONFIG = {
    "host": os.environ.get("PG_HOST", "10.88.202.214"),
    "port": int(os.environ.get("PG_PORT", "5432")),
    "dbname": os.environ.get("PG_DATABASE", "appdb"),
    "user": os.environ.get("PG_USER", "appuser"),
    "password": os.environ.get("PG_PASSWORD", "op3mF+-LKez3AN-mQP9D"),
}


def get_table_ddl(conn, table_name):
    """获取建表语句（列定义 + 注释）"""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 1. 获取列信息
    cur.execute("""
        SELECT
            c.column_name,
            c.data_type,
            c.character_maximum_length,
            c.numeric_precision,
            c.numeric_scale,
            c.is_nullable,
            c.column_default,
            pgd.description AS column_comment
        FROM information_schema.columns c
        LEFT JOIN pg_catalog.pg_statio_all_tables st
            ON c.table_schema = st.schemaname AND c.table_name = st.relname
        LEFT JOIN pg_catalog.pg_description pgd
            ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
        WHERE c.table_schema = 'public'
          AND c.table_name = %s
        ORDER BY c.ordinal_position
    """, (table_name,))
    columns = cur.fetchall()

    if not columns:
        print(f"❌ 表 '{table_name}' 不存在或无列信息")
        cur.close()
        return None

    # 2. 获取表注释
    cur.execute("""
        SELECT obj_description(
            (quote_ident(table_schema) || '.' || quote_ident(table_name))::regclass,
            'pg_class'
        ) AS table_comment
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = %s
    """, (table_name,))
    table_comment_row = cur.fetchone()
    table_comment = table_comment_row['table_comment'] if table_comment_row else None

    cur.close()

    # 3. 拼接 DDL
    lines = []
    if table_comment:
        lines.append(f"-- 表注释: {table_comment}")
    lines.append(f"CREATE TABLE {table_name} (")

    col_defs = []
    for col in columns:
        dtype = col['data_type']
        # 类型映射
        if col['character_maximum_length']:
            dtype += f"({col['character_maximum_length']})"
        elif dtype == 'numeric' and col['numeric_precision']:
            dtype += f"({col['numeric_precision']},{col['numeric_scale'] or 0})"
        elif dtype == 'timestamp without time zone':
            dtype = 'TIMESTAMP'
        elif dtype == 'timestamp with time zone':
            dtype = 'TIMESTAMPTZ'

        nullable = "" if col['is_nullable'] == 'YES' else " NOT NULL"
        default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
        comment = f"  -- {col['column_comment']}" if col['column_comment'] else ""

        col_defs.append(f"    {col['column_name']} {dtype}{nullable}{default}{comment}")

    lines.append(",\n".join(col_defs))
    lines.append(");")

    return "\n".join(lines), columns


def get_sample_data(conn, table_name, limit=10):
    """获取样例数据（返回列名列表和行列表）"""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(f"SELECT * FROM {table_name} ORDER BY created_at DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description] if rows else []
    cur.close()
    return colnames, rows


def escape_sql_value(val):
    """将 Python 值转为 SQL 字面量"""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, datetime):
        return f"'{val.isoformat()}'"
    # 字符串：转义单引号
    s = str(val).replace("'", "''")
    return f"'{s}'"


def main():
    parser = argparse.ArgumentParser(description="导出表结构 + 样例数据")
    parser.add_argument("--table", type=str, default="usage_records", help="要导出的表名（默认 usage_records）")
    parser.add_argument("--limit", type=int, default=10, help="导出行数（默认 10）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（默认自动生成）")
    args = parser.parse_args()

    # 输出路径
    if args.output:
        output_path = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.table}_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql")

    # 连接数据库
    print(f"🔗 连接数据库 {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}...")
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        # 1. 获取 DDL
        print(f"📋 获取表 '{args.table}' 结构...")
        ddl_result = get_table_ddl(conn, args.table)
        if ddl_result is None:
            sys.exit(1)
        ddl_sql, columns = ddl_result
        print(f"   ✓ {len(columns)} 个字段")

        # 2. 获取样例数据
        print(f"📦 获取 {args.limit} 条样例数据...")
        colnames, rows = get_sample_data(conn, args.table, args.limit)
        print(f"   ✓ 获取到 {len(rows)} 条记录")

        # 3. 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"-- ============================================\n")
            f.write(f"-- 表名: {args.table}\n")
            f.write(f"-- 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"-- 样例数据: {len(rows)} 条\n")
            f.write(f"-- ============================================\n\n")

            # 写建表语句
            f.write(f"-- [1] 表结构\n")
            f.write(f"-- --------------------------------------------\n")
            f.write(ddl_sql + "\n\n")

            # 写数据
            if rows:
                f.write(f"-- [2] 样例数据 ({len(rows)} 条)\n")
                f.write(f"-- --------------------------------------------\n")

                # 收集所有列名（使用查询返回的列名，保证顺序）
                all_cols = colnames

                f.write(f"INSERT INTO {args.table} (\n")
                f.write(f"    {', '.join(all_cols)}\n")
                f.write(f") VALUES\n")

                value_lines = []
                for row in rows:
                    vals = [escape_sql_value(row.get(c)) for c in all_cols]
                    value_lines.append(f"    ({', '.join(vals)})")

                f.write(",\n".join(value_lines) + ";\n\n")

            # 写列统计摘要
            f.write(f"-- [3] 列信息摘要\n")
            f.write(f"-- --------------------------------------------\n")
            f.write(f"-- {'列名':<30} {'类型':<25} {'可空':<8} {'默认值':<20}\n")
            f.write(f"-- {'-'*30} {'-'*25} {'-'*8} {'-'*20}\n")
            for col in columns:
                dtype = col['data_type']
                if col['character_maximum_length']:
                    dtype += f"({col['character_maximum_length']})"
                nullable = col['is_nullable']
                default = str(col['column_default'] or '')
                f.write(f"-- {col['column_name']:<30} {dtype:<25} {nullable:<8} {default:<20}\n")

        print(f"\n✅ 导出完成: {output_path}")
        file_size = os.path.getsize(output_path)
        print(f"   文件大小: {file_size:,} bytes")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
