#!/usr/bin/env python3
"""
PostgreSQL 向量库上下文分布统计脚本

用于连接到PostgreSQL向量库，统计模型调用记录中上下文长度的分布情况。

使用方法：
    python analyze_pg_vector_context.py --help
    python analyze_pg_vector_context.py --host 10.88.202.214 --port 5432 --database your_db --user your_user --password your_password
"""

import argparse
import sys
from typing import Dict, List, Optional
import json
from datetime import datetime
from collections import Counter

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("错误: 需要安装 psycopg2 库")
    print("请运行: pip install psycopg2-binary")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("警告: pandas 未安装，将使用基础统计功能")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 未安装，无法生成图表")


class PGVectorAnalyzer:
    """PostgreSQL 向量库分析器"""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None

    def connect(self):
        """连接到数据库"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print(f"✓ 成功连接到数据库 {self.database}@{self.host}:{self.port}")
            return True
        except psycopg2.Error as e:
            print(f"✗ 数据库连接失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("✓ 数据库连接已关闭")

    def list_tables(self) -> List[str]:
        """列出所有表"""
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """
        self.cursor.execute(query)
        tables = [row['table_name'] for row in self.cursor.fetchall()]
        return tables

    def get_table_columns(self, table_name: str) -> List[Dict]:
        """获取表的列信息"""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """
        self.cursor.execute(query, (table_name,))
        return self.cursor.fetchall()

    def get_table_count(self, table_name: str) -> int:
        """获取表的记录数"""
        query = f"SELECT COUNT(*) as count FROM {table_name};"
        self.cursor.execute(query)
        return self.cursor.fetchone()['count']

    def analyze_context_distribution(self, table_name: str, context_column: str = 'context',
                                    limit: Optional[int] = None) -> Dict:
        """
        分析上下文长度分布

        Args:
            table_name: 表名
            context_column: 上下文列名（可能是 'context', 'prompt', 'input_text' 等）
            limit: 限制分析的记录数，None表示全部

        Returns:
            包含统计信息的字典
        """
        print(f"\n分析表 '{table_name}' 中的 '{context_column}' 列...")

        # 构建查询
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
            SELECT
                {context_column},
                LENGTH({context_column}) as context_length,
                LENGTH({context_column}) - LENGTH(REPLACE({context_column}, ' ', '')) + 1 as word_count
            FROM {table_name}
            WHERE {context_column} IS NOT NULL
            {limit_clause};
        """

        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            if not results:
                print(f"✗ 表 '{table_name}' 中没有找到数据")
                return {}

            # 提取长度数据
            lengths = [row['context_length'] for row in results]
            word_counts = [row['word_count'] for row in results]

            # 基础统计
            stats = {
                'total_records': len(results),
                'char_length': {
                    'min': min(lengths),
                    'max': max(lengths),
                    'mean': sum(lengths) / len(lengths),
                    'median': sorted(lengths)[len(lengths) // 2],
                },
                'word_count': {
                    'min': min(word_counts),
                    'max': max(word_counts),
                    'mean': sum(word_counts) / len(word_counts),
                    'median': sorted(word_counts)[len(word_counts) // 2],
                }
            }

            # 长度分布区间统计
            bins = [0, 100, 500, 1000, 2000, 5000, 10000, 50000, float('inf')]
            bin_labels = ['0-100', '100-500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '10K-50K', '50K+']
            distribution = Counter()

            for length in lengths:
                for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                    if lower <= length < upper:
                        distribution[bin_labels[i]] += 1
                        break

            stats['length_distribution'] = dict(distribution)

            return stats, results

        except psycopg2.Error as e:
            print(f"✗ 查询失败: {e}")
            return {}, []

    def print_statistics(self, stats: Dict):
        """打印统计信息"""
        if not stats:
            return

        print("\n" + "="*60)
        print("上下文统计分析结果")
        print("="*60)

        print(f"\n总记录数: {stats['total_records']:,}")

        print("\n字符长度统计:")
        print(f"  最小值: {stats['char_length']['min']:,} 字符")
        print(f"  最大值: {stats['char_length']['max']:,} 字符")
        print(f"  平均值: {stats['char_length']['mean']:.2f} 字符")
        print(f"  中位数: {stats['char_length']['median']:,} 字符")

        print("\n词数统计:")
        print(f"  最小值: {stats['word_count']['min']:,} 词")
        print(f"  最大值: {stats['word_count']['max']:,} 词")
        print(f"  平均值: {stats['word_count']['mean']:.2f} 词")
        print(f"  中位数: {stats['word_count']['median']:,} 词")

        print("\n长度分布:")
        total = stats['total_records']
        for range_label, count in sorted(stats['length_distribution'].items(),
                                         key=lambda x: ['0-100', '100-500', '500-1K', '1K-2K',
                                                       '2K-5K', '5K-10K', '10K-50K', '50K+'].index(x[0])):
            percentage = (count / total) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {range_label:>10}: {count:>6} ({percentage:>5.1f}%) {bar}")

    def export_to_csv(self, results: List[Dict], output_file: str):
        """导出结果到CSV"""
        if not PANDAS_AVAILABLE:
            print("✗ 需要安装 pandas 才能导出 CSV")
            return

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ 结果已导出到: {output_file}")

    def plot_distribution(self, stats: Dict, output_file: str = 'context_distribution.png'):
        """绘制分布图"""
        if not MATPLOTLIB_AVAILABLE:
            print("✗ 需要安装 matplotlib 才能生成图表")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 长度分布柱状图
        ranges = ['0-100', '100-500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '10K-50K', '50K+']
        counts = [stats['length_distribution'].get(r, 0) for r in ranges]

        ax1.bar(ranges, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('上下文长度范围 (字符)', fontsize=12)
        ax1.set_ylabel('记录数', fontsize=12)
        ax1.set_title('上下文长度分布', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, count in enumerate(counts):
            if count > 0:
                ax1.text(i, count, str(count), ha='center', va='bottom')

        # 统计概览
        ax2.axis('off')
        summary_text = f"""
        统计概览
        {'='*30}

        总记录数: {stats['total_records']:,}

        字符长度:
          平均: {stats['char_length']['mean']:.0f}
          中位数: {stats['char_length']['median']:,}
          范围: {stats['char_length']['min']:,} - {stats['char_length']['max']:,}

        词数:
          平均: {stats['word_count']['mean']:.0f}
          中位数: {stats['word_count']['median']:,}
          范围: {stats['word_count']['min']:,} - {stats['word_count']['max']:,}
        """

        ax2.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='PostgreSQL 向量库上下文分布统计工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python analyze_pg_vector_context.py --host 10.88.202.214 --port 5432 \\
      --database mydb --user postgres --password mypass \\
      --table model_calls --context-column context

  # 列出所有表
  python analyze_pg_vector_context.py --host 10.88.202.214 --port 5432 \\
      --database mydb --user postgres --password mypass --list-tables

  # 导出结果
  python analyze_pg_vector_context.py --host 10.88.202.214 --port 5432 \\
      --database mydb --user postgres --password mypass \\
      --table model_calls --export results.csv --plot distribution.png
        """
    )

    # 数据库连接参数
    parser.add_argument('--host', default='10.88.202.214', help='数据库主机 (默认: 10.88.202.214)')
    parser.add_argument('--port', type=int, default=5432, help='数据库端口 (默认: 5432)')
    parser.add_argument('--database', required=True, help='数据库名称')
    parser.add_argument('--user', required=True, help='数据库用户名')
    parser.add_argument('--password', required=True, help='数据库密码')

    # 操作参数
    parser.add_argument('--list-tables', action='store_true', help='列出所有表')
    parser.add_argument('--table', help='要分析的表名')
    parser.add_argument('--context-column', default='context', help='上下文列名 (默认: context)')
    parser.add_argument('--limit', type=int, help='限制分析的记录数')

    # 输出参数
    parser.add_argument('--export', help='导出结果到 CSV 文件')
    parser.add_argument('--plot', help='生成分布图并保存')
    parser.add_argument('--json', help='导出统计结果为 JSON 文件')

    args = parser.parse_args()

    # 创建分析器
    analyzer = PGVectorAnalyzer(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )

    # 连接数据库
    if not analyzer.connect():
        sys.exit(1)

    try:
        # 列出所有表
        if args.list_tables:
            tables = analyzer.list_tables()
            print(f"\n数据库 '{args.database}' 中的表:")
            print("="*60)
            for i, table in enumerate(tables, 1):
                count = analyzer.get_table_count(table)
                print(f"{i:3}. {table:30} ({count:,} 条记录)")

                # 显示列信息
                columns = analyzer.get_table_columns(table)
                print(f"     列: {', '.join([col['column_name'] for col in columns[:5]])}", end='')
                if len(columns) > 5:
                    print(f" ... (+{len(columns)-5} 列)")
                else:
                    print()
            return

        # 分析表
        if args.table:
            stats, results = analyzer.analyze_context_distribution(
                table_name=args.table,
                context_column=args.context_column,
                limit=args.limit
            )

            if stats:
                analyzer.print_statistics(stats)

                # 导出 CSV
                if args.export and results:
                    analyzer.export_to_csv(results, args.export)

                # 生成图表
                if args.plot:
                    analyzer.plot_distribution(stats, args.plot)

                # 导出 JSON
                if args.json:
                    with open(args.json, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2, ensure_ascii=False)
                    print(f"✓ 统计结果已导出到: {args.json}")
        else:
            print("错误: 请使用 --list-tables 列出表，或使用 --table 指定要分析的表")
            parser.print_help()

    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
