#!/bin/bash
# PostgreSQL 向量库快速分析脚本
#
# 使用方法:
#   ./quick_analyze.sh list                     # 列出所有表
#   ./quick_analyze.sh analyze table_name       # 分析指定表
#   ./quick_analyze.sh analyze table_name column_name  # 分析指定表的指定列

set -e

# 默认连接参数（可以通过环境变量覆盖）
PG_HOST="${PG_HOST:-10.88.202.214}"
PG_PORT="${PG_PORT:-5432}"
PG_DATABASE="${PG_DATABASE:-appdb}"
PG_USER="${PG_USER:-appuser}"
PG_PASSWORD="${PG_PASSWORD:-op3mF+-LKez3AN-mQP9D}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查必需参数
if [ -z "$PG_DATABASE" ] || [ -z "$PG_USER" ] || [ -z "$PG_PASSWORD" ]; then
    echo -e "${RED}错误: 请设置数据库连接信息${NC}"
    echo ""
    echo "使用环境变量:"
    echo "  export PG_DATABASE=your_database"
    echo "  export PG_USER=your_username"
    echo "  export PG_PASSWORD=your_password"
    echo ""
    echo "可选环境变量:"
    echo "  export PG_HOST=10.88.202.214  # 默认值"
    echo "  export PG_PORT=5432           # 默认值"
    echo ""
    echo "然后运行:"
    echo "  $0 list"
    echo "  $0 analyze table_name"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/analyze_pg_vector_context.py"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 找不到 Python 脚本: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 检查依赖
if ! python3 -c "import psycopg2" 2>/dev/null; then
    echo -e "${YELLOW}警告: psycopg2 未安装${NC}"
    echo "安装命令: pip install psycopg2-binary"
    echo ""
    read -p "是否现在安装? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install psycopg2-binary
    else
        exit 1
    fi
fi

# 基础命令参数
BASE_CMD="python3 $PYTHON_SCRIPT --host $PG_HOST --port $PG_PORT --database $PG_DATABASE --user $PG_USER --password $PG_PASSWORD"

# 解析命令
COMMAND=${1:-help}

case $COMMAND in
    list)
        echo -e "${GREEN}列出数据库中的所有表...${NC}"
        $BASE_CMD --list-tables
        ;;

    analyze)
        if [ -z "$2" ]; then
            echo -e "${RED}错误: 请指定表名${NC}"
            echo "用法: $0 analyze table_name [context_column]"
            exit 1
        fi

        TABLE_NAME=$2
        CONTEXT_COLUMN=${3:-context}

        echo -e "${GREEN}分析表 '$TABLE_NAME' 的 '$CONTEXT_COLUMN' 列...${NC}"

        # 创建输出目录
        OUTPUT_DIR="$SCRIPT_DIR/../outputs/pg_analysis_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$OUTPUT_DIR"

        # 运行分析
        $BASE_CMD \
            --table "$TABLE_NAME" \
            --context-column "$CONTEXT_COLUMN" \
            --export "$OUTPUT_DIR/results.csv" \
            --plot "$OUTPUT_DIR/distribution.png" \
            --json "$OUTPUT_DIR/statistics.json"

        echo ""
        echo -e "${GREEN}分析完成！${NC}"
        echo -e "输出目录: ${YELLOW}$OUTPUT_DIR${NC}"
        ls -lh "$OUTPUT_DIR"
        ;;

    help|*)
        echo "PostgreSQL 向量库快速分析工具"
        echo ""
        echo "用法:"
        echo "  $0 list                              列出所有表"
        echo "  $0 analyze table_name [column_name]  分析指定表"
        echo ""
        echo "环境变量:"
        echo "  PG_HOST      - 数据库主机 (默认: 10.88.202.214)"
        echo "  PG_PORT      - 数据库端口 (默认: 5432)"
        echo "  PG_DATABASE  - 数据库名称 (必需)"
        echo "  PG_USER      - 用户名 (必需)"
        echo "  PG_PASSWORD  - 密码 (必需)"
        echo ""
        echo "示例:"
        echo "  export PG_DATABASE=vector_db"
        echo "  export PG_USER=postgres"
        echo "  export PG_PASSWORD=mypassword"
        echo "  $0 list"
        echo "  $0 analyze model_calls context"
        ;;
esac
