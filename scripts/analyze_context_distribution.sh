#!/bin/bash
# 上下文分布全面分析工具
# 基于 Python + psycopg2，不需要 psql 客户端
#
# 使用方法:
#   ./analyze_context_distribution.sh                      # 默认全部分析 (30天)
#   ./analyze_context_distribution.sh --days 7             # 最近 7 天
#   ./analyze_context_distribution.sh --module cache       # 只分析缓存
#   ./analyze_context_distribution.sh --no-plot            # 不生成图表
#
# 模块列表:
#   overview     - 总体概览
#   distribution - 上下文长度分布
#   model        - 按模型分析
#   cache        - 缓存命中率分析
#   trend        - 每日趋势
#   users        - 用户维度分析
#   ratio        - 上下文窗口使用率

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/analyze_context.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 找不到 Python 脚本: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 检查 Python 依赖
echo -e "${GREEN}检查 Python 依赖...${NC}"
for pkg in psycopg2 matplotlib; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${YELLOW}安装 $pkg ...${NC}"
        pip install ${pkg/psycopg2/psycopg2-binary} -q
    fi
done

# 运行分析
echo -e "${GREEN}开始分析...${NC}"
python3 "$PYTHON_SCRIPT" "$@"
