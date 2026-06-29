# PostgreSQL 向量库上下文分析工具项目文档

## 📋 项目概述

PostgreSQL 向量库上下文分析工具是一个专门用于连接和分析 PostgreSQL 向量数据库中模型调用记录的 Python 工具集。该工具能够统计分析模型调用时的上下文长度分布情况，帮助开发者和数据分析师了解模型使用模式、优化资源配置和发现潜在问题。

### 核心价值

- **数据洞察**：深入理解模型调用的上下文使用模式
- **性能优化**：识别异常长度的上下文，优化模型配置
- **资源规划**：基于实际数据分布进行容量规划
- **问题诊断**：快速发现上下文长度相关的异常情况

## 🎯 应用场景

### 1. 模型服务监控
- 监控生产环境中模型调用的上下文长度分布
- 识别超长上下文导致的性能问题
- 分析不同时间段的上下文使用趋势

### 2. 资源容量规划
- 根据历史数据规划模型服务器资源
- 评估不同上下文长度区间的请求占比
- 预测资源需求增长趋势

### 3. 数据质量分析
- 检测异常的上下文输入
- 识别可能的数据污染或攻击
- 验证数据清洗效果

### 4. 成本优化
- 分析 token 使用分布
- 识别可以优化的长上下文调用
- 评估不同定价策略的影响

## 🏗️ 项目架构

### 目录结构

```
Resource_Planning_Tool/
├── scripts/                                    # 分析工具脚本目录
│   ├── analyze_pg_vector_context.py           # 主分析脚本（核心）
│   ├── quick_analyze.sh                       # 快速启动脚本
│   ├── requirements_pg_analysis.txt           # Python 依赖列表
│   └── README_PG_ANALYSIS.md                  # 使用说明文档
├── outputs/                                    # 分析结果输出目录
│   └── pg_analysis_YYYYMMDD_HHMMSS/          # 时间戳命名的分析结果
│       ├── results.csv                        # 详细数据导出
│       ├── distribution.png                   # 可视化图表
│       └── statistics.json                    # 统计数据
└── docs/                                      # 项目文档
    └── PG_VECTOR_ANALYSIS_PROJECT.md          # 本文档
```

### 核心组件

#### 1. 主分析脚本 (`analyze_pg_vector_context.py`)

**功能模块：**

- **数据库连接模块**
  - 支持 PostgreSQL 标准连接
  - 兼容 pgvector 扩展
  - 连接池管理和超时控制

- **数据查询模块**
  - 表结构探索
  - 字段类型识别
  - 高效批量查询

- **统计分析模块**
  - 字符长度统计（精确）
  - 词数统计（基于空格分隔）
  - 分位数计算（最小、最大、平均、中位数）
  - 分布区间统计

- **数据导出模块**
  - CSV 格式导出（支持大数据集）
  - JSON 格式导出（结构化统计）
  - 可视化图表生成（PNG）

- **用户交互模块**
  - 命令行参数解析
  - 进度显示和状态反馈
  - 错误处理和友好提示

#### 2. 快速启动脚本 (`quick_analyze.sh`)

**特性：**

- 环境变量配置管理
- 依赖自动检测和安装提示
- 输出目录自动创建（时间戳命名）
- 彩色输出和友好交互
- 错误处理和参数验证

## 🔧 技术栈

### 核心依赖

| 组件 | 版本要求 | 用途 | 类型 |
|------|----------|------|------|
| Python | >= 3.8 | 运行环境 | 必需 |
| psycopg2-binary | >= 2.9.0 | PostgreSQL 驱动 | 必需 |
| pandas | >= 2.0.0 | 数据处理和导出 | 可选 |
| matplotlib | >= 3.7.0 | 图表生成 | 可选 |
| numpy | >= 1.24.0 | 数值计算 | 可选 |

### 数据库兼容性

- ✅ PostgreSQL 12+
- ✅ PostgreSQL 13+
- ✅ PostgreSQL 14+
- ✅ PostgreSQL 15+
- ✅ PostgreSQL 16+
- ✅ 完全兼容 pgvector 扩展

### 操作系统支持

- ✅ Linux (Ubuntu 20.04+, CentOS 7+, RHEL 8+)
- ✅ macOS (10.15+)
- ✅ Windows 10/11 (通过 WSL 或原生)
- ✅ Docker 容器环境

## 📦 安装部署

### 前置要求

1. **Python 环境**
   ```bash
   python3 --version  # 确认 Python >= 3.8
   ```

2. **网络连接**
   - 能够访问目标 PostgreSQL 服务器
   - 端口 5432 可达（或自定义端口）

3. **数据库权限**
   - SELECT 权限（查询表数据）
   - 能够查询 information_schema（获取表结构）

### 安装步骤

#### 方式 1: 最小化安装（仅核心功能）

```bash
# 安装核心依赖
pip install psycopg2-binary

# 测试连接
python scripts/analyze_pg_vector_context.py --help
```

#### 方式 2: 完整安装（推荐）

```bash
# 安装所有依赖
pip install -r scripts/requirements_pg_analysis.txt

# 或者逐个安装
pip install psycopg2-binary pandas matplotlib numpy

# 验证安装
python -c "import psycopg2, pandas, matplotlib; print('所有依赖安装成功')"
```

#### 方式 3: 虚拟环境安装（推荐生产环境）

```bash
# 创建虚拟环境
python3 -m venv venv_pg_analysis

# 激活虚拟环境
source venv_pg_analysis/bin/activate  # Linux/macOS
# 或
venv_pg_analysis\Scripts\activate     # Windows

# 安装依赖
pip install -r scripts/requirements_pg_analysis.txt

# 使用完毕后退出
deactivate
```

### Docker 部署（可选）

如果需要在容器中运行：

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY scripts/requirements_pg_analysis.txt .
RUN pip install --no-cache-dir -r requirements_pg_analysis.txt

# 复制脚本
COPY scripts/ ./scripts/

# 设置入口点
ENTRYPOINT ["python", "scripts/analyze_pg_vector_context.py"]
```

构建和运行：

```bash
# 构建镜像
docker build -t pg-vector-analyzer .

# 运行分析
docker run --rm pg-vector-analyzer \
    --host 10.88.202.214 \
    --port 5432 \
    --database mydb \
    --user postgres \
    --password mypassword \
    --list-tables
```

## 🚀 使用指南

### 快速开始

#### 1. 配置数据库连接

```bash
# 方法 A: 使用环境变量（推荐，更安全）
export PG_HOST="10.88.202.214"
export PG_PORT="5432"
export PG_DATABASE="your_database"
export PG_USER="your_username"
export PG_PASSWORD="your_password"

# 方法 B: 直接在命令行传参（适合测试）
# 见下面的命令示例
```

#### 2. 探索数据库

```bash
# 使用快捷脚本
./scripts/quick_analyze.sh list

# 或使用完整命令
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database \
    --user your_username \
    --password your_password \
    --list-tables
```

**输出示例：**
```
✓ 成功连接到数据库 your_database@10.88.202.214:5432

数据库 'your_database' 中的表:
============================================================
  1. model_inference_logs          (1,234,567 条记录)
     列: id, model_id, prompt, response, context_length, created_at
  2. embeddings                    (500,123 条记录)
     列: id, vector, text, metadata, timestamp
  3. user_queries                  (89,456 条记录)
     列: id, user_id, query_text, result, timestamp
```

#### 3. 分析上下文分布

```bash
# 基础分析
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database \
    --user your_username \
    --password your_password \
    --table model_inference_logs \
    --context-column prompt

# 完整分析（含导出）
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database \
    --user your_username \
    --password your_password \
    --table model_inference_logs \
    --context-column prompt \
    --export analysis_results.csv \
    --plot distribution.png \
    --json statistics.json
```

### 高级使用场景

#### 场景 1: 大数据集采样分析

对于超大数据集（百万级以上），建议先采样分析：

```bash
# 分析最近 10,000 条记录
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database \
    --user your_username \
    --password your_password \
    --table model_inference_logs \
    --context-column prompt \
    --limit 10000
```

#### 场景 2: 多表批量分析

创建批处理脚本：

```bash
#!/bin/bash
# batch_analyze.sh

TABLES=("table1" "table2" "table3")
COLUMNS=("context" "prompt" "input_text")

for i in "${!TABLES[@]}"; do
    echo "分析 ${TABLES[$i]} 的 ${COLUMNS[$i]} 列..."
    python scripts/analyze_pg_vector_context.py \
        --host 10.88.202.214 \
        --port 5432 \
        --database your_database \
        --user your_username \
        --password your_password \
        --table "${TABLES[$i]}" \
        --context-column "${COLUMNS[$i]}" \
        --export "results_${TABLES[$i]}.csv" \
        --json "stats_${TABLES[$i]}.json"
done
```

#### 场景 3: 定时监控

使用 cron 定期执行分析：

```bash
# 编辑 crontab
crontab -e

# 每天凌晨 2 点执行分析
0 2 * * * cd /path/to/Resource_Planning_Tool && ./scripts/quick_analyze.sh analyze model_logs context
```

#### 场景 4: 多环境配置

创建配置文件管理不同环境：

```bash
# config_prod.env
export PG_HOST="10.88.202.214"
export PG_DATABASE="production_db"
export PG_USER="prod_user"
export PG_PASSWORD="prod_password"

# config_dev.env
export PG_HOST="10.88.202.215"
export PG_DATABASE="development_db"
export PG_USER="dev_user"
export PG_PASSWORD="dev_password"

# 使用时加载对应配置
source config_prod.env
./scripts/quick_analyze.sh list
```

## 📊 输出结果说明

### 1. 控制台输出

统计摘要直接显示在终端：

```
============================================================
上下文统计分析结果
============================================================

总记录数: 150,234

字符长度统计:
  最小值: 15 字符
  最大值: 125,456 字符
  平均值: 2,345.67 字符
  中位数: 1,234 字符

词数统计:
  最小值: 3 词
  最大值: 18,567 词
  平均值: 456.78 词
  中位数: 234 词

长度分布:
      0-100:  12,345 ( 8.2%) ████
    100-500:  45,678 (30.4%) ███████████████
    500-1K:   34,567 (23.0%) ███████████
     1K-2K:   28,901 (19.2%) █████████
     2K-5K:   18,234 (12.1%) ██████
    5K-10K:    7,890 ( 5.3%) ██
   10K-50K:    2,345 ( 1.6%)
       50K+:      274 ( 0.2%)
```

### 2. CSV 文件 (`results.csv`)

包含每条记录的详细数据：

| context | context_length | word_count |
|---------|----------------|------------|
| "用户输入的完整文本..." | 1234 | 156 |
| "另一条记录..." | 567 | 89 |

**用途：**
- 进一步的数据分析
- Excel 中打开查看
- 导入其他分析工具

### 3. JSON 文件 (`statistics.json`)

结构化的统计数据：

```json
{
  "total_records": 150234,
  "char_length": {
    "min": 15,
    "max": 125456,
    "mean": 2345.67,
    "median": 1234
  },
  "word_count": {
    "min": 3,
    "max": 18567,
    "mean": 456.78,
    "median": 234
  },
  "length_distribution": {
    "0-100": 12345,
    "100-500": 45678,
    "500-1K": 34567,
    "1K-2K": 28901,
    "2K-5K": 18234,
    "5K-10K": 7890,
    "10K-50K": 2345,
    "50K+": 274
  }
}
```

**用途：**
- 程序化处理统计结果
- API 集成
- 自动化报告生成

### 4. 可视化图表 (`distribution.png`)

生成双面板图表：

- **左侧**：柱状图展示各长度区间的记录数
- **右侧**：文本摘要展示关键统计指标

**特点：**
- 高分辨率（300 DPI）
- 适合放入报告和演示文稿
- 清晰的数据标签

## 🔍 数据分析方法

### 统计指标说明

#### 1. 字符长度（Character Length）

- **定义**：文本中的字符总数（包括空格、标点）
- **计算方法**：PostgreSQL `LENGTH()` 函数
- **适用场景**：
  - Token 数量估算（通常字符数 ÷ 4 ≈ token 数）
  - 存储空间评估
  - API 限制检查

#### 2. 词数（Word Count）

- **定义**：基于空格分隔的词语数量
- **计算方法**：统计空格数量 + 1
- **局限性**：不适合中文等无空格分隔的语言
- **适用场景**：
  - 英文文本分析
  - 简单的内容长度评估

#### 3. 分布区间（Distribution Bins）

| 区间 | 说明 | 典型场景 |
|------|------|---------|
| 0-100 | 极短文本 | 简单问答、关键词查询 |
| 100-500 | 短文本 | 常规对话、简单问题 |
| 500-1K | 中等文本 | 详细问题、段落级输入 |
| 1K-2K | 较长文本 | 长对话、代码片段 |
| 2K-5K | 长文本 | 文章、长代码 |
| 5K-10K | 很长文本 | 完整文档、大型代码文件 |
| 10K-50K | 超长文本 | 书籍章节、完整项目代码 |
| 50K+ | 极长文本 | 可能存在问题的输入 |

### 数据解读建议

#### 健康的分布特征

✅ **正常分布**：大部分数据集中在中间区间（100-5K）
✅ **长尾少**：超长文本（50K+）占比 < 1%
✅ **极短少**：极短文本（0-100）占比 < 10%

#### 需要关注的异常

⚠️ **双峰分布**：可能存在两种不同的使用模式
⚠️ **极端偏斜**：大量极短或极长文本，需要检查数据质量
⚠️ **异常峰值**：某个区间占比过高（>50%），可能是数据问题

## 🛠️ 故障排查

### 常见问题

#### 1. 无法连接数据库

**错误信息：**
```
✗ 数据库连接失败: could not connect to server
```

**解决方案：**

```bash
# 检查网络连通性
ping 10.88.202.214

# 检查端口是否开放
telnet 10.88.202.214 5432
# 或
nc -zv 10.88.202.214 5432

# 检查 PostgreSQL 服务状态
sudo systemctl status postgresql  # 在数据库服务器上执行

# 检查 pg_hba.conf 配置
# 确保允许远程连接
```

#### 2. 权限不足

**错误信息：**
```
✗ 查询失败: permission denied for table xxx
```

**解决方案：**

```sql
-- 在数据库中授予权限（需要管理员执行）
GRANT SELECT ON TABLE your_table TO your_user;

-- 或授予所有表的查询权限
GRANT SELECT ON ALL TABLES IN SCHEMA public TO your_user;
```

#### 3. psycopg2 安装失败

**错误信息：**
```
Error: pg_config executable not found
```

**解决方案：**

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev libpq-dev

# CentOS/RHEL
sudo yum install python3-devel postgresql-devel

# macOS
brew install postgresql

# 然后重新安装
pip install psycopg2-binary
```

#### 4. 内存不足（大数据集）

**错误信息：**
```
MemoryError: Unable to allocate array
```

**解决方案：**

```bash
# 使用 --limit 参数限制记录数
python scripts/analyze_pg_vector_context.py \
    ... \
    --limit 100000

# 或分批处理
for i in {0..10}; do
    python scripts/analyze_pg_vector_context.py \
        ... \
        --table "SELECT * FROM your_table LIMIT 100000 OFFSET $((i*100000))"
done
```

### 调试模式

启用详细日志：

```python
# 在脚本中添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 性能优化

### 查询优化

#### 1. 创建索引

在上下文字段上创建索引可以加快分析速度：

```sql
-- 创建长度索引
CREATE INDEX idx_context_length ON model_logs(LENGTH(context));

-- 创建时间范围索引（用于时间序列分析）
CREATE INDEX idx_created_at ON model_logs(created_at);
```

#### 2. 使用物化视图

对于频繁分析的数据，可以创建物化视图：

```sql
CREATE MATERIALIZED VIEW mv_context_stats AS
SELECT
    LENGTH(context) as context_length,
    LENGTH(context) - LENGTH(REPLACE(context, ' ', '')) + 1 as word_count,
    created_at
FROM model_logs;

-- 创建索引
CREATE INDEX idx_mv_context_length ON mv_context_stats(context_length);

-- 定期刷新
REFRESH MATERIALIZED VIEW mv_context_stats;
```

#### 3. 分区表

对于超大表，使用分区可以提高查询效率：

```sql
-- 按时间分区
CREATE TABLE model_logs_2024_01 PARTITION OF model_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 脚本优化

#### 1. 批量处理

```python
# 修改脚本，使用游标批量获取
cursor = self.conn.cursor(name='server_side_cursor')
cursor.itersize = 10000  # 每次获取 10000 条
```

#### 2. 并行分析

```bash
# 使用 GNU Parallel 并行处理多个表
parallel -j 4 python scripts/analyze_pg_vector_context.py \
    --table {} \
    ::: table1 table2 table3 table4
```

## 🔐 安全最佳实践

### 1. 密码管理

❌ **不要**：在命令行直接输入密码
```bash
# 不推荐 - 密码会留在 shell 历史中
python script.py --password mypassword
```

✅ **推荐**：使用环境变量或密码文件
```bash
# 方法 1: 环境变量
export PG_PASSWORD="mypassword"
python script.py --password "$PG_PASSWORD"

# 方法 2: .pgpass 文件
echo "10.88.202.214:5432:mydb:myuser:mypassword" >> ~/.pgpass
chmod 600 ~/.pgpass

# 方法 3: 交互式输入
read -s PG_PASSWORD
export PG_PASSWORD
```

### 2. 网络安全

```bash
# 使用 SSL 连接（修改脚本支持）
--sslmode=require

# 使用 SSH 隧道
ssh -L 5432:10.88.202.214:5432 user@jumphost
# 然后连接到 localhost:5432
```

### 3. 权限最小化

```sql
-- 创建只读用户
CREATE USER analyzer_readonly WITH PASSWORD 'secure_password';

-- 只授予必要的权限
GRANT CONNECT ON DATABASE your_database TO analyzer_readonly;
GRANT USAGE ON SCHEMA public TO analyzer_readonly;
GRANT SELECT ON specific_table TO analyzer_readonly;
```

### 4. 审计日志

```bash
# 记录所有分析操作
python scripts/analyze_pg_vector_context.py ... 2>&1 | \
    tee -a /var/log/pg_analysis_$(date +%Y%m%d).log
```

## 📚 扩展开发

### 自定义分析指标

你可以修改脚本添加自定义统计：

```python
# 在 analyze_context_distribution 方法中添加

# 示例：统计特定关键词出现频率
keyword_counts = {}
for row in results:
    context = row['context'].lower()
    if 'error' in context:
        keyword_counts['error'] = keyword_counts.get('error', 0) + 1
    if 'success' in context:
        keyword_counts['success'] = keyword_counts.get('success', 0) + 1

stats['keyword_counts'] = keyword_counts
```

### 集成到其他系统

#### 作为 Python 模块使用

```python
from scripts.analyze_pg_vector_context import PGVectorAnalyzer

# 创建分析器实例
analyzer = PGVectorAnalyzer(
    host='10.88.202.214',
    port=5432,
    database='mydb',
    user='myuser',
    password='mypass'
)

# 连接并分析
analyzer.connect()
stats, results = analyzer.analyze_context_distribution(
    table_name='model_logs',
    context_column='prompt'
)
analyzer.close()

# 使用统计结果
print(f"平均长度: {stats['char_length']['mean']}")
```

#### REST API 包装

```python
# api_server.py
from fastapi import FastAPI
from scripts.analyze_pg_vector_context import PGVectorAnalyzer

app = FastAPI()

@app.post("/analyze")
async def analyze(table: str, column: str = "context"):
    analyzer = PGVectorAnalyzer(...)
    analyzer.connect()
    stats, _ = analyzer.analyze_context_distribution(table, column)
    analyzer.close()
    return stats
```

## 📖 参考资料

### 相关文档

- [PostgreSQL 官方文档](https://www.postgresql.org/docs/)
- [psycopg2 文档](https://www.psycopg.org/docs/)
- [pgvector 扩展](https://github.com/pgvector/pgvector)
- [Pandas 文档](https://pandas.pydata.org/docs/)
- [Matplotlib 文档](https://matplotlib.org/stable/contents.html)

### 学习资源

- [PostgreSQL 性能优化](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Python 数据分析最佳实践](https://realpython.com/tutorials/data-science/)
- [SQL 查询优化技巧](https://use-the-index-luke.com/)

## 🤝 贡献指南

### 报告问题

如果发现 bug 或有功能建议，请创建 issue 并包含：

1. 详细的问题描述
2. 重现步骤
3. 预期行为 vs 实际行为
4. 环境信息（Python 版本、操作系统、PostgreSQL 版本）
5. 错误日志（如有）

### 提交改进

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 推送到分支
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加适当的注释和文档字符串
- 编写单元测试
- 更新相关文档

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 📞 支持与联系

- **文档**：`scripts/README_PG_ANALYSIS.md`
- **问题追踪**：项目 issue 页面
- **技术支持**：参考故障排查章节

## 🗺️ 路线图

### 已完成 ✅

- [x] 基础连接和查询功能
- [x] 统计分析和分布计算
- [x] CSV/JSON/图表导出
- [x] 命令行界面
- [x] 快捷脚本

### 计划中 🚧

- [ ] Web UI 界面
- [ ] 实时监控仪表板
- [ ] 自动告警功能
- [ ] 历史趋势分析
- [ ] 多数据库并行分析
- [ ] 机器学习异常检测
- [ ] 交互式可视化报告

### 未来展望 💡

- [ ] 支持更多数据库（MySQL、MongoDB）
- [ ] 云原生部署（Kubernetes）
- [ ] SaaS 版本
- [ ] 集成到 CI/CD 流程

---

**版本**：1.0.0
**最后更新**：2026-06-08
**维护者**：Resource Planning Tool Team
