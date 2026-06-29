# PostgreSQL 向量库上下文分析工具

这个工具用于连接 PostgreSQL 向量库并统计分析模型调用记录中上下文长度的分布情况。

## 功能特性

- ✅ 连接到 PostgreSQL 数据库（包括向量库）
- ✅ 列出数据库中的所有表及其结构
- ✅ 分析上下文字段的长度分布
- ✅ 提供详细的统计信息（最小值、最大值、平均值、中位数）
- ✅ 支持按区间统计分布（0-100, 100-500, 500-1K, 1K-2K, 2K-5K, 5K-10K, 10K-50K, 50K+）
- ✅ 导出结果为 CSV 文件
- ✅ 生成可视化分布图表
- ✅ 导出统计数据为 JSON 格式

## 安装依赖

### 方式 1: 仅安装必需依赖（基础功能）

```bash
pip install psycopg2-binary
```

### 方式 2: 安装全部依赖（完整功能）

```bash
pip install -r scripts/requirements_pg_analysis.txt
```

或者单独安装：

```bash
pip install psycopg2-binary pandas matplotlib numpy
```

## 使用方法

### 1. 列出数据库中的所有表

首先，你需要知道数据库中有哪些表：

```bash
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database_name \
    --user your_username \
    --password your_password \
    --list-tables
```

示例输出：
```
✓ 成功连接到数据库 your_database_name@10.88.202.214:5432

数据库 'your_database_name' 中的表:
============================================================
  1. model_calls                   (150,234 条记录)
     列: id, model_name, context, created_at, user_id
  2. embeddings                    (50,123 条记录)
     列: id, vector, metadata, text, timestamp
```

### 2. 分析特定表的上下文分布

假设你要分析的表名是 `model_calls`，上下文字段名是 `context`：

```bash
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database_name \
    --user your_username \
    --password your_password \
    --table model_calls \
    --context-column context
```

### 3. 限制分析的记录数（大数据集时）

如果数据量很大，可以先分析一部分数据：

```bash
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database_name \
    --user your_username \
    --password your_password \
    --table model_calls \
    --context-column context \
    --limit 10000
```

### 4. 导出分析结果

#### 导出为 CSV

```bash
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database_name \
    --user your_username \
    --password your_password \
    --table model_calls \
    --context-column context \
    --export analysis_results.csv
```

#### 生成可视化图表

```bash
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database_name \
    --user your_username \
    --password your_password \
    --table model_calls \
    --context-column context \
    --plot context_distribution.png
```

#### 导出统计数据为 JSON

```bash
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database your_database_name \
    --user your_username \
    --password your_password \
    --table model_calls \
    --context-column context \
    --json statistics.json
```

### 5. 完整示例（所有功能）

```bash
python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database vector_db \
    --user postgres \
    --password mypassword \
    --table model_calls \
    --context-column prompt \
    --export results.csv \
    --plot distribution.png \
    --json stats.json
```

## 输出示例

### 控制台输出

```
✓ 成功连接到数据库 vector_db@10.88.202.214:5432

分析表 'model_calls' 中的 'context' 列...

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

✓ 结果已导出到: results.csv
✓ 图表已保存到: distribution.png
✓ 统计结果已导出到: stats.json
✓ 数据库连接已关闭
```

## 常见问题

### Q1: 如何找到上下文字段的列名？

使用 `--list-tables` 查看所有表的列信息。常见的列名包括：
- `context`
- `prompt`
- `input_text`
- `user_input`
- `message`
- `content`

### Q2: 连接超时怎么办？

检查以下几点：
1. 确认网络能连接到 10.88.202.214:5432
2. 确认防火墙允许 PostgreSQL 端口（5432）
3. 确认数据库允许远程连接
4. 尝试使用 `telnet 10.88.202.214 5432` 测试连接

### Q3: 密码包含特殊字符怎么办？

将密码放在引号中：
```bash
--password 'my@pass#word!'
```

### Q4: 如何分析中文文本？

工具自动支持 UTF-8 编码，可以正确处理中文文本。词数统计基于空格分隔，中文文本的词数统计可能不准确，但字符长度统计是准确的。

## 高级用法

### 使用环境变量存储密码

为了安全，可以使用环境变量：

```bash
export PG_PASSWORD="your_password"

python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database vector_db \
    --user postgres \
    --password "$PG_PASSWORD" \
    --table model_calls
```

### 创建快捷脚本

创建一个 shell 脚本 `analyze.sh`：

```bash
#!/bin/bash

python scripts/analyze_pg_vector_context.py \
    --host 10.88.202.214 \
    --port 5432 \
    --database vector_db \
    --user postgres \
    --password "$PG_PASSWORD" \
    "$@"
```

然后使用：
```bash
chmod +x analyze.sh
./analyze.sh --list-tables
./analyze.sh --table model_calls --context-column context
```

## 技术说明

- **数据库驱动**: psycopg2（Python 最常用的 PostgreSQL 驱动）
- **向量库支持**: 完全兼容 pgvector 扩展
- **字符长度**: 使用 PostgreSQL 的 `LENGTH()` 函数，准确统计字符数
- **词数统计**: 基于空格分隔，适用于英文文本
- **性能**: 对于大数据集建议使用 `--limit` 参数先采样分析

## 许可证

MIT License
