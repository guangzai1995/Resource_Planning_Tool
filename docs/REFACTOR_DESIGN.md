# Resource Planning Tool — 重构设计方案

> 版本：v2.1 | 日期：2026-04-29 | 状态：设计稿
> 
> **定位**：内部工具，面向团队使用，**单容器部署**，不引入外部服务依赖（无 PostgreSQL/Redis/Celery）。

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [现状分析与痛点](#2-现状分析与痛点)
3. [整体架构设计](#3-整体架构设计)
4. [技术选型](#4-技术选型)
5. [数据库设计](#5-数据库设计)
6. [后端服务设计](#6-后端服务设计)
7. [性能预测引擎设计（无数据推测）](#7-性能预测引擎设计无数据推测)
8. [前端应用设计](#8-前端应用设计)
9. [基准测试集成](#9-基准测试集成)
10. [部署方案](#10-部署方案)
11. [迁移计划](#11-迁移计划)
12. [附录：数据字典](#附录数据字典)

---

## 1. 背景与目标

### 1.1 项目背景

本工具服务于 LLM 推理资源规划场景：用户基于业务并发量、延迟要求和成本预算，选择最合适的 GPU 型号和卡数方案。  
数据来源于 vLLM 部署的实测基准（`benchmark_tools/`），覆盖 4090/A40/A100/A800/H20/H200/910B 等 GPU 型号，支持 7B/32B/72B/235B-MOE 等主流模型。

### 1.2 重构目标

| 维度 | 现状 | 目标 |
|------|------|------|
| UI 框架 | Gradio 5.x（Demo级）| React + Ant Design + ECharts |
| 数据存储 | 本地 CSV（旧，无 prefix cache）| SQLite（内嵌，单文件，零依赖）|
| 数据来源 | data/ CSV 文件 | **资源规划工具.xlsx**（Excel Sheet 导入）|
| 后端服务 | 单文件 FastAPI | 分层 FastAPI（服务/路由解耦）|
| 性能预测 | 仅插值已有数据 | 插值 + vLLM 参数建模双引擎 |
| 基准测试 | Gradio 界面触发 | asyncio 后台任务 + WebSocket 推流 |
| 认证 | 无 | 无（内部工具，局域网部署）|
| 可观测性 | 无 | structlog 结构化日志 |
| 部署 | 手动运行 | **单 Docker 容器**（`docker run` 一行启动）|

---

## 2. 现状分析与痛点

### 2.1 Gradio 的局限性

```
现有架构（单进程）：
resourse_plan_tool.py
├── gr.Blocks()             ← 事件循环绑定耦合严重
├── src/data_loader.py      ← 启动时全量加载 CSV 到内存
├── src/performance.py      ← 无缓存，每次点击重算
├── src/visualization.py    ← 返回 matplotlib Figure，无法复用
└── src/run_benchmark.py    ← subprocess 阻塞 Gradio 主线程
```

**核心痛点：**
- Gradio 的事件回调无法支持多用户并发（全局状态污染）
- 测试数据全量常驻内存，数据增长后启动时间线性增加
- 图表是静态 PNG，用户无法交互（缩放/筛选/下载）
- 基准测试通过 `subprocess` 阻塞 UI，无法多任务并行

### 2.2 本地 CSV 的局限性

- 旧 data/ 目录下的测试数据**均未开启 prefix caching**，数据质量存疑，直接废弃
- 新数据来源统一为 `资源规划工具.xlsx`，按 Sheet 页组织（覆盖 P800/H200/H20 × 4B~671B）
- 无索引，跨型号查询需要全量遍历

### 2.3 预测能力的局限性

当前仅对已有数据做插值（线性/最近邻），对于**没有实测数据的 GPU 型号 + 模型组合**，系统无法给出任何估算，是核心能力缺口。

---

## 3. 整体架构设计

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        浏览器                                │
│   React SPA (Ant Design + ECharts)                          │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│   │ 资源规划  │ │ 成本优化  │ │ 基准测试  │ │ 数据管理     │  │
│   └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP / WebSocket
┌───────────────────────────▼──────────── 单个 Docker 容器 ───┐
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              FastAPI + Uvicorn                       │  │
│  │   /api/v1/*   REST 接口                              │  │
│  │   /api/v1/benchmark/*/stream   WebSocket             │  │
│  │   /           前端静态文件（Vite build）              │  │
│  │                                                      │  │
│  │   ┌─────────────────┐   ┌─────────────────────────┐  │  │
│  │   │  预测引擎        │   │  后台任务管理器           │  │  │
│  │   │  ├ 插值预测      │   │  asyncio.TaskGroup       │  │  │
│  │   │  └ vLLM 建模     │   │  ├ benchmark subprocess  │  │  │
│  │   │  TTLCache 内存   │   │  └ asyncio.Queue → WS    │  │  │
│  │   └─────────────────┘   └─────────────────────────┘  │  │
│  └────────────────────────────────┬─────────────────────┘  │
│                                   │                        │
│  ┌────────────────────────────────▼─────────────────────┐  │
│  │              SQLite（/data/rpt.db）                   │  │
│  │  ├ benchmark_data   ├ gpu_specs   ├ models            │  │
│  │  ├ benchmark_runs   └ task_status                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  挂载卷：/data/              （SQLite 文件持久化）            │
│           /app/benchmark_tools/  （压测脚本只读挂载）        │
│           /app/model/            （分词器只读挂载）          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流设计

```
【查询流】
用户请求 → API → 预测引擎
                    ├─ 命中 TTLCache（内存）→ 直接返回
                    ├─ 有实测数据 → SQLite 查询 → 多维插值 → 写入 TTLCache → 返回
                    └─ 无实测数据 → vLLM 参数建模推算 → 置信度标注 → 写入 TTLCache → 返回

【基准测试流】
UI 提交任务 → POST /api/v1/benchmark/run
           → asyncio.create_task(run_benchmark)
               ├─ subprocess(benchmark_parallel.py) 逐行读取 stdout
               ├─ asyncio.Queue.put(line) → WebSocket 消费端实时推送
               └─ 完成 → 解析 CSV → INSERT INTO SQLite benchmark_data

【数据初始化流】
首次启动 → 检测 SQLite 是否存在
         → 不存在 → 解析 资源规划工具.xlsx 所有 Sheet → 批量导入 SQLite
         → 已存在 → 跳过（直接启动）

【追加导入流】
Excel/CSV 上传 → /api/v1/data/import → 解析 → UPSERT SQLite → 清空 TTLCache
```

---

## 4. 技术选型

### 4.1 后端

| 组件 | 选型 | 版本 | 理由 |
|------|------|------|------|
| Web 框架 | FastAPI | ≥0.115 | 已使用，async 原生，OpenAPI 自动生成 |
| ASGI 服务器 | Uvicorn | — | 单进程，内置异步 I/O |
| ORM | SQLAlchemy 2.x | ≥2.0 | 支持 SQLite，同步即可（数据量小）|
| 数据库 | **SQLite** | 内置 | 零依赖，单文件持久化，完全满足本工具需求 |
| 迁移工具 | Alembic | — | 支持 SQLite，schema 版本化 |
| 内存缓存 | **cachetools.TTLCache** | — | 替代 Redis，进程内 TTL 缓存，轻量 |
| 后台任务 | **asyncio + ThreadPoolExecutor** | 内置 | 替代 Celery，运行 subprocess 压测任务 |
| WS 日志流 | **asyncio.Queue** | 内置 | 替代 Redis Pub/Sub，每任务一个 Queue |
| 数据读取 | openpyxl + pandas | — | 解析 Excel(.xlsx) + CSV 导入 |
| 数据校验 | Pydantic v2 | — | FastAPI 原生集成 |
| 科学计算 | scipy + numpy | — | 三维插值 (RegularGridInterpolator) |
| 日志 | structlog | — | 结构化日志，输出到控制台 |

### 4.2 前端

| 组件 | 选型 | 理由 |
|------|------|------|
| 框架 | React 18 + TypeScript | 生态成熟，类型安全 |
| UI 组件库 | Ant Design 5.x | 企业级，表格/表单/布局完善 |
| 图表 | Apache ECharts 5.x | 交互式图表，支持缩放/下载，比 matplotlib 强 |
| 状态管理 | Zustand | 轻量，比 Redux 简单 |
| 请求库 | TanStack Query (React Query) v5 | 自动缓存/重试/加载状态 |
| 表单 | React Hook Form + Zod | 高性能表单 + schema 校验 |
| 构建工具 | Vite 5.x | 快速 HMR |
| WebSocket | 原生 API + reconnect 库 | 基准测试实时日志 |

### 4.3 基础设施

| 组件 | 选型 |
|------|------|
| 容器化 | **单个 Docker 容器**，一行命令启动 |
| 静态资源 | Vite build → 嵌入镜像 → FastAPI StaticFiles 挂载 |
| 持久化 | Docker volume 挂载 `/data/`（存放 SQLite 文件）|
| CI/CD | GitHub Actions（可选）|

---

## 5. 数据库设计

> 数据库使用 **SQLite**，文件存储在挂载卷 `/data/rpt.db`，容器重建后数据不丢失。  
> SQLAlchemy 2.x 同步模式即可（数据量小，无高并发写入场景）。

### 5.1 核心表结构（SQLite 兼容 DDL）

#### 5.1.1 GPU 规格表 `gpu_specs`

```sql
CREATE TABLE IF NOT EXISTS gpu_specs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,   -- "H200", "H20", "P800"
    vendor          TEXT,                   -- "NVIDIA", "Huawei"
    memory_gb       REAL NOT NULL,          -- 显存 GB
    memory_bw_gbps  REAL,                   -- 显存带宽 GB/s
    bf16_tflops     REAL,                   -- BF16 算力 TFlops
    price_per_hour  REAL NOT NULL DEFAULT 0, -- 租赁单价 ¥/卡·小时
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
```

#### 5.1.2 模型表 `models`

```sql
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,   -- "7B", "72B", "671B"
    size_b          REAL NOT NULL,          -- 参数量（十亿）
    is_moe          INTEGER DEFAULT 0,      -- 0/1
    num_layers      INTEGER,
    hidden_size     INTEGER,
    num_kv_heads    INTEGER,
    head_size       INTEGER,
    quantization    TEXT,                   -- NULL / "AWQ" / "FP8"
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
```

#### 5.1.3 基准测试批次表 `benchmark_runs`

```sql
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uuid        TEXT NOT NULL UNIQUE,   -- Python uuid4() 生成
    gpu_name        TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    gpu_count       INTEGER NOT NULL,
    vllm_args       TEXT,                   -- JSON 字符串
    benchmark_args  TEXT,                   -- JSON 字符串
    status          TEXT DEFAULT 'pending', -- pending/running/done/failed
    source          TEXT DEFAULT 'benchmark', -- 'benchmark' 或 'excel_import'
    started_at      TEXT,
    finished_at     TEXT,
    error_message   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
```

#### 5.1.4 基准测试数据点表 `benchmark_data`（核心）

```sql
CREATE TABLE IF NOT EXISTS benchmark_data (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 INTEGER REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    -- 测试维度（4 维联合唯一）
    gpu_name               TEXT NOT NULL,
    model_name             TEXT NOT NULL,
    gpu_count              INTEGER NOT NULL,
    input_tokens           INTEGER NOT NULL,
    output_tokens          INTEGER NOT NULL,
    concurrency            INTEGER NOT NULL,
    -- 性能指标
    output_throughput      REAL,   -- tokens/s（系统级）
    ttft_mean_ms           REAL,   -- 平均首 token 延迟
    ttft_p90_ms            REAL,
    ttft_p99_ms            REAL,
    ttft_max_ms            REAL,
    decode_latency_mean_ms REAL,   -- 平均增量 token 延迟
    decode_latency_p90_ms  REAL,
    decode_latency_p99_ms  REAL,
    decode_latency_max_ms  REAL,
    recorded_at            TEXT DEFAULT (datetime('now')),
    UNIQUE(gpu_name, model_name, gpu_count, input_tokens, output_tokens, concurrency)
);

CREATE INDEX IF NOT EXISTS idx_bdata_lookup
    ON benchmark_data(gpu_name, model_name, gpu_count, input_tokens, concurrency);
```

> **说明**：去掉外键到 `gpu_specs`/`models` 表的关联，直接用字符串存储 `gpu_name`/`model_name`，减少 JOIN 复杂度。GPU 规格信息通过 `gpu_specs` 表单独查询，用于成本计算和建模推算。

### 5.2 内存缓存策略（替代 Redis）

```python
from cachetools import TTLCache

# 全局缓存（进程内，重启后清空）
_prediction_cache = TTLCache(maxsize=1000, ttl=3600)    # 预测结果 1h
_interp_cache     = TTLCache(maxsize=100,  ttl=1800)    # 插值数据集 30min
_meta_cache       = TTLCache(maxsize=50,   ttl=600)     # GPU/模型元数据 10min

def cache_key(*args) -> str:
    return ":".join(str(a) for a in args)
```

### 5.3 数据量估算（基于 Excel 数据）

| 表 | 初始行数（从 Excel 导入）| 月增量（新测试）| SQLite 文件大小 |
|----|----------------------|----------------|----------------|
| `benchmark_data` | ~1,800 行（21 个 Sheet × 平均 ~85 行）| ~500 行/月 | < 5MB/年 |
| `benchmark_runs` | 21 条（Excel 导入批次）| ~10 次/月 | 忽略 |
| `gpu_specs` | 3 条（H200/H20/P800）| 极少 | 忽略 |
| `models` | 7 条（4B~671B）| 少量 | 忽略 |

SQLite 单文件完全满足，无需外部数据库服务。

---

## 6. 后端服务设计

### 6.1 项目目录结构

```
backend/
├── app/
│   ├── main.py                  # FastAPI 初始化、CORS、路由注册、启动事件（DB初始化+Excel导入）
│   ├── core/
│   │   ├── config.py            # 配置（环境变量 / .env）
│   │   ├── database.py          # SQLAlchemy 同步 engine（SQLite）+ Session factory
│   │   ├── cache.py             # TTLCache 实例（prediction/interp/meta）
│   │   └── logging.py           # structlog 配置
│   ├── models/                  # SQLAlchemy ORM 模型（对应 5.1 各表）
│   │   ├── gpu_spec.py
│   │   ├── model.py
│   │   ├── benchmark_run.py
│   │   └── benchmark_data.py
│   ├── schemas/                 # Pydantic v2 请求/响应 schema
│   │   ├── predict.py
│   │   ├── cost.py
│   │   └── benchmark.py
│   ├── api/
│   │   └── v1/
│   │       ├── router.py        # 聚合所有子路由
│   │       ├── predict.py       # POST /predict（性能预测）
│   │       ├── cost.py          # GET /cost/optimize
│   │       ├── benchmark.py     # 压测任务 CRUD + WebSocket 日志流
│   │       ├── data.py          # Excel/CSV 导入、数据查询
│   │       └── meta.py          # GPU/模型元数据 CRUD
│   ├── services/
│   │   ├── prediction/
│   │   │   ├── interpolation.py # 基于实测数据的插值预测（scipy）
│   │   │   ├── vllm_model.py    # 基于 vLLM 参数建模的理论推算
│   │   │   └── ensemble.py      # 混合预测（插值优先 + 建模兜底）
│   │   ├── cost_optimizer.py    # 成本优化遍历算法
│   │   ├── excel_importer.py    # 解析 资源规划工具.xlsx → SQLite
│   │   └── token_counter.py     # 分词器 Token 统计
│   └── tasks/
│       └── benchmark_runner.py  # asyncio 后台任务：subprocess + Queue → WebSocket
├── alembic/                     # 数据库迁移（Alembic 支持 SQLite）
│   ├── env.py
│   └── versions/
├── tests/
├── Dockerfile
├── pyproject.toml
└── .env.example
```

### 6.2 核心 API 接口设计

#### 6.2.1 性能预测接口

```
POST /api/v1/predict
```

**请求体：**
```json
{
  "gpu_name": "H200",
  "model_name": "Qwen2.5-72B-Instruct",
  "gpu_count": 4,
  "input_tokens": 2048,
  "output_tokens": 512,
  "concurrency": 16,
  "max_ttft_ms": 3000,
  "min_throughput_per_user": 5.0
}
```

**响应体：**
```json
{
  "source": "interpolation",        // "interpolation" | "model_based" | "ensemble"
  "confidence": 0.92,               // 置信度，建模预测时 < 0.7
  "data_points_used": 24,           // 参与插值的实测数据点数
  "result": {
    "predicted_ttft_mean_ms": 1850,
    "predicted_ttft_p90_ms": 2100,
    "predicted_throughput_tokens_s": 320,
    "max_safe_concurrency": 22,
    "recommended_concurrency": 18
  },
  "warnings": ["预测基于相邻数据点外推，误差可能偏大"],
  "metadata": {
    "interpolation_range": {
      "concurrency": [8, 32],
      "input_tokens": [1024, 4096]
    }
  }
}
```

#### 6.2.2 成本优化接口

```
GET /api/v1/cost/optimize
```

**查询参数：**
```
target_concurrency=100
model_name=Qwen2.5-72B-Instruct
input_tokens=2048
output_tokens=512
max_ttft_ms=3000
top_k=5
```

**响应体：**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "gpu_name": "H200",
      "gpu_count": 4,
      "price_per_hour": 240.0,
      "max_concurrency": 128,
      "utilization_rate": 0.78,
      "cost_per_1m_tokens": 12.5,
      "source": "interpolation",
      "confidence": 0.95
    },
    {
      "rank": 2,
      "gpu_name": "A800",
      "gpu_count": 8,
      "price_per_hour": 280.0,
      "max_concurrency": 96,
      "utilization_rate": 1.04,       // > 1 表示不满足要求
      "cost_per_1m_tokens": 18.2,
      "source": "model_based",
      "confidence": 0.65,
      "warnings": ["该组合无实测数据，为理论估算"]
    }
  ],
  "query_params": { ... }
}
```

#### 6.2.3 基准测试任务接口

```
POST /api/v1/benchmark/run        → 提交任务，返回 task_id
GET  /api/v1/benchmark/{task_id}/status  → 查询任务状态
WS   /api/v1/benchmark/{task_id}/stream  → WebSocket 实时日志
GET  /api/v1/benchmark/{task_id}/result  → 获取结果（done 后）
DELETE /api/v1/benchmark/{task_id}       → 取消任务
```

#### 6.2.4 数据管理接口

```
POST /api/v1/data/import           → 上传 CSV 批量导入
GET  /api/v1/data/coverage         → 查询数据覆盖情况（热力图数据）
GET  /api/v1/gpus                  → GPU 列表
POST /api/v1/gpus                  → 新增 GPU 规格
GET  /api/v1/models                → 模型列表
POST /api/v1/models                → 新增模型
```

### 6.3 启动事件与中间件

```python
# app/main.py
app.add_middleware(CORSMiddleware, allow_origins=["*"])  # 内部工具，放开跨域
app.add_middleware(RequestLogMiddleware)                 # structlog 请求日志（方法/路径/耗时）

@app.on_event("startup")
async def startup():
    # 1. 创建 SQLite 表（如不存在）
    create_all_tables()
    # 2. 检查是否为空库 → 自动导入 Excel
    if is_benchmark_data_empty():
        excel_path = Path(settings.EXCEL_DATA_PATH)  # 默认 /app/资源规划工具.xlsx
        if excel_path.exists():
            import_excel(excel_path)
            logger.info("Excel 数据导入完成")

# 全局异常处理（统一 JSON 错误格式）
@app.exception_handler(RequestValidationError)
@app.exception_handler(PredictionError)

# 前端静态文件挂载（生产构建嵌入镜像后）
app.mount("/", StaticFiles(directory="/app/frontend/dist", html=True), name="static")
```

---

## 7. 性能预测引擎设计（无数据推测）

> 这是本次重构的核心创新点：当没有某个 GPU + 模型组合的实测数据时，利用 vLLM 的架构参数和 Roofline 模型进行理论推算。

### 7.1 预测引擎架构

```
PredictionRequest
      │
      ▼
EnsemblePredictionEngine
      ├─ 查询 DB：有多少实测数据点覆盖该请求范围？
      │
      ├─ [数据充足: ≥4个邻近点]
      │       ▼
      │   InterpolationEngine (置信度 0.85~0.99)
      │   └─ scipy.RegularGridInterpolator（三线性插值）
      │      维度：input_tokens × concurrency × output_tokens
      │
      ├─ [数据稀疏: 1~3个邻近点]
      │       ▼
      │   HybridEngine（插值 + 建模加权，置信度 0.60~0.85）
      │
      └─ [无数据: 0个邻近点]
              ▼
          VLLMModelEngine (置信度 0.40~0.65)
          基于 vLLM 参数建模理论推算
```

### 7.2 vLLM 参数建模推算方法

#### 7.2.1 显存容量约束（判断是否可部署）

```python
def estimate_gpu_memory_requirement(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    quantization: str | None,
    max_model_len: int,
) -> MemoryEstimate:
    """
    参考 vLLM CacheConfig 的 KV Cache 计算公式
    """
    # 1. 模型权重显存
    quant_factor = {None: 2.0, "AWQ": 0.5, "GPTQ": 0.5, "FP8": 1.0}[quantization]
    weight_bytes = model.size_b * 1e9 * quant_factor  # bf16 = 2 bytes/param
    weight_gb_per_card = weight_bytes / 1e9 / gpu_count

    # 2. KV Cache 每 token 显存（来自 vLLM FullAttentionSpec 公式）
    # bytes_per_token_per_layer = 2 × num_kv_heads × head_size × dtype_bytes
    kv_dtype_bytes = 1 if kv_cache_dtype == "fp8" else 2
    kv_bytes_per_token = (
        2 * model.num_kv_heads * model.head_size * kv_dtype_bytes * model.num_layers
        / gpu_count  # TP 分片
    )

    # 3. 可用于 KV Cache 的显存
    total_gpu_mem_gb = gpu.memory_gb
    available_for_kv = (
        total_gpu_mem_gb * GPU_MEMORY_UTILIZATION   # 默认 0.9
        - weight_gb_per_card
        - FRAMEWORK_OVERHEAD_GB                     # ~1GB CUDA context etc.
    )

    # 4. 最大可容纳 token 数（即 max_num_blocks × block_size）
    max_kv_tokens = available_for_kv * 1e9 / kv_bytes_per_token

    return MemoryEstimate(
        weight_gb_per_card=weight_gb_per_card,
        available_kv_gb=available_for_kv,
        max_kv_tokens=max_kv_tokens,
        can_deploy=(available_for_kv > 0),
    )
```

#### 7.2.2 首 Token 延迟（TTFT）理论推算

TTFT 主要由 **Prefill 阶段** 决定，计算量为：

$$\text{FLOPs}_{\text{prefill}} = 2 \times N_{\text{layers}} \times (4 \times L_{\text{in}}^2 \times d + 2 \times L_{\text{in}} \times d \times d_{\text{ffn}})$$

其中：
- $L_{\text{in}}$ = 输入 token 数（prompt length）
- $d$ = hidden\_size
- $d_{\text{ffn}}$ = FFN 中间层大小（通常 $= 4d$ 或 $\frac{8}{3}d$ for SwiGLU）
- $N_{\text{layers}}$ = num\_layers

```python
def estimate_ttft(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    input_tokens: int,
    concurrency: int,
    enable_chunked_prefill: bool = True,
) -> float:
    """
    估算首 Token 延迟（ms）

    思路：
    1. 计算单个请求的 Prefill FLOPs
    2. 根据 GPU 算力和 TP 并行度估算 Prefill 时间
    3. 加入排队等待时间（受并发数影响）
    4. 加入调度开销和通信开销
    """
    # 1. Prefill FLOPs（Attention + FFN）
    d = model.hidden_size
    # Attention: Q、K、V 投影 + Attention 计算 + Output 投影
    attn_flops = 4 * input_tokens * d * d  # QKV proj + O proj（近似）
    attn_flops += 2 * input_tokens * input_tokens * d  # Attention scores
    # FFN (SwiGLU): 2.67x hidden_size ≈ d_ffn
    ffn_flops = model.num_layers * 2 * input_tokens * d * (d * 8 // 3)
    total_flops = model.num_layers * attn_flops + ffn_flops

    # 2. 有效算力（TFLOPS，考虑 TP 通信效率衰减）
    peak_tflops = gpu.bf16_tflops * gpu_count
    tp_efficiency = TP_EFFICIENCY_TABLE.get(gpu_count, 0.85)  # 经验值
    effective_tflops = peak_tflops * tp_efficiency * GPU_COMPUTE_UTILIZATION

    # 3. 计算时间（ms）
    compute_time_ms = total_flops / (effective_tflops * 1e12) * 1000

    # 4. 批处理等待：如果高并发，请求可能需要等待前面的 prefill 完成
    # 使用 M/M/1 队列模型的近似：在 chunked prefill 模式下影响减小
    if enable_chunked_prefill:
        queue_factor = 1 + (concurrency / MAX_NUM_SEQS) * 0.3
    else:
        queue_factor = 1 + (concurrency / MAX_NUM_SEQS) * 0.8

    ttft_ms = compute_time_ms * queue_factor + SCHEDULING_OVERHEAD_MS

    return ttft_ms
```

#### 7.2.3 生成吞吐量（Decode Throughput）推算

Decode 阶段是 **内存带宽瓶颈**（每步只生成1个 token，算力利用率低）：

$$\text{Throughput}_{\text{decode}} = \frac{\text{Memory\_BW} \times \eta_{\text{BW}}}{\text{bytes\_per\_token\_decode}}$$

其中 $\text{bytes\_per\_token\_decode}$ 包括：
- 模型权重：$2 \times \text{model\_params} / \text{gpu\_count}$ bytes（每步全量加载）
- KV Cache 读写：$2 \times L_{\text{seq}} \times \text{kv\_bytes\_per\_token}$

```python
def estimate_decode_throughput(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    concurrency: int,
    avg_seq_len: int,  # input_tokens + output_tokens / 2
) -> float:
    """
    估算系统级 Decode 吞吐量（tokens/s）

    Decode 阶段通常是内存带宽瓶颈（Bandwidth-Bound），
    吞吐量 ≈ GPU内存带宽 / 每生成1个token需要读取的字节数
    """
    # 每生成1个 token 需要读取的字节数
    # 1. 模型权重（全量参数，BF16）
    quant_bytes = 2 if model.quantization is None else 0.5  # AWQ/GPTQ ≈ 0.5
    weight_bytes = model.size_b * 1e9 * quant_bytes / gpu_count

    # 2. KV Cache（当前 batch 中所有序列的历史 KV）
    kv_per_token = (
        2 * model.num_kv_heads * model.head_size * 2  # bf16 KV
        * model.num_layers / gpu_count
    )
    kv_bytes = concurrency * avg_seq_len * kv_per_token

    bytes_per_decode_step = weight_bytes + kv_bytes

    # 内存带宽（GB/s → bytes/s）
    effective_bw = gpu.memory_bw_gbps * 1e9 * gpu_count * BW_EFFICIENCY

    # 系统级吞吐 = 单步并发数 / 单步时间
    step_time_s = bytes_per_decode_step / effective_bw
    throughput = concurrency / step_time_s

    return throughput
```

#### 7.2.4 最大并发数推算

```python
def estimate_max_concurrency(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    input_tokens: int,
    output_tokens: int,
    max_ttft_ms: float,
    min_throughput_per_user: float,
) -> ConcurrencyEstimate:
    """
    二分搜索满足约束的最大并发数
    """
    lo, hi = 1, 1024
    result = None

    while lo <= hi:
        mid = (lo + hi) // 2
        ttft = estimate_ttft(model, gpu, gpu_count, input_tokens, mid)
        tput = estimate_decode_throughput(model, gpu, gpu_count, mid,
                                          input_tokens + output_tokens // 2)
        per_user_tput = tput / mid

        if ttft <= max_ttft_ms and per_user_tput >= min_throughput_per_user:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return ConcurrencyEstimate(
        max_concurrency=result or 0,
        feasible=(result is not None),
        bottleneck="memory" if result else "compute",
    )
```

### 7.3 经验系数校准（利用现有测试数据）

通过已有实测数据对建模预测进行系数校准，减小理论误差：

```python
class ModelCalibrator:
    """
    用现有测试数据校准理论模型的经验系数
    """

    def calibrate(self, benchmark_data: list[BenchmarkDataPoint]) -> CalibrationCoefs:
        """
        对每个 (gpu, model) 组合，拟合：
          ttft_predicted × α = ttft_measured
          throughput_predicted × β = throughput_measured

        返回 {(gpu_name, model_name): (α, β)} 映射
        """
        coefs = {}
        for (gpu, model), points in groupby(benchmark_data, key=lambda x: (x.gpu, x.model)):
            predicted = [estimate_ttft(..., p) for p in points]
            measured = [p.ttft_mean_ms for p in points]
            alpha = np.mean([m / p for m, p in zip(measured, predicted) if p > 0])
            coefs[(gpu, model)] = CalibrationCoef(alpha=alpha)
        return coefs
```

### 7.4 置信度评分

```python
def compute_confidence(
    request: PredictionRequest,
    data_points: list[BenchmarkDataPoint],
    prediction_source: str,
) -> float:
    score = 1.0

    if prediction_source == "model_based":
        score = 0.55  # 基础置信度较低
    elif prediction_source == "interpolation":
        # 外推比内插置信度低
        if is_extrapolation(request, data_points):
            score = 0.70
        else:
            score = 0.90

    # 相邻数据点数量加权
    score *= min(1.0, len(data_points) / 10)

    # 参数差距惩罚（请求参数与最近邻数据点的距离）
    nearest_distance = compute_parameter_distance(request, data_points[0])
    score *= max(0.4, 1.0 - nearest_distance * 0.3)

    return round(min(0.99, score), 2)
```

---

## 8. 前端应用设计

### 8.1 页面结构

```
/                → 重定向到 /planner
/planner         → 资源规划（性能预测 + 成本优化，合并为一页）
/benchmark       → 基准测试（提交任务 + 历史记录 + 实时日志）
/data            → 数据管理（CSV导入、覆盖率热力图、数据明细）
/settings        → 元数据管理（GPU规格、模型信息、价格配置）
```

### 8.2 核心页面：资源规划器 `/planner`

```
┌─────────────────────────────────────────────────────────────┐
│  资源规划器                                          [导出报告]│
├──────────────────┬──────────────────────────────────────────┤
│   参数配置面板    │              结果展示区                    │
│                  │                                          │
│  模型选择        │  ┌─────────────────────────────────────┐ │
│  [Qwen2.5-72B ▼] │  │  并发数 vs 首Token延迟（折线图）       │ │
│                  │  │  可交互：拖拽查看、标注约束线           │ │
│  GPU 型号        │  └─────────────────────────────────────┘ │
│  [H200      ▼]   │                                          │
│                  │  ┌──────────────────────────────────────┐│
│  GPU 数量        │  │  TOP 5 推荐方案（对比表格）             ││
│  [ 4         ]   │  │  ┌─────┬──────┬───────┬──────┬─────┐ ││
│                  │  │  │方案 │GPU  │最大并发│成本/h│置信 │ ││
│  输入 token      │  │  │ 1   │H200 │  128  │ ¥240 │ 95%│ ││
│  [── 2048 ──]    │  │  │ 2   │A800 │   96  │ ¥280 │ 65%│ ││
│                  │  │  └─────┴──────┴───────┴──────┴─────┘ ││
│  输出 token      │  └──────────────────────────────────────┘│
│  [── 512  ──]    │                                          │
│                  │  ┌──────────────────────────────────────┐│
│  延迟约束        │  │  成本对比柱状图（ECharts，可交互）       ││
│  最大TTFT: 3s    │  └──────────────────────────────────────┘│
│  最小吞吐: 5t/s  │                                          │
│                  │  ⚠ 部分方案基于理论推算，置信度 < 0.7      │
│  [计算最优方案]   │     建议对低置信度方案运行基准测试           │
│  [运行基准测试]   │                                          │
└──────────────────┴──────────────────────────────────────────┘
```

**关键 UX 设计：**
- 数据来源徽章：`实测数据` / `理论估算` 不同颜色标注
- 置信度进度条显示在每个推荐方案旁
- 无数据的方案显示"运行基准测试"快捷入口
- ECharts 折线图支持：鼠标悬停显示数值、多系列对比、导出PNG/CSV

### 8.3 核心页面：基准测试 `/benchmark`

```
┌──────────────────────────────────────────────────────────────┐
│  基准测试                                                      │
├────────────────────────┬─────────────────────────────────────┤
│   新建测试任务           │  测试历史                            │
│                        │  ┌──────────────────────────────┐   │
│  服务地址: [http://...] │  │ 时间    GPU  模型  状态   操作 │   │
│  模型名称: [Qwen2...]   │  │ 04-29  H200  72B  ✓完成 查看 │   │
│  分词器:  [/model/...] │  │ 04-28  A800  32B  ⚡运行 查看 │   │
│                        │  │ 04-27  4090  7B   ✗失败 重试 │   │
│  并发数梯度:            │  └──────────────────────────────┘   │
│  [1,4,8,16,32,64]      │                                     │
│  输入长度:              │  实时日志（WebSocket）               │
│  [512,1024,2048,4096]  │  ┌──────────────────────────────┐   │
│  输出长度:              │  │ [14:23:01] Warmup epoch 1/3  │   │
│  [256,256,256,256]     │  │ [14:23:15] 并发=1, 输入=512   │   │
│                        │  │   TTFT: 245ms, 吞吐: 38 t/s  │   │
│  最大首Token延迟: [3000]│  │ [14:23:40] 并发=4, 输入=512   │   │
│                        │  │ ...                          │   │
│  [提交测试任务]         │  └──────────────────────────────┘   │
└────────────────────────┴─────────────────────────────────────┘
```

### 8.4 核心页面：数据管理 `/data`

```
┌──────────────────────────────────────────────────────────────┐
│  数据管理                                                      │
├──────────────────────────────────────────────────────────────┤
│  数据覆盖热力图                             [导入CSV] [刷新]    │
│                                                              │
│  纵轴：GPU型号 (4090/A40/A800/H20/H200/910B)                  │
│  横轴：模型 × 卡数 (7B×1, 7B×2, 32B×4, 72B×4...)            │
│  颜色：数据点数量 (0=空白, 1-9=浅绿, 10-49=中绿, 50+=深绿)    │
│                                                              │
│  ┌────┬────┬────┬────┬────┬─────┬────┐                      │
│  │    │7B×1│32B×4│72B×4│72B×8│72B-AWQ×4│32B-AWQ×4│          │
│  │H200│ 45 │  38 │  52 │  28 │    31   │   22    │          │
│  │H20 │  0 │  31 │  48 │  22 │    26   │   18    │          │
│  │A800│  0 │  28 │   0 │   0 │    24   │   19    │          │
│  │4090│ 33 │  35 │   0 │   0 │     0   │   28    │          │
│  └────┴────┴────┴────┴────┴─────┴────┘                      │
│                                                              │
│  点击任意格子 → 查看该组合的详细数据列表（分页表格）              │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. 基准测试集成

### 9.1 asyncio 后台任务设计

替代 Celery+Redis，使用 Python 内置的 asyncio + subprocess + Queue：

```python
# tasks/benchmark_runner.py
import asyncio, subprocess, uuid
from pathlib import Path

# 全局任务状态表（进程内，重启后清空）
_task_queues: dict[str, asyncio.Queue] = {}
_task_status: dict[str, str] = {}          # task_id → pending/running/done/failed

async def run_benchmark(task_id: str, config: BenchmarkConfig):
    """
    asyncio 后台任务：调用 benchmark_parallel.py 并实时推流日志
    任务状态同步写入 SQLite（重启可查历史）
    """
    queue = asyncio.Queue()
    _task_queues[task_id] = queue
    _task_status[task_id] = "running"
    update_task_status_db(task_id, "running")  # 同步写 SQLite

    cmd = build_benchmark_command(config)  # 与现有 benchmark_parallel.py 参数兼容
    loop = asyncio.get_event_loop()

    def _stream_subprocess():
        """在线程池中运行 subprocess，逐行推送到 Queue"""
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in process.stdout:
            loop.call_soon_threadsafe(queue.put_nowait, line.strip())
        process.wait()
        return process.returncode

    try:
        rc = await loop.run_in_executor(None, _stream_subprocess)
        if rc == 0:
            # 解析结果 CSV 并入库（UPSERT）
            import_csv_to_sqlite(config.output_csv, task_id, config.gpu_name,
                                  config.model_name, config.gpu_count)
            _task_status[task_id] = "done"
            update_task_status_db(task_id, "done")
        else:
            _task_status[task_id] = "failed"
            update_task_status_db(task_id, "failed")
    except Exception as e:
        _task_status[task_id] = "failed"
        update_task_status_db(task_id, "failed", str(e))
    finally:
        await queue.put(None)  # 结束信号
```

### 9.2 WebSocket 实时日志推流

```python
# api/v1/benchmark.py
@router.post("/run")
async def submit_benchmark(config: BenchmarkConfig, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    create_task_db(task_id, config)          # 写入 SQLite task_status
    background_tasks.add_task(run_benchmark, task_id, config)
    return {"task_id": task_id}

@router.websocket("/{task_id}/stream")
async def stream_log(websocket: WebSocket, task_id: str):
    await websocket.accept()
    queue = _task_queues.get(task_id)

    if queue is None:
        # 任务已结束（历史任务），直接返回状态
        status = get_task_status_db(task_id)
        await websocket.send_json({"type": "end", "status": status or "not_found"})
        return

    try:
        while True:
            line = await asyncio.wait_for(queue.get(), timeout=60)
            if line is None:  # 结束信号
                await websocket.send_json({"type": "end",
                                           "status": _task_status.get(task_id)})
                break
            await websocket.send_json({"type": "log", "content": line})
    except (asyncio.TimeoutError, WebSocketDisconnect):
        pass
```

### 9.3 Excel 数据解析导入

Sheet 命名规则：`{model_size}-{GPU}-{gpu_count}测试数据`（如 `72B-H200-4测试数据`）

```python
# services/excel_importer.py
import re
import openpyxl
import pandas as pd

SHEET_PATTERN = re.compile(
    r'^(\d+B(?:-A\d+B)?)'   # model_size: 4B / 235B-A22B / 671B
    r'-(H200|H20|P800)'      # gpu_name
    r'-(\d+)'                # gpu_count
    r'测试\s*数据$'           # 后缀（允许有空格）
)

COLUMN_MAPPING = {
    "输入长度": "input_tokens",
    "输出长度": "output_tokens",
    "并发数": "concurrency",
    "输出tokens总吞吐": "output_throughput",
    "首tokens时延TP90（ms）": "ttft_p90_ms",
    "首tokens时延TP99（ms）": "ttft_p99_ms",
    "最大首tokens时延（ms）": "ttft_max_ms",
    "平均首tokens时延（ms）": "ttft_mean_ms",
    "增量时延TP90（ms）": "decode_latency_p90_ms",
    "增量时延TP99（ms）": "decode_latency_p99_ms",
    "最大增量时延（ms）": "decode_latency_max_ms",
    "平均增量时延（ms）": "decode_latency_mean_ms",
}

def import_excel(excel_path: Path) -> dict:
    """解析 Excel 所有测试 Sheet，批量写入 SQLite"""
    wb = openpyxl.load_workbook(excel_path, data_only=True, read_only=True)
    stats = {"sheets": 0, "rows": 0, "skipped": []}

    for sheet_name in wb.sheetnames:
        m = SHEET_PATTERN.match(sheet_name)
        if not m:
            stats["skipped"].append(sheet_name)
            continue

        model_name, gpu_name, gpu_count = m.group(1), m.group(2), int(m.group(3))

        # 用 pandas 读取，自动处理空行
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        df = df.iloc[:, :12]  # 只取前12列（忽略 32B-P800-4 中的额外空列）
        df.columns = list(COLUMN_MAPPING.keys())
        df = df.dropna(subset=["input_tokens", "并发数"])
        df = df.rename(columns=COLUMN_MAPPING)
        df["gpu_name"] = gpu_name
        df["model_name"] = model_name
        df["gpu_count"] = gpu_count
        df["output_tokens"] = df["output_tokens"].round().astype(int)
        df["input_tokens"] = df["input_tokens"].astype(int)
        df["concurrency"] = df["concurrency"].astype(int)

        # UPSERT（ON CONFLICT REPLACE）
        upsert_benchmark_data(df)
        stats["sheets"] += 1
        stats["rows"] += len(df)

    return stats
```

**Excel Sheet 覆盖范围**（来自 `资源规划工具.xlsx`）：

| GPU | 模型 | 卡数 | 数据行数 |
|-----|------|------|--------|
| P800 | 7B | 1 | ~103 |
| P800 | 14B | 2 | ~91 |
| P800 | 32B | 4 | ~100 |
| P800 | 72B | 8 | ~88 |
| P800 | 671B | 8 | ~51 |
| H200 | 4B | 1 | ~139 |
| H200 | 7B | 1 | ~152 |
| H200 | 14B | 1 | ~130 |
| H200 | 32B | 1 | ~96 |
| H200 | 32B | 2 | ~132 |
| H200 | 72B | 2 | ~104 |
| H200 | 72B | 4 | ~134 |
| H200 | 235B-A22B | 4 | ~113 |
| H20 | 4B | 1 | ~133 |
| H20 | 7B | 1 | ~127 |
| H20 | 14B | 1 | ~100 |
| H20 | 32B | 1 | ~64 |
| H20 | 32B | 2 | ~94 |
| H20 | 72B | 2 | ~61 |
| H20 | 72B | 4 | ~89 |
| H20 | 235B-A22B | 4 | ~111 |
| **合计** | — | — | **~2,112 行** |

---

## 10. 部署方案

### 10.1 单容器 Dockerfile

前端 build 直接嵌入镜像，无需额外容器：

```dockerfile
# Dockerfile（多阶段构建）

# ── Stage 1: 前端构建 ──────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # 输出到 /frontend/dist

# ── Stage 2: 最终镜像 ─────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# 系统依赖（运行 benchmark_tools 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY backend/pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 应用代码
COPY backend/app/ ./app/
COPY backend/alembic/ ./alembic/
COPY backend/alembic.ini ./

# 前端静态文件（嵌入镜像）
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# 数据目录（会被 volume 覆盖）
RUN mkdir -p /data

# Excel 数据文件（种子数据，首次启动自动导入）
COPY 资源规划工具.xlsx ./资源规划工具.xlsx

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.2 启动命令（一行）

```bash
# 生产启动（数据持久化到宿主机 ./data 目录）
docker run -d \
  --name rpt \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -v $(pwd)/model:/app/model:ro \
  -v $(pwd)/benchmark_tools:/app/benchmark_tools:ro \
  -e SQLITE_PATH=/data/rpt.db \
  -e EXCEL_DATA_PATH=/app/资源规划工具.xlsx \
  rpt:latest

# 访问 http://localhost:8000
```

### 10.3 环境变量配置

```bash
# .env.example
SQLITE_PATH=/data/rpt.db                        # SQLite 文件路径
EXCEL_DATA_PATH=/app/资源规划工具.xlsx          # 初始化数据 Excel
BENCHMARK_TOOLS_DIR=/app/benchmark_tools        # 压测脚本目录
MODEL_DIR=/app/model                            # 分词器目录

# 内存缓存 TTL（秒）
PREDICTION_CACHE_TTL=3600

# vLLM 建模经验系数
GPU_MEMORY_UTILIZATION=0.9
TP_EFFICIENCY_4GPU=0.88
TP_EFFICIENCY_8GPU=0.82
```

### 10.4 本地开发启动

```bash
# 1. 安装后端依赖
cd backend && pip install -e .

# 2. 初始化 SQLite（自动）+ 导入 Excel（自动，首次）
export SQLITE_PATH=./data/rpt.db
export EXCEL_DATA_PATH=../资源规划工具.xlsx
uvicorn app.main:app --reload --port 8000

# 3. 启动前端开发服务器（另一个终端）
cd frontend && npm run dev   # 代理到 localhost:8000
```

> 开发模式无需 Docker，无需任何外部服务，`pip install` + `uvicorn` 即可运行。

---

## 11. 迁移计划

### 11.1 分阶段实施

#### Phase 1：后端重构（2周）

```
Week 1:
  ✓ 创建 SQLite Schema + Alembic 迁移脚本
  ✓ 实现 excel_importer.py（解析 资源规划工具.xlsx → SQLite）
  ✓ 验证 21 个 Sheet 全部导入正确（2,112 行数据）
  ✓ 重构 server.py → 分层 FastAPI 目录结构
  ✓ 实现插值预测引擎（基于 SQLite 数据，替换现有 performance.py）

Week 2:
  ✓ 实现 vLLM 参数建模预测引擎（vllm_model.py）
  ✓ 实现 EnsemblePredictionEngine（混合预测）
  ✓ 实现成本优化器（替换现有 visualization.py 逻辑）
  ✓ asyncio 基准测试后台任务 + WebSocket 日志推流
  ✓ API 单元测试（pytest）
```

#### Phase 2：前端重构（2周）

```
Week 3:
  ✓ React + Ant Design + Vite 项目脚手架
  ✓ 资源规划器页面（ECharts 交互式图表 + 推荐方案表格）
  ✓ API 集成（React Query）

Week 4:
  ✓ 基准测试页面（WebSocket 实时日志）
  ✓ 数据管理页面（Excel/CSV 导入 + 覆盖率热力图）
  ✓ 元数据设置页面（GPU/模型管理）
```

#### Phase 3：集成与收尾（3天）

```
  ✓ 多阶段 Dockerfile 构建验证
  ✓ docker run 单命令启动测试
  ✓ 文档更新（API 文档 /docs、部署 README）
```

### 11.2 数据来源变更

**废弃**：`data/` 目录下所有 CSV 文件（测试时未启用 prefix caching，数据无效）

**新数据源**：`资源规划工具.xlsx`

```bash
# 手动触发重新导入（清空现有数据后重导）
curl -X POST http://localhost:8000/api/v1/data/reimport
# 或直接删除 SQLite 文件，重启容器自动重导
rm ./data/rpt.db && docker restart rpt
```

### 11.3 向后兼容性

- 保留现有 `server.py` 的 `/healthz`、`/concurrency`、`/cost` 接口，通过适配层接入新后端
- `client.py` 无需修改（仍可正常使用）
- `benchmark_tools/` 脚本**完全不修改**，通过 asyncio 后台任务调用

---

## 附录：数据字典

### A1. 现有 CSV 列与 DB 字段映射

| CSV 列名 | DB 字段名 | 类型 | 单位 |
|---------|---------|------|------|
| 输入长度 | input_tokens | INT | tokens |
| 输出长度 | output_tokens | INT | tokens |
| 并发数 | concurrency | INT | — |
| 输出tokens总吞吐 | output_throughput | FLOAT | tokens/s |
| 首tokens时延TP90（ms） | ttft_p90_ms | FLOAT | ms |
| 首tokens时延TP99（ms） | ttft_p99_ms | FLOAT | ms |
| 最大首tokens时延（ms） | ttft_max_ms | FLOAT | ms |
| 平均首tokens时延（ms） | ttft_mean_ms | FLOAT | ms |
| 增量时延TP90（ms） | decode_latency_p90_ms | FLOAT | ms |
| 增量时延TP99（ms） | decode_latency_p99_ms | FLOAT | ms |
| 最大增量时延（ms） | decode_latency_max_ms | FLOAT | ms |
| 平均增量时延（ms） | decode_latency_mean_ms | FLOAT | ms |

### A2. 初始 GPU 型号数据（来自 Excel 测试数据）

| GPU | 供应商 | 显存 | 内存带宽 | BF16算力 | 参考单价(¥/卡·h) |
|-----|--------|------|---------|---------|----------------|
| H200 | NVIDIA | 141GB | 4800 GB/s | 989 TFLOPS | 60 |
| H20 | NVIDIA | 96GB | 4000 GB/s | 296 TFLOPS | 30 |
| P800 | Huawei | 64GB | 2000 GB/s | 280 TFLOPS | 20 |

> P800 参数为估算值（昇腾 910B 系列服务器），实际部署时可在设置页面更新。

### A3. 初始模型数据（来自 Excel 测试数据）

| 模型标识 | 参数量 | 架构 | num_layers | hidden_size | num_kv_heads | 备注 |
|---------|------|------|-----------|------------|-------------|------|
| 4B | 4B | Qwen3 | 36 | 2560 | 8 | Qwen3-4B |
| 7B | 7B | Qwen2 | 28 | 3584 | 4 | Qwen2.5-7B |
| 14B | 14B | Qwen2 | 40 | 5120 | 8 | Qwen2.5-14B |
| 32B | 32B | Qwen2 | 64 | 5120 | 8 | Qwen2.5-32B |
| 72B | 72B | Qwen2 | 80 | 8192 | 8 | Qwen2.5-72B |
| 235B-A22B | 235B | Qwen3MoE | 94 | 7168 | 4 | Qwen3-235B-A22B（MoE）|
| 671B | 671B | DeepSeek MoE | 61 | 7168 | 128 | DeepSeek-R1-671B（MoE）|

### A4. vLLM 参数建模经验系数

| 系数 | 默认值 | 说明 |
|------|--------|------|
| `GPU_MEMORY_UTILIZATION` | 0.9 | vLLM 默认显存利用率 |
| `FRAMEWORK_OVERHEAD_GB` | 1.5 | CUDA context + PyTorch 框架开销 |
| `GPU_COMPUTE_UTILIZATION` | 0.70 | 实际算力利用率（相对理论峰值） |
| `BW_EFFICIENCY` | 0.80 | 内存带宽实际利用率 |
| `TP_EFFICIENCY_TABLE` | {1:1.0, 2:0.92, 4:0.88, 8:0.82} | TP 通信效率衰减 |
| `SCHEDULING_OVERHEAD_MS` | 5.0 | vLLM 调度器固定开销 |
| `MAX_NUM_SEQS` | 256 | 非 H100/H200 时的默认最大序列数 |

---

*文档版本：v2.0 | 最后更新：2026-04-29*
