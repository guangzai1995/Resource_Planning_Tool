## server.py 服务说明

本说明文档面向 `server.py` 提供的 FastAPI 服务，涵盖目录结构、功能说明、使用方法与依赖配置，便于快速本地启动与集成调用。

---

## 目录结构（与服务相关）

仅列出与 HTTP 服务直接相关的关键文件/目录，完整工程请参考仓库根目录。

```
Resource_Planning_Tool/
├─ server.py                # FastAPI 服务入口，定义各 HTTP 路由
├─ client.py                # 命令行客户端，便于调用服务接口
├─ requirements.txt         # Python 依赖清单
├─ src/
│  ├─ __init__.py
│  ├─ data_loader.py        # 加载性能数据（GPU→模型→卡数→输入长度→指标）
│  ├─ performance.py        # 并发/成本等核心计算逻辑
│  ├─ token_processor.py    # 分词器初始化与 Token 统计
│  ├─ utils.py              # 工具函数（列出可用分词器等）
│  └─ visualization.py      # 可视化（与服务返回解耦，仅供扩展）
├─ model/                   # 本地分词器/模型目录（如 Qwen* 等）
└─ data/                    # 性能数据（按 GPU / 模型 / 卡数 / 输入长度 分层）
```

---

## 功能说明（HTTP API）

服务采用 JSON 输入/输出（application/json）。统一约定：

- 所有响应体都包含 `status_code: number` 字段，值与 HTTP 状态码一致。
- 计算/查询类接口通常包含 `success: boolean` 与 `data`/`error` 字段；简单信息接口（如 `/healthz`、`/tokenizers`）无 `success` 字段。
- 失败响应体包含 `error: { message, reason, ... }`；`reason` 常见取值：`data_missing` / `constraints_unsatisfied` / `calculation_error` / `invalid`。

### 1) 健康检查
- 路由：`GET /healthz`（亦兼容 `POST /healthz`）
- 功能：返回服务存活与版本信息。

请求参数：无

成功响应（200）：
```
{
  "status_code": 200,
  "status": "ok",
  "service": "resource-planning",
  "version": "1.0.0"
}
```

### 2) 列出可用分词器
- 路由：`GET /tokenizers`（亦兼容 `POST /tokenizers`）
- 功能：列出 `model/` 目录下可用分词器名称数组。

请求参数：无

成功响应（200）：
```
{
  "status_code": 200,
  "tokenizers": ["Qwen2___5-0___5B-Instruct", "..."]
}
```

### 3) 枚举可选项树（GPU/模型/卡数/输入长度）
- 路由：`GET /options`
- 功能：枚举可选配置的层级树；可按层级过滤并返回可选列表。

请求参数（Query，可选）：

| 名称        | 类型  | 必填 | 说明                                 |
|-------------|-------|------|--------------------------------------|
| gpu_model   | str   | 否   | GPU 型号过滤，如 4090、A100、H200 等 |
| model_name  | str   | 否   | 模型名称过滤，如 32B、72B、7B 等      |
| card_count  | int   | 否   | 卡数过滤，如 1、2、4、8               |

成功响应（200）：
```
{
  "status_code": 200,
  "success": true,
  "data": {
    "filters": {"gpu_model": "4090", "model_name": "32B", "card_count": 8},
    "available": {
      "gpus": ["4090", "A100", ...],
      "models": ["32B", "32B-AWQ", ...],
      "card_counts": [4, 8],
      "input_lengths": [2048, 4096, 8192, 12000, 16000, 20000]
    },
    "options": [
      {
        "gpu_model": "4090",
        "models": [
          {
            "model_name": "32B",
            "card_counts": [
              { "card_count": 8, "input_lengths": [2048, 4096, ...] }
            ]
          }
        ]
      }
    ]
  }
}
```

失败响应（500）：
```
{
  "status_code": 500,
  "success": false,
  "error": {"message": "内部错误: ...", "reason": "calculation_error"}
}
```

### 4) 计算最大并发
- 路由：`POST /concurrency/calculate`
- 功能：根据性能曲线、延时与单用户吞吐要求，计算最大可达并发与相关指标。

请求体（JSON）：

| 名称                | 类型   | 必填 | 说明                                   |
|---------------------|--------|------|----------------------------------------|
| gpu_model           | str    | 是   | GPU 型号（如 4090、A100、H200 等）      |
| model_name          | str    | 是   | 模型名称（如 32B、72B、7B 等）          |
| card_count          | int    | 是   | 卡数（节点内）                          |
| input_length        | int    | 是   | 输入 token 长度（上下文长度）           |
| max_delay           | float  | 否   | 首 token 延时上限（秒），默认 5.0       |
| user_throughput     | float  | 否   | 单用户吞吐要求（tokens/s），默认 10.0   |
| output_length       | int    | 否   | 生成 token 长度，默认 1200              |

示例请求：
```
{
  "gpu_model": "4090",
  "model_name": "32B",
  "card_count": 8,
  "input_length": 2048,
  "max_delay": 5.0,
  "user_throughput": 10.0,
  "output_length": 1200
}
```

成功响应（200）：
```
{
  "status_code": 200,
  "success": true,
  "data": {
    "gpu_model": "4090",
    "model_name": "32B",
    "card_count": 8,
    "input_length": 2048,
    "output_length": 1200,
    "max_delay": 5.0,
    "user_throughput_requirement": 10.0,
    "max_concurrency": 123,
    "per_user_throughput": 0.12,
    "total_throughput": 14.7,
    "interpolation_used": false,
    "boundary": { "max_acceptable_concurrency": 120.5, "estimated_total_throughput": 14.4 }
  }
}
```

失败响应（400/500）：
```
{
  "status_code": 400,
  "success": false,
  "error": {
    "message": "错误：未找到所选配置的性能数据",
    "reason": "data_missing | constraints_unsatisfied | calculation_error | invalid",
    "suggestions": "4090/32B/8卡 可用输入长度: 2048, 4096, ..." | null,
    "input": { 同请求体回显 }
  }
}
```

### 5) 计算输入 Token 统计
- 路由：`POST /tokens/calculate`
- 功能：计算系统提示词与用户输入的 tokens/字符统计；支持指定本地分词器目录。

请求体（JSON）：

| 名称           | 类型 | 必填 | 说明                                     |
|----------------|------|------|------------------------------------------|
| system_prompt  | str  | 否   | 系统提示词文本                           |
| user_prompt    | str  | 否   | 用户输入文本                             |
| tokenizer_name | str  | 否   | 分词器目录名（位于 `model/` 下）          |

示例请求：
```
{ "system_prompt": "你好", "user_prompt": "今天天气如何", "tokenizer_name": "Qwen2___5-0___5B-Instruct" }
```

成功响应（200）：
```
{
  "status_code": 200,
  "success": true,
  "data": {
    "tokenizer": "Qwen2___5-0___5B-Instruct",
    "status": "已加载",
    "system_tokens": 12,
    "user_tokens": 34,
    "total_tokens": 46,
    "system_chars": 10,
    "user_chars": 20,
    "total_chars": 30
  }
}
```

失败响应（500）：
```
{
  "status_code": 500,
  "success": false,
  "error": {"message": "内部错误: ...", "reason": "calculation_error"}
}
```

### 6) 成本优化
- 路由：`POST /cost/optimize`
- 功能：为给定目标并发与上下文长度，寻找成本最低的 TOP3 方案。

请求体（JSON）：

| 名称               | 类型  | 必填 | 说明                         |
|--------------------|-------|------|------------------------------|
| target_concurrency | int   | 是   | 目标最大并发用户数           |
| model_name         | str   | 是   | 模型名称（如 32B、72B 等）   |
| context_length     | int   | 是   | 上下文长度（输入 token 长度） |
| max_delay          | float | 否   | 首 token 延时上限（秒）       |

示例请求：
```
{ "target_concurrency": 50, "model_name": "32B", "context_length": 2048, "max_delay": 5.0 }
```

成功响应（200）：
```
{
  "status_code": 200,
  "success": true,
  "data": {
    "solutions": [
      {
        "gpu_model": "A800",
        "card_count_per_node": 4,
        "nodes_required": 2,
        "total_cards": 8,
        "total_price": 123456.0,
        "max_conc_per_node": 28
      }
    ]
  }
}
```

无可行解（404）：
```
{
  "status_code": 404,
  "success": false,
  "error": {"message": "未找到符合约束的方案", "reason": "data_missing", "input": { ... }}
}
```

内部错误（500）：
```
{
  "status_code": 500,
  "success": false,
  "error": {"message": "内部错误: ...", "reason": "calculation_error"}
}
```

---

## 使用说明

### 方式一：直接运行 `server.py`

无需额外参数，使用内置的 `uvicorn.run` 启动：

```bash
python server.py
```

支持环境变量 `PORT` 指定端口（默认 15422）：

```bash
PORT=15422 python server.py
```

### 方式二：使用 uvicorn 命令

```bash
uvicorn server:app --host 0.0.0.0 --port 15422
```

### 验证接口（示例）

- 健康检查：
```bash
curl -s http://localhost:15422/healthz | jq .
```

- 枚举可选项树：
```bash
curl -s "http://localhost:15422/options?gpu_model=4090&model_name=32B&card_count=8" | jq .
```

- 计算最大并发：
```bash
curl -s -X POST http://localhost:15422/concurrency/calculate \
  -H 'Content-Type: application/json' \
  -d '{"gpu_model":"4090","model_name":"32B","card_count":8,"input_length":2048,"max_delay":5,"user_throughput":10,"output_length":1200}' | jq .
```

### 使用命令行客户端 `client.py`

```bash
# 健康检查（GET）
python client.py healthz

# 列出分词器（GET）
python client.py tokenizers

# 枚举可选项（GET）
python client.py options --gpu-model 4090 --model-name 32B --cards 8

# 并发计算（POST）
python client.py concurrency --gpu-model 4090 --model-name 32B --cards 8 \
  --input-length 2048 --max-delay 5 --user-throughput 10 --output-length 1200

# Token 统计（POST）
python client.py tokens --system "你好" --user "今天天气如何" --tokenizer-name Qwen2___5-0___5B-Instruct

# 成本优化（POST）
python client.py cost --target 50 --model-name 32B --context-length 2048 --max-delay 5
```

---

## 依赖配置

- 运行环境：Python 3.10+（建议）
- 安装依赖：
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

核心依赖（与服务直接相关）：
- fastapi / starlette / pydantic：Web 框架与模型定义
- uvicorn：ASGI 服务器
- requests：仅 `client.py` 需要（HTTP 客户端）

> 说明：项目 `requirements.txt` 较为全面，包含可视化、NLP、笔记本等扩展依赖；若仅运行服务与客户端，安装全量依赖即可，或根据需要精简。

---

## 错误处理与返回约定

- 成功：`{"success": true, "data": {...}}`
- 失败：`{"success": false, "error": {"message": str, "reason": str, "suggestions"?: str, "input"?: {...}}}`
- 常见 `reason`：`data_missing` / `constraints_unsatisfied` / `calculation_error` / `invalid`
- `/options` 在过滤时会在 `available` 中给出各层级的可选项，便于客户端引导用户选择。

---

## 维护者提示

- 性能数据由 `src/data_loader.py` 读入，结构：`gpu_model -> model_name -> card_count -> input_length -> metrics`。
- 并发/成本计算由 `src/performance.py` 提供，服务只做 JSON 封装与错误统一处理。
- 分词器位于 `model/` 下的子目录，`/tokens/calculate` 可指定 `tokenizer_name`，未指定使用默认 `Qwen2___5-0___5B-Instruct`；若模型未就绪会降级为字符计数。
