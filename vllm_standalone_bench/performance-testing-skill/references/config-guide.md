# 配置文件辅助指南

本文档用于在运行性能测试前，帮助确认用户需求并正确设置 `configs/*.json`。除非用户明确只要求生成文档或配置，否则在正式发起请求前必须先完成需求确认和 dry-run。

## 运行前需求确认清单

正式运行测试前，先和用户交互确认以下信息：

| 确认项 | 需要确认的内容 |
|---|---|
| 测试目标 | 是验证链路、冒烟测试、容量评估、对比测试，还是排查延迟/吞吐问题。 |
| 接口类型 | 选择 `openai-chat`、`openai-completion`、`openai-asr` 或 `generic-http`。 |
| 服务地址 | `base_url` 和 `endpoint` 是否正确，目标服务是否已经启动。 |
| 模型 | `model` 名称是否和服务端实际加载的模型一致。 |
| 认证 | 是否需要 `Authorization`、内部网关 token 或其它请求头。 |
| 数据集 | 使用已有 `datasets/*.jsonl`，还是需要用户提供 prompt/audio manifest。 |
| 压测规模 | `requests`、`concurrency`、`warmup_requests` 和 `request_timeout_sec`。 |
| 输出位置 | `run.output_dir` 或命令行 `--output-dir`。 |
| 运行模式 | 先执行 `dry-run`；只有用户确认 dry-run 请求计划后，再正式运行。 |
| 风险边界 | 是否允许对真实服务施压、是否有 QPS/并发上限、是否需要避开生产环境。 |

确认后可以用一句话复述计划，例如：

```text
我将使用 openai-chat 配置，请求 http://127.0.0.1:8000/v1/chat/completions，
模型 demo-chat-model，requests=20，concurrency=2，先 dry-run，不会立即压测。
```

只有用户确认后，才进入下一步。

## 配置文件选择

从最接近目标接口的示例复制一份新配置，再修改字段：

| 配置文件 | 适用场景 |
|---|---|
| `configs/openai_chat.json` | OpenAI Chat Completions 兼容接口，例如 `/v1/chat/completions`。 |
| `configs/openai_completion.json` | OpenAI Completions 兼容接口，例如 `/v1/completions`。 |
| `configs/openai_asr.json` | OpenAI Audio transcription / ASR 请求计划。 |
| `configs/generic_http.json` | 任意 JSON HTTP POST 接口，通过 `body_template` 渲染请求体。 |

建议不要直接改示例文件用于长期场景；复制为带场景名的新文件更容易复现。

## 顶层字段说明

每个配置包含 `run`、`target`、`dataset` 三个顶层字段。

### run

| 字段 | 说明 |
|---|---|
| `name` | 本次测试名称，会写入报告摘要。 |
| `requests` | 正式计入报告的请求总数。 |
| `concurrency` | 并发请求数。首次调试建议从 `1` 或 `2` 开始。 |
| `warmup_requests` | 预热请求数，不计入最终结果。 |
| `request_timeout_sec` | 单请求超时时间。ASR 或长输出场景应适当调大。 |
| `output_dir` | 默认报告输出目录，可被命令行 `--output-dir` 覆盖。 |

### target

| 字段 | 说明 |
|---|---|
| `type` | 请求构造类型：`openai-chat`、`openai-completion`、`openai-asr`、`generic-http`。 |
| `base_url` | 服务基础地址，例如 `http://127.0.0.1:8000`。 |
| `endpoint` | 接口路径，例如 `/v1/chat/completions`。 |
| `method` | HTTP 方法，默认 `POST`。 |
| `model` | 传给服务端的模型名。`generic-http` 可不设置。 |
| `headers` | 请求头，认证通常写在这里。 |
| `extra_body` | OpenAI 兼容请求体的额外字段，例如 `max_tokens`、`temperature`。 |
| `body_template` | 仅 `generic-http` 使用，支持 `{prompt}`、`{case}` 等样本字段替换。 |

认证示例：

```json
{
  "headers": {
    "Authorization": "Bearer <token>"
  }
}
```

### dataset

| 字段 | 说明 |
|---|---|
| `type` | `jsonl` 或 `audio_manifest`。 |
| `path` | 数据集路径。相对路径从 skill 包根目录解析。 |

文本样本示例：

```jsonl
{"prompt":"Say hello in one sentence.","case":"short","metadata":{"case":"short"}}
```

音频 manifest 示例：

```jsonl
{"audio_path":"samples/hello.wav","prompt":"Transcribe the short greeting.","metadata":{"case":"short"}}
```

## target.type 配置差异

### openai-chat

生成如下请求体：

```json
{
  "model": "demo-chat-model",
  "messages": [
    {"role": "user", "content": "样本 prompt"}
  ],
  "max_tokens": 32,
  "temperature": 0
}
```

适合 Chat Completions 兼容服务。需要重点确认 `model`、`endpoint`、`extra_body.max_tokens`。

### openai-completion

生成如下请求体：

```json
{
  "model": "demo-completion-model",
  "prompt": "样本 prompt",
  "max_tokens": 32,
  "temperature": 0
}
```

适合老式 Completions 接口或只接受 `prompt` 字段的服务。

### openai-asr

生成音频转写请求计划：

```json
{
  "model": "demo-asr-model",
  "audio_path": "samples/hello.wav",
  "prompt": "Transcribe the short greeting."
}
```

正式运行前必须 dry-run，确认音频路径、模型名和接口路径无误。

### generic-http

通过 `body_template` 把数据集字段渲染成请求体：

```json
{
  "body_template": {
    "prompt": "{prompt}",
    "case": "{case}",
    "source": "performance-testing-skill"
  }
}
```

适合非 OpenAI 协议的 JSON HTTP 接口。

## 推荐运行流程

1. 和用户完成“运行前需求确认清单”。
2. 复制最接近的配置文件并修改 `run`、`target`、`dataset`。
3. 执行 dry-run：

   ```bash
   scripts/run_auto.sh --config configs/openai_chat.json --dry-run --limit 3
   ```

4. 查看 `dry_run_plan.json`，向用户确认 URL、headers、body、请求数量和并发设置。
5. 用户确认后，再正式运行：

   ```bash
   scripts/run_auto.sh --config configs/openai_chat.json --output-dir reports/openai-chat-smoke
   ```

6. 查看 `results.json`、`results.csv`、`summary.md`，并说明失败请求、延迟分位数和吞吐结果。

## 常见问题

| 现象 | 处理方式 |
|---|---|
| `401` 或 `403` | 检查 `target.headers.Authorization` 或网关认证。 |
| `404` | 检查 `base_url` 和 `endpoint` 拼接后的完整 URL。 |
| 全部请求超时 | 降低 `concurrency`，调大 `request_timeout_sec`，确认服务是否可访问。 |
| dry-run 请求体字段不符合预期 | 检查 `target.type`、`extra_body` 或 `body_template`。 |
| 报告目录不对 | 检查 `run.output_dir`，或在命令行显式设置 `--output-dir`。 |
