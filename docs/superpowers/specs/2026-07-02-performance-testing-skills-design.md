# 可迁移性能测试 Skills 包设计

日期：2026-07-02

## 背景

当前项目已有针对 vLLM standalone benchmark 的文本和 ASR 压测脚本，但这些能力与本工程目录、数据集和运行约定绑定较深。用户希望沉淀一套可复用的 performance testing skills 包，能被 Codex、Claude 或类似 AI 编码助手在任意项目中调用，用于自动化性能测试和非自动化接口测试。

该包需要覆盖文本模型、语音 ASR 模型以及通用 HTTP 推理接口。它不应依赖本仓库路径，也不应要求目标服务是本工程的一部分。

## 目标

1. 生成一个可迁移的 skills 包目录，可复制到任意项目使用。
2. 包内包含两个独立 skill：
   - 自动化性能测试 skill。
   - 非自动化接口测试 skill。
3. 包内包含必要脚本，可由 Codex、Claude 或人工直接运行。
4. 支持 OpenAI 兼容协议模板：
   - `/v1/chat/completions`
   - `/v1/completions`
   - `/v1/audio/transcriptions`
5. 支持通用 HTTP 配置，不强制接口符合 OpenAI 协议。
6. 支持文本、语音 ASR，并为多模态数据集格式预留扩展位。
7. 自动化测试输出结构化报告，包含吞吐、延迟、成功率、错误类型和瓶颈判断。
8. 非自动化测试支持 smoke test、单请求、小批量请求和等价 curl 输出。

## 非目标

1. 不绑定本仓库的 `vllm_standalone_bench`、模型目录或内网服务地址。
2. 不内置大型音频、图片或文本数据集。
3. 不实现复杂前端、Dashboard 或服务端控制台。
4. 不承担模型精度评测，例如 WER、CER、BLEU、ROUGE。
5. 不自动启动或管理被测模型服务。
6. 不默认要求 Docker、Kubernetes 或云厂商环境。

## 交付形态

生成目录包，暂定名称为 `performance-testing-skills`：

```text
performance-testing-skills/
  README.md
  skills/
    automated-performance-testing/
      SKILL.md
    manual-interface-performance-testing/
      SKILL.md
  scripts/
    perf_auto.py
    perf_manual.py
    run_auto.sh
    run_manual.sh
    lib/
      clients.py
      datasets.py
      metrics.py
      reporters.py
  configs/
    openai_chat.json
    openai_completion.json
    openai_asr.json
    generic_http.json
  datasets/
    text_prompts.example.jsonl
    audio_manifest.example.jsonl
  reports/
    .gitkeep
```

包内所有相对路径都以包根目录为基准。复制整个目录后，不需要本仓库即可运行。

## Skill 职责边界

### automated-performance-testing

用于自动化压测和性能回归。触发场景包括：

- 用户要求压测、性能测试、并发阶梯、吞吐分析、瓶颈分析。
- 用户需要对文本模型、ASR 模型或通用 HTTP 推理接口做批量请求。
- 用户要求生成 CSV、JSON、Markdown 报告。

该 skill 必须优先指导助手调用包内脚本，而不是临时手写循环请求。执行后必须分析成功率、吞吐、延迟拐点、错误类型、推荐稳定并发和瓶颈区间。

### manual-interface-performance-testing

用于非自动化接口验证。触发场景包括：

- 用户要求试一下接口是否可用。
- 用户要求发一条或几条请求。
- 用户要求生成 curl。
- 用户需要验证文本、ASR 或通用 HTTP 接口的请求格式。

该 skill 默认只做 smoke test 或小批量验证，不给出完整压测结论。它可以作为自动化压测前的前置诊断步骤。

## 脚本分层

脚本分为共享库、Python CLI 和 Shell 包装器三层。

共享库：

- `scripts/lib/clients.py`：OpenAI Chat、OpenAI Completion、OpenAI ASR、Generic HTTP 客户端。
- `scripts/lib/datasets.py`：文本 JSONL、音频 manifest JSONL、多模态预留格式读取。
- `scripts/lib/metrics.py`：请求级指标和聚合指标计算。
- `scripts/lib/reporters.py`：JSON、CSV、Markdown 报告输出。

Python CLI：

- `scripts/perf_auto.py`：自动化并发阶梯压测入口。
- `scripts/perf_manual.py`：非自动化接口验证入口。

Shell 包装器：

- `scripts/run_auto.sh`：为自动化压测提供稳定 shell 入口。
- `scripts/run_manual.sh`：为非自动化接口验证提供稳定 shell 入口。

Shell 包装器只做参数转发、当前目录定位和 Python 解释器选择，不承载核心逻辑。

## 自动化脚本 CLI

`perf_auto.py` 支持：

```bash
python3 scripts/perf_auto.py \
  --config configs/openai_asr.json \
  --concurrency 1,4,8,16,32 \
  --epochs 3 \
  --output-dir reports/asr-run
```

核心参数：

- `--config`：接口、数据集、压测和报告配置。
- `--dataset`：覆盖配置中的数据集路径。
- `--concurrency`：覆盖配置中的并发阶梯，例如 `1,4,8,16,32`。
- `--epochs`：每个并发档的轮数。
- `--duration-seconds`：按固定时间压测，与 `--epochs` 二选一。
- `--output-dir`：报告目录。
- `--fail-fast`：某一档失败率或延迟超过阈值后停止后续高并发。
- `--max-error-rate`：最大允许失败率，默认 `0.05`。
- `--max-p90-latency-ms`：可选 P90 延迟阈值。
- `--dry-run`：只打印展开后的配置和计划，不发送请求。

## 非自动化脚本 CLI

`perf_manual.py` 支持：

```bash
python3 scripts/perf_manual.py \
  --config configs/openai_chat.json \
  --mode smoke \
  --input "hello"
```

核心参数：

- `--config`：接口配置。
- `--mode`：`smoke`、`single`、`small-batch` 或 `curl`。
- `--request-count`：小批量请求数。
- `--input`：临时文本输入。
- `--audio-file`：临时语音文件。
- `--print-curl`：打印等价 curl。
- `--save-response`：保存响应 JSON 或文本。
- `--dry-run`：只展示请求计划，不发送请求。

`curl` 模式不发送请求，只输出可复制的 curl 命令。

## 配置格式

配置文件统一使用 JSON，避免引入 YAML 解析依赖。脚本必须支持环境变量展开，例如 `${API_KEY}`。

OpenAI ASR 示例：

```json
{
  "name": "openai_asr_example",
  "protocol": "openai_asr",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "Qwen3-ASR",
  "headers": {
    "Authorization": "Bearer ${API_KEY}"
  },
  "request": {
    "language": "en",
    "temperature": 0
  },
  "dataset": {
    "type": "audio_manifest",
    "path": "datasets/audio_manifest.example.jsonl"
  },
  "bench": {
    "concurrency": [1, 4, 8, 16],
    "epochs": 3,
    "timeout_seconds": 600
  },
  "report": {
    "output_dir": "reports",
    "formats": ["json", "csv", "md"]
  }
}
```

Generic HTTP 示例：

```json
{
  "name": "generic_http_example",
  "protocol": "generic_http",
  "method": "POST",
  "url": "http://127.0.0.1:8000/custom/infer",
  "headers": {
    "Content-Type": "application/json"
  },
  "body_template": {
    "model": "${model}",
    "input": "${prompt}",
    "max_tokens": "${max_tokens}"
  },
  "dataset": {
    "type": "text_prompts",
    "path": "datasets/text_prompts.example.jsonl"
  }
}
```

## 数据集格式

文本 JSONL：

```jsonl
{"prompt": "Write a short introduction to benchmarking.", "expected_output_len": 128, "metadata": {"case": "intro"}}
```

语音 manifest JSONL：

```jsonl
{"audio": "datasets/audio/sample.wav", "prompt": "Transcribe the audio.", "reference": "example transcript", "duration_s": 12.3}
```

多模态预留 JSONL：

```jsonl
{"prompt": "Describe this image.", "image": "datasets/images/sample.jpg"}
```

第一版只需要提供文本和语音示例文件。多模态格式在文档中保留，不要求实现图片上传。

## 自动化测试流程

`perf_auto.py` 执行流程：

1. 读取配置并展开环境变量。
2. 读取数据集。
3. 执行 1 次 smoke request，验证接口、鉴权、模型名和请求体。
4. 展开并发阶梯。
5. 对每个并发档生成请求数：

   ```text
   request_count = concurrency * epochs
   ```

6. 使用异步请求或线程池保持指定最大在途请求数。
7. 记录每个请求的状态、延迟、错误和响应摘要。
8. 每档计算聚合指标。
9. 根据阈值判断是否 fail-fast。
10. 生成报告。

如果配置使用 `duration_seconds`，请求数不再由 `epochs` 决定，而是在指定时间内持续补齐并发窗口。

## 非自动化测试流程

`perf_manual.py` 执行流程：

1. 读取配置并展开环境变量。
2. 根据 `--input`、`--audio-file` 或数据集首条记录生成请求。
3. 如果 `--print-curl` 或 `--mode curl`，输出等价 curl。
4. 如果不是 dry run，发送请求。
5. 输出 URL、method、status code、latency、响应摘要和错误诊断。
6. 如果 `--request-count` 大于 1，执行小批量稳定性验证。

非自动化测试不生成瓶颈结论，只报告接口是否可用和请求格式是否正确。

## 指标设计

请求级指标：

- `request_id`
- `success`
- `status_code`
- `error_type`
- `latency_ms`
- `ttft_ms`，仅流式或协议支持时记录。
- `input_tokens`，响应 usage 支持时记录。
- `output_tokens`，响应 usage 支持时记录。
- `audio_duration_s`，ASR 请求可用时记录。

聚合指标：

- `request_throughput_req_s`
- `output_throughput_tok_s`
- `success_rate`
- `error_rate`
- `latency_mean_ms`
- `latency_p50_ms`
- `latency_p90_ms`
- `latency_p99_ms`
- `ttft_mean_ms`
- `ttft_p90_ms`
- `tpot_mean_ms`

ASR 特有指标：

- `audio_duration_s_total`
- `audio_duration_s_avg`
- `rtfx`

RTFx 定义：

```text
rtfx = total_audio_duration_s / benchmark_duration_s
```

## 瓶颈判断规则

自动化报告需要给出可读判断，但不得把配置错误或全部请求失败误判为性能瓶颈。

规则：

1. 吞吐平台化：并发翻倍后吞吐提升低于 20% 到 30%。
2. 延迟拐点：P90 latency 或 TTFT 比上一档增长超过 50%。
3. 失败出现：`error_rate > 0`。
4. 硬瓶颈：吞吐下降或失败率超过阈值。
5. 稳定并发：最后一个 `error_rate == 0` 且 P90 未明显恶化的档位。
6. 极限并发：吞吐最高且失败率仍可接受的档位。
7. 禁用区间：吞吐下降或失败率显著上升的档位。

报告示例：

```markdown
## 结论

- 推荐稳定并发：16
- 极限吞吐并发：32
- 64 开始出现断连，判定为过载

## 关键证据

| concurrency | success_rate | req/s | p90 latency | rtfx |
```

## 错误分类

统一错误类型用于请求级 `errors.jsonl`：

- `config_error`：配置缺失、协议不支持、字段类型错误。
- `dataset_error`：数据集文件不存在、JSONL 格式错误、音频文件缺失。
- `auth_error`：HTTP 401 或 403。
- `not_found`：HTTP 404，常见于 base_url 或 path 错误。
- `bad_request`：HTTP 400 或 422。
- `timeout`：请求超时。
- `server_disconnected`：服务端断开连接。
- `broken_pipe`：客户端写入时连接中断。
- `http_5xx`：服务端 5xx。
- `unknown_error`：未分类异常。

错误记录格式：

```json
{
  "request_id": "req-0001",
  "phase": "request",
  "error_type": "server_disconnected",
  "message": "Server disconnected",
  "latency_ms": 120003
}
```

## 报告输出

自动化测试报告目录：

```text
reports/
  <run_id>/
    summary.md
    metrics.csv
    metrics.json
    requests.jsonl
    errors.jsonl
    responses/
```

`summary.md` 面向人和 AI 助手阅读，包含结论、表格、瓶颈判断和建议。`metrics.csv` 用于 Excel 或后续分析。`metrics.json` 保存完整结构化结果。`requests.jsonl` 保存请求级指标。`errors.jsonl` 保存失败详情。

非自动化测试默认只输出终端摘要。设置 `--save-response` 时写入 `reports/manual-<timestamp>/`。

## 依赖策略

第一版优先使用 Python 标准库：

- `urllib.request` 或 `http.client` 可作为同步 HTTP 基线。
- `json`、`csv`、`statistics`、`time`、`concurrent.futures`、`asyncio` 使用标准库。

可选依赖：

- 如果环境安装 `aiohttp`，自动化脚本可使用异步客户端提高并发效率。
- 如果环境安装 `soundfile`，ASR 可读取真实音频时长；否则使用 manifest 中的 `duration_s` 或在报告中标记未知。

脚本必须在依赖缺失时给出明确错误或降级说明，不应静默失败。

## 验收标准

1. 包目录可从本仓库复制到任意空目录。
2. 两个 `SKILL.md` 均包含完整 frontmatter。
3. `python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run` 可运行并输出请求计划。
4. `python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run` 可运行并输出并发计划。
5. `bash scripts/run_manual.sh --config configs/openai_chat.json --dry-run` 可运行。
6. `bash scripts/run_auto.sh --config configs/openai_chat.json --dry-run` 可运行。
7. 示例配置不包含真实内网地址和密钥。
8. 示例数据集不依赖大型二进制文件。
9. 自动化脚本能生成 `summary.md`、`metrics.csv`、`metrics.json` 的报告结构。
10. README 说明如何复制、安装、运行和让 Codex/Claude 使用两个 skill。

## 后续实现计划范围

实现计划应按以下顺序拆分：

1. 创建可迁移包目录和两个 skill 文档。
2. 实现共享配置读取、环境变量展开和数据集读取。
3. 实现 manual 脚本和 shell 包装器。
4. 实现 auto 脚本、并发执行和指标聚合。
5. 实现报告输出和瓶颈判断。
6. 添加 dry-run 和本地单元测试。
7. 验证包从仓库复制到临时目录后仍可运行。
