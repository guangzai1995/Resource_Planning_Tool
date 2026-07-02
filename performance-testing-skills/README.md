# 性能测试 Skills

`performance-testing-skills` 是一个独立可携带的模型 API 性能测试工具包，也是一个 independent portable package。它 does not depend on this repository；可以 copy 到任意机器或任意项目目录中使用，只要保持包内目录结构不变，并从包根目录执行脚本即可。

本工具包不负责启动、停止、部署或管理目标服务。请先自行启动 API 服务，再用这里的脚本验证请求格式、执行 smoke 检查，并在你明确选择的测试条件下测量性能。

## 目录结构

```text
performance-testing-skills/
  skills/      Codex 或 Claude 可读取的手动测试、自动压测 skill 说明
  configs/     每种支持协议的 JSON 配置示例
  datasets/    文本 JSONL prompt 示例和音频 manifest JSONL 示例
  scripts/     CLI 入口、shell wrapper 和共享 Python helper
  reports/     默认报告输出目录，通常为 reports/latest
  tests/       包结构、dry-run、client、metrics 和本地 HTTP 测试
```

## 支持协议

- `openai_chat`：POST 到 `<base_url>/chat/completions`
- `openai_completion`：POST 到 `<base_url>/completions`
- `openai_asr`：multipart POST 到 `<base_url>/audio/transcriptions`
- `generic_http`：使用 `method`、`url`、`headers` 和 `body_template` 直接构造 HTTP 请求

## 快速开始

复制工具包或 checkout 仓库后，请从包根目录执行以下命令。

手动 dry-run：只构造请求并打印 curl，不发送网络流量。

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run --input "hello dry run"
```

手动 curl 模式：只打印等价 curl 命令。

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --mode curl --input "hello curl"
```

自动压测 dry-run：打印压测计划，不发送请求。

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --dry-run --concurrency 1,2 --epochs 2
```

自动压测执行：发送请求并写出报告。

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2 --epochs 2 --output-dir reports/latest
```

如果你更希望直接调用 Python 入口，也可以这样执行：

```bash
python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run
python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run
```

## 配置说明

每个配置文件都是一个 JSON 对象。相对路径统一按包根目录解析。

- `protocol`：协议类型，取值为 `openai_chat`、`openai_completion`、`openai_asr` 或 `generic_http`。
- `base_url`：OpenAI-compatible 协议的 API 基础 URL，脚本会自动拼接 endpoint path。
- `model`：OpenAI-compatible 请求体或 ASR multipart 字段中的模型名。
- `headers`：HTTP headers。类似 `"Bearer ${API_KEY}"` 的值会从环境变量展开。
- `request`：可选的协议专属字段，会合并到生成的 OpenAI-compatible body 或 multipart form 中。
- `dataset`：包含 `type` 和 `path` 的对象，例如 `text_prompts` 或 `audio_manifest`。
- `body_template`：`generic_http` 的 JSON body 模板。它不会做环境变量展开；其中 `${prompt}` 和 `${max_tokens}` 是样本占位符，会由 dataset 行填充，不是 shell 环境变量。

环境变量示例：

```bash
export API_KEY="..."
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run
```

对于 `generic_http`，如下模板：

```json
{
  "prompt": "${prompt}",
  "max_tokens": "${max_tokens}"
}
```

会使用每条 dataset 样本中的 `prompt` 和 `expected_output_len`。

## 数据集

文本数据集是 JSONL 文件，每行是一条样本：

```jsonl
{"prompt": "Write a short introduction to API benchmarking.", "expected_output_len": 128, "metadata": {"case": "intro"}}
```

常用字段：

- `prompt`：chat、completion 或 generic HTTP 模板使用的文本输入。
- `expected_output_len`：作为 `generic_http` 中 `${max_tokens}` 的默认值。
- `metadata`：可选的调用方自定义信息，用于追踪样本来源或场景。

ASR 音频 manifest 也是 JSONL 文件：

```jsonl
{"audio": "datasets/audio/sample.wav", "prompt": "Transcribe the audio.", "reference": "example transcript", "duration_s": 12.3}
```

常用字段：

- `audio`：作为 multipart `file` 上传的音频文件路径。
- `prompt`：可选 ASR prompt。
- `reference`：可选的期望转写结果，便于人工对比。
- `duration_s`：可选音频时长，会记录到请求指标中；失败请求不会计入 ASR RTFX。

## 手动接口测试

自动压测前建议先做手动接口测试，用来验证 endpoint 形状、鉴权、ASR 上传格式和 curl 复现方式。

参数：

- `--config`：配置 JSON 路径，默认 `configs/openai_chat.json`。
- `--mode request|curl`：`request` 会发送或 dry-run 请求；`curl` 只打印 curl。默认 `request`。
- `--input`：覆盖第一条 dataset 样本的文本 prompt。
- `--audio-file`：覆盖第一条 ASR manifest 样本的音频路径。
- `--request-count`：request 模式下发送的请求数，默认 `1`。
- `--print-curl`：发送请求前打印 curl。
- `--save-response`：将响应 JSON 写入文件。
- `--dry-run`：只构造并打印请求，不发送网络流量。

推荐给 Codex 或 Claude 调用的命令：

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run --input "smoke test"
./scripts/run_manual.sh --config configs/openai_chat.json --mode curl --input "smoke test"
./scripts/run_manual.sh --config configs/openai_asr.json --audio-file path/to/audio.wav --print-curl --save-response reports/manual-asr.json
```

## 自动性能测试

手动 smoke 通过后，再使用自动性能测试。

参数：

- `--config`：配置 JSON 路径，默认 `configs/openai_chat.json`。
- `--dataset`：覆盖配置中的 dataset 路径。
- `--concurrency`：并发列表，支持逗号或空格分隔，例如 `1,2,4` 或 `1 2 4`。
- `--epochs`：未设置 `--duration-seconds` 时，每个并发 worker 的请求数。
- `--duration-seconds`：可选的按时间压测模式；每个并发档会持续提交替换请求直到计时结束。总请求数只有运行后才能确定，请以 `metrics.json`、`metrics.csv` 或 `summary.md` 为准。
- `--max-error-rate`：可选错误率阈值，范围 `0` 到 `1`。
- `--max-p90-latency-ms`：可选 p90 延迟阈值，单位毫秒。
- `--fail-fast`：当某档 0 请求完成、100% 请求失败，或违反 `--max-error-rate` 时提前停止后续并发档。p90 threshold does not trigger fail-fast；违反 `--max-p90-latency-ms` 会被记录，并在 sweep 完成后让命令以非 0 退出。
- `--output-dir`：报告输出目录，默认 `reports/latest`。
- `--dry-run`：只打印计划，不发送网络流量。

推荐给 Codex 或 Claude 调用的命令：

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --dry-run --concurrency 1,2 --epochs 2
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2 --epochs 2 --max-error-rate 0.05 --fail-fast --output-dir reports/latest
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2 --epochs 2 --max-p90-latency-ms 2000 --output-dir reports/latency-check
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2,4 --duration-seconds 30 --output-dir reports/sweep
```

建议流程：

1. 先运行手动 dry-run 和 curl 检查。
2. 再运行自动压测 dry-run，确认压测计划。
3. 先跑一个小 smoke 档，例如 `--concurrency 1 --epochs 1`。
4. 最后按计划执行正式 sweep，指定并发、epochs 或 duration、阈值和输出目录。duration 模式下，报告才是总请求数的事实来源。

## 报告输出

自动压测会在 `--output-dir` 下写出这些文件：

- `summary.md`：人类可读的结论、稳定并发、峰值吞吐并发档、过载起点和指标表。
- `metrics.json`：每个并发档的聚合指标。
- `metrics.csv`：同一份聚合指标的 CSV 版本。
- `requests.jsonl`：每个请求一行的标准化结果。
- `errors.jsonl`：只包含失败请求行。

建议阅读顺序：

1. 先看 `summary.md`，确认结论、stable/peak/overload 并发档。
2. 再看 `metrics.json` 或 `metrics.csv`，关注成功率、错误率、延迟分位数、吞吐和请求数。
3. 查看 `errors.jsonl`，定位重复出现的 `auth_error`、`bad_request`、`not_found`、`file_not_found` 或 timeout。
4. 需要逐请求延迟或响应摘要时，再看 `requests.jsonl`。

100% request failure 不是性能瓶颈。如果所有请求都失败，应先修接口：检查 `base_url`、endpoint path、凭据、模型名、请求体、ASR 文件路径和目标服务可用性。不要把全失败场景下的吞吐或延迟解释为服务容量。

## 注意事项

- 不要把鉴权失败、授权失败、bad request、文件缺失或路由错误当作性能瓶颈。
- 自动压测前始终先运行手动 smoke 检查。
- 本工具包只测量你指向的目标服务；不负责目标服务启动、停止、预热、部署或扩缩容。
- 不要把结论外推到未测试的模型、硬件、数据集、并发、持续时间、请求体或服务配置。
