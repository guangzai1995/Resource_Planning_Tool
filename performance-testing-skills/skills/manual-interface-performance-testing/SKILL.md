---
name: manual-interface-performance-testing
description: 用于手动验证文本、ASR 或 generic HTTP 模型 API 请求，包括 smoke 测试、小批量请求、curl 生成和请求结构调试。
---

# 手动接口性能测试

当用户需要在自动压测前验证请求结构、生成 curl、测试一个或少量请求、验证 ASR multipart 上传、查看错误或调试目标 API 时，使用此 skill。

手动检查回答的是“这个请求能不能工作”。它不能证明性能瓶颈，也不能证明容量上限。

## 当前 CLI 模式

当前手动 CLI 只支持这些模式：

- `--mode request`：构造并发送请求；配合 `--dry-run` 时只构造并打印请求。这是默认模式。
- `--mode curl`：只打印等价 curl 命令，不发送网络流量。

不要使用旧示例里的 smoke mode。需要 smoke 行为时，使用 `--mode request --request-count 1`，必要时加 `--print-curl`。

## 必需流程

1. 识别协议和配置：`openai_chat`、`openai_completion`、`openai_asr` 或 `generic_http`。
2. 确认目标服务已启动，并确认 credentials、`base_url` 或 `url`、model 和 dataset path 都是预期值。
3. 先运行 `--dry-run`，在不发送网络流量的情况下检查 method、URL、protocol 和 curl。
4. 当用户需要可复现命令用于另一个终端或问题报告时，使用 `--mode curl`。
5. 用 `--mode request --request-count 1` 发送一个真实请求；有必要时加 `--print-curl`。
6. 对 ASR，如果 manifest 中的样本不是要上传的音频，用 `--audio-file` 指定目标文件。
7. 调试 status code、response body 或错误分类时，用 `--save-response` 保存响应。
8. 先诊断错误，再运行自动化负载测试。

## 推荐命令

从包根目录使用 wrapper：

```bash
./scripts/run_manual.sh --config configs/openai_chat.json --dry-run --input "smoke test"
./scripts/run_manual.sh --config configs/openai_chat.json --mode curl --input "smoke test"
./scripts/run_manual.sh --config configs/openai_chat.json --mode request --request-count 1 --print-curl --save-response reports/manual-chat.json
./scripts/run_manual.sh --config configs/openai_asr.json --mode request --audio-file path/to/audio.wav --print-curl --save-response reports/manual-asr.json
```

也可以直接使用 Python 入口：

```bash
python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run --input "hello"
python3 scripts/perf_manual.py --config configs/openai_chat.json --mode curl --input "hello"
python3 scripts/perf_manual.py --config configs/generic_http.json --mode request --request-count 1 --print-curl
python3 scripts/perf_manual.py --config configs/openai_asr.json --audio-file path/to/audio.wav --save-response reports/manual-asr.json
```

## 输出解释

Dry-run 输出包含：

- `protocol`：选择的 client 行为。
- `method`：HTTP method。
- `url`：解析后的 endpoint。
- `curl`：用于复现的等价命令。

请求输出每个请求一行，包含：

- `request_id`：手动请求编号。
- `success`：HTTP 响应是否为 2xx。
- `status_code`：收到响应时的 HTTP status。
- `latency_ms`：请求耗时。
- `response_summary`：提取的文本或简短响应体摘要。
- `error_type` 和 `error_message`：请求失败时出现。

使用 `--save-response PATH` 时，CLI 会写出包含 `requests` 数组的 JSON 对象。在 `--dry-run` 或 `--mode curl` 下，因为没有发送请求，数组为空。

## 错误诊断

- `auth_error`：检查 headers、tokens、scopes 和环境变量展开。
- `bad_request`：检查 body shape、model name、request parameters 和 ASR multipart 字段。
- `not_found`：检查 `base_url`、`url` 和 endpoint path。
- `file_not_found` 或 `file_read_error`：检查 ASR `audio` manifest path 或 `--audio-file`。
- `timeout`、`connection_reset` 或 `disconnect`：增加请求数前先确认目标服务可用。

不要把手动请求失败称为性能瓶颈。先修接口或服务可用性问题。

## 何时切换到自动化测试

只有在满足以下条件后，才切换到 automated-performance-testing skill：

- Dry-run 展示的请求结构符合预期。
- Curl 模式生成的命令已被用户接受。
- 一个真实请求已经成功；或者用户明确选择为了诊断而压测一个已知失败条件。
- 数据集、并发、epochs 或 duration、阈值和输出目录已经确认。

如果手动结果是 401、403、400、404、ASR 文件缺失，或小批量下 100% request failure，应继续停留在手动诊断阶段。
