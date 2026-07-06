---
name: generic-performance-testing
description: 需要创建、运行或排查独立的 HTTP/OpenAI 兼容性能测试、压测配置、延迟/吞吐摘要，或生成 CSV/JSON/Markdown 报告，且不能依赖项目专用 benchmark 框架。
---

# 通用性能测试

这个 skill 提供一套独立的性能测试工作流，适用于 HTTP API、OpenAI 兼容的 Chat/Completion 接口，以及轻量级 ASR 请求规划。它不导入、不依赖父级 benchmark 项目，可以作为完整文件夹单独复制和运行。

## 工作流程

1. **运行前确认需求**：先和用户确认测试目标、接口类型、`base_url`、`model`、认证方式、数据集、`requests`、`concurrency`、是否只做 dry-run、输出目录和风险边界。
2. 在用户确认后，从 `configs/` 选择一个接近目标 API 的配置，或复制后按需修改。配置细节参考 `references/config-guide.md`。
3. 将文本 prompt 或音频 manifest 样本放入 `datasets/`。
4. 先运行 `scripts/run_auto.sh --config configs/openai_chat.json --dry-run`，在真正发请求前检查生成的请求内容。
5. 用户确认 dry-run 结果后，再运行 `scripts/run_auto.sh --config <config>` 执行性能测试。
6. 使用 `scripts/run_manual.sh --url <url> --body '{"prompt":"hello"}'` 做单请求冒烟验证。
7. 查看 `reports/results.json`、`reports/results.csv` 和 `reports/summary.md`，也可以通过 `--output-dir` 指定其它输出目录。

## Shell 使用方式

自动化性能测试 dry-run：

```bash
scripts/run_auto.sh --config configs/openai_chat.json --dry-run --limit 3
```

自动化性能测试正式执行：

```bash
scripts/run_auto.sh --config configs/generic_http.json --output-dir reports/generic-smoke
```

手动单请求测试：

```bash
scripts/run_manual.sh \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --header "Authorization: Bearer token" \
  --body '{"model":"demo","messages":[{"role":"user","content":"hello"}]}'
```

如果 shell 包装脚本需要使用指定 Python 解释器，设置 `PYTHON_BIN=/path/to/python`。

## 配置文件

更完整的配置说明、字段含义、示例选择和运行前确认清单见 `references/config-guide.md`。

每个配置文件包含三个顶层字段：

- `run`：测试名称、请求数、并发数、预热请求数、超时时间和输出目录。
- `target`：接口类型、基础 URL、接口路径、模型名称、请求头，以及可选的通用请求体模板。
- `dataset`：JSONL 数据集路径，内容可以是文本 prompt，也可以是音频 manifest 样本。

支持的 `target.type`：

- `openai-chat`：生成 `/v1/chat/completions` 风格的 JSON 请求体。
- `openai-completion`：生成 `/v1/completions` 风格的 JSON 请求体。
- `openai-asr`：为音频转写接口生成 JSON 请求计划；建议先用 dry-run 检查文件路径和 payload 结构。
- `generic-http`：使用样本字段渲染 `body_template`，例如 `{prompt}`。

相对数据集路径会从 skill 包根目录解析，因此整个目录可以作为自包含包移动。

## 报告输出

`scripts/perf_auto.py` 会写入：

- `results.json`：完整摘要和逐请求记录。
- `results.csv`：适合导入表格工具的逐请求明细。
- `summary.md`：便于阅读的延迟和吞吐摘要。

脚本实现仅使用 Python 标准库模块。
