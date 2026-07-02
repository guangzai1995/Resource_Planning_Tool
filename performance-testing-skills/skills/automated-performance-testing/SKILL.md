---
name: automated-performance-testing
description: 用于对文本、ASR 或 generic HTTP 模型 API 运行自动化性能基准测试，包括并发 sweep、吞吐分析、延迟分析和瓶颈判断。
---

# 自动化性能测试

当用户需要对模型 API 做负载测试、性能基准测试、并发 sweep、吞吐、延迟、瓶颈分析、阈值检查或性能回归验证时，使用此 skill。

不要把自动化压测作为第一次接口检查。如果请求结构、鉴权、ASR 上传格式或 endpoint path 还不确定，先切换到 manual-interface-performance-testing skill 做手动接口验证。

## 运行前确认

发送负载前先确认这些输入：

1. 目标服务：host、port、endpoint path、model name、credentials，以及服务是否已经启动。
2. 协议：`openai_chat`、`openai_completion`、`openai_asr` 或 `generic_http`。
3. 数据集：文本 JSONL prompts 或音频 manifest JSONL，以及是否有 override path。
4. 负载形状：`--concurrency` 列表，以及使用 `--epochs` 还是 `--duration-seconds`。
5. 阈值：可选 `--max-error-rate`、`--max-p90-latency-ms`，以及是否使用 `--fail-fast`。
6. 输出位置：`--output-dir`，尤其是在对比多次运行时。

`--fail-fast` 只会在某个并发档 0 请求完成、100% request failure，或违反 `--max-error-rate` 时提前停止。p90 阈值不会触发 fail-fast；违反 `--max-p90-latency-ms` 会被记录，并在 sweep 完成后让命令以非 0 退出。

## 必需流程

1. 先运行手动 smoke 检查，或确认手动 smoke 已经成功。
2. 运行自动压测 dry-run，检查 config、dataset path、output directory、concurrency tiers 和 request counts。duration 模式下，dry-run 会标记这是 duration-based 计划，因为最终请求数取决于响应耗时。
3. 先跑小并发 smoke 档，通常是 `--concurrency 1 --epochs 1`。
4. 按确认过的并发、epochs 或 duration、阈值和报告目录执行正式 sweep。
5. 在做任何瓶颈判断前，先阅读生成的报告。

## 推荐脚本用法

优先使用包内 wrapper，而不是临时手写请求循环：

```bash
./scripts/run_auto.sh --config configs/openai_chat.json --dry-run --concurrency 1,2 --epochs 2
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1 --epochs 1 --output-dir reports/smoke
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2,4,8 --epochs 5 --max-error-rate 0.05 --fail-fast --output-dir reports/sweep
./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2,4,8 --epochs 5 --max-p90-latency-ms 2000 --output-dir reports/latency-check
```

也可以直接使用 Python 入口：

```bash
python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run
python3 scripts/perf_auto.py --config configs/openai_asr.json --concurrency 1,2 --epochs 3 --output-dir reports/asr-sweep
python3 scripts/perf_auto.py --config configs/generic_http.json --concurrency 1,2,4 --duration-seconds 30 --fail-fast --output-dir reports/generic-http
```

使用 `--duration-seconds` 时，runner 会持续提交替换请求，直到当前并发档计时结束。总请求数以 `metrics.json`、`metrics.csv` 和 `summary.md` 为准。

## 阅读报告

自动化运行会在 `--output-dir` 下写出报告：

- `summary.md`：人类可读的结论、稳定并发、峰值吞吐并发档、过载档和指标表。
- `metrics.json`：每个并发档的聚合指标。
- `metrics.csv`：适合电子表格查看的聚合指标。
- `requests.jsonl`：每个标准化请求结果一行。
- `errors.jsonl`：只包含失败请求行。

按这个顺序阅读报告：

1. 先看 `summary.md`，识别稳定并发、峰值吞吐、过载起点和是否存在全失败并发档。
2. 再看 `metrics.json` 或 `metrics.csv`，关注成功率、错误率、p50/p90/p99 延迟、吞吐和请求数。
3. 检查 `errors.jsonl`，定位重复出现的 `auth_error`、`bad_request`、`not_found`、`file_not_found`、timeout 或连接类错误。
4. 当某个并发档表现异常时，再用 `requests.jsonl` 查看单请求延迟和响应摘要。

## 瓶颈判断规则

- 有价值的瓶颈信号必须包含成功请求，以及可观测的退化模式：吞吐下降、延迟分位数升高、错误率增加，或随并发升高出现阈值违反。
- 100% request failure 的并发档不是瓶颈。应把 100% request failure 视为接口、鉴权、服务可用性、数据集或请求结构问题。
- 如果低并发就开始失败，先运行 manual-interface-performance-testing skill，不要继续加压。
- 只比较模型、数据集、请求体、目标服务配置、硬件、duration 或 epochs、并发计划完全一致的运行结果。
- 不要外推到未测试条件。报告中应包含精确命令、config、dataset、thresholds、output directory 和运行时间。

## 安全规则

- 不要把鉴权、授权、路由、bad request、文件缺失或 100% request failure 当作性能瓶颈。
- 先修接口和凭据；不要针对错误请求调并发。
- 不要假设本包会管理目标服务生命周期。服务启动、停止、预热和扩缩容应由用户或外部自动化负责。
- 对共享、付费、生产或第三方服务运行高并发压测前，必须获得明确许可。
