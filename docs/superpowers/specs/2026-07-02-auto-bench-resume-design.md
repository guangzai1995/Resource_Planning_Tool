# Auto Bench Stop/Resume 设计

## Status

Ready for user review.

## Context

`vllm_standalone_bench/run_auto_bench.sh` 是薄包装脚本，实际控制逻辑在
`vllm_standalone_bench/auto_bench.py`。当前后台任务控制已经支持：

- `status`：读取 `results/<run_id>/state.json` 和 `manifest.json`。
- `logs`：展示当前 bench log 或 controller log。
- `stop`：读取 `controller.pid` 和 `controller.json`，确认进程确实是本次
  `auto_bench.py run --child --run-id <run_id>` controller 后发送 `SIGTERM`。

当前 `stop` 语义不是暂停容器，而是请求 controller 优雅中止。controller 捕获
`SIGTERM` 后会走 `StopRequested` 路径，将 run 标记为 `interrupted`，并清理当前
bench 容器、vLLM 容器和本次拥有的 Docker network。因此 stop 后测试资源会被释放，
但没有可直接继续的容器状态。

现有状态文件已经提供恢复基础：

- `manifest.json` 记录每个 case 的 `model` / `serve_profile` / `bench_profile` /
  `status` 和结果文件路径。
- `state.json` 记录 run 总状态、当前 case 和 counts。
- `config.resolved.json` 记录当次运行展开后的配置输入。
- `.run.lock` 和 `controller.pid` 防止同一 run 并发启动。

缺口是：没有 `resume` 命令，`run` 也不会按历史 manifest 跳过已经成功的 case。

## User-Approved Direction

恢复语义采用方案 A：同一个 `RUN_ID` 下跳过已 `passed` 的 case，重跑
`interrupted`、`failed`、`skipped` 和未记录的 case。恢复粒度是
`model / serve_profile / bench_profile` case，不做单个 bench 请求内部断点续跑。

## Goals

- 新增清晰的 `resume` 子命令：
  `RUN_ID=<id> ./vllm_standalone_bench/run_auto_bench.sh resume`。
- 恢复时保留已经 `passed` 的 case 结果，不重复消耗 GPU 时间。
- 恢复时补跑未成功 case，包括中断时被标记为 `interrupted` 的 case。
- 复用现有安全边界：run lock、active run 检查、controller metadata 校验和 Docker
  资源 ownership label。
- 恢复完成后生成一致的 `manifest.json`、`state.json`、`compare.csv`、
  `compare.xlsx` 和 plots。
- 让 `status/logs/stop/resume` 的 shell wrapper 使用方式保持一致。

## Non-Goals

- 不实现容器级暂停/恢复。stop 后容器会被清理，resume 会重新启动所需 vLLM 容器。
- 不实现 `run_bench_multi.py` 内部的请求级断点续跑。正在运行的 bench case 被中断后，
  该 case 下的 `result.csv` / `result.xlsx` 视为不可信，恢复时整 case 重跑。
- 不让普通 `run` 自动恢复旧 run。恢复必须通过显式 `resume` 触发，避免误覆盖。
- 不改变现有 `stop` 的安全检查和 `SIGTERM` 优雅中止模型。

## CLI Design

### Shell Wrapper

`run_auto_bench.sh` 新增 `resume` 子命令：

```bash
RUN_ID=qwen_smoke_001 ./vllm_standalone_bench/run_auto_bench.sh resume
```

转发为：

```bash
python3 vllm_standalone_bench/auto_bench.py resume \
  --results-dir "${RESULTS_DIR}" \
  --run-id "${RUN_ID}" \
  --detach
```

`resume` 与 `run` 一样尊重 `DETACH`：默认 `DETACH=true` 后台恢复；设置
`DETACH=false` 时前台恢复。`resume` 与 `status/logs/stop` 一样要求显式
`RUN_ID`。`FOLLOW` 只影响 `logs`，不影响 `resume`。

### Python CLI

`auto_bench.py` 新增 parser：

```bash
python3 vllm_standalone_bench/auto_bench.py resume \
  --results-dir vllm_standalone_bench/results \
  --run-id <run_id> \
  [--detach]
```

`--detach` 行为与 `run --detach` 对齐：父进程只启动后台 controller 并打印 run id、
controller log 和 logs 命令；子进程执行实际恢复流程。无论前台还是后台，恢复执行时都
必须写新的 `controller.pid` / `controller.json`，让 `status/logs/stop` 行为一致。

## Resume Eligibility

恢复前检查 `results_dir / run_id`：

1. `state.json` 必须存在且为 JSON object。
2. 当前状态不能是 `starting` 或 `running`。如果 active run 存在，拒绝恢复。
3. `manifest.json` 必须存在且为 JSON object，`cases` 必须是 list。
4. `config.resolved.json` 必须存在并可加载。
5. `run_id` 必须与目录名和 manifest 中的 `run_id` 一致。

第一版支持从这些终态恢复：

- `interrupted`
- `failed`
- `completed_with_failures`

对 `completed` 默认拒绝恢复，并提示没有需要恢复的 case。这样避免用户误覆盖已完成结果。

## Config Source

恢复优先使用 `results/<run_id>/config.resolved.json`，而不是当前磁盘上的原始 config 文件。
原因是原始 config 可能已经被修改；resume 应该复现被中止那次 run 的矩阵和 Docker 参数。

实现细节：

- 继续复用 `load_config()` 解析 resolved config。当前 resolved config 中额外的
  `host_model_path`、`host_tokenizer_path` 字段会被 parser 忽略。
- 命令行 `--results-dir` 是恢复时的权威结果目录。加载 config 后要用
  `dataclasses.replace(config.run, results_dir=args.results_dir)` 生成恢复用 config，避免
  `config.resolved.json` 中相对 `results_dir` 被不同工作目录错误解释。
- `run_id` 使用命令行传入值，不从 config 重新生成。

## Case Selection

恢复先按 resolved config 调用 `expand_cases(config, run_id=run_id)` 得到完整 case 列表。
然后读取 manifest，按 `(model, serve_profile, bench_profile)` 建立历史状态索引。

分类规则：

- `passed`：跳过，保留原 manifest row 和结果文件。
- `failed` / `skipped` / `interrupted`：加入待重跑集合。
- manifest 中不存在：加入待重跑集合。
- manifest 中出现不属于当前 resolved config 的 case：保留在诊断信息中，但不参与恢复；
  同时在 controller log 中写 warning，说明 resolved config 与 manifest 不一致。

为了避免重复 manifest row，恢复运行应创建新的 in-memory `Manifest`：

1. 先把历史 `passed` rows 按原顺序拷贝进去。
2. 跑待恢复 case 时追加新 row。
3. 终态写回 `manifest.json`，不保留旧的非 passed row。

这样最终 manifest 表示“本次 run 当前可信结果”：已成功的旧结果 + 恢复后新结果。

## Execution Flow

恢复执行复用现有 `run_controller` 的主体逻辑，但需要支持 skip set：

```text
load resolved config
validate run is resumable
expand full cases
passed_keys = manifest cases where status == passed
pending_cases = all cases whose key not in passed_keys
run controller with:
  initial manifest = old passed rows
  cases_to_run = pending_cases
  total = full case count
```

serve group 行为：

- 如果某个 serve group 下所有 case 都已 `passed`，不启动 vLLM 容器。
- 如果 serve group 下至少一个 case 待重跑，启动一次 vLLM 容器，只运行该 group 的待重跑
  bench case。
- 中断恢复过程时，只把本次恢复中尚未完成的 pending case 标记为 `interrupted`；历史
  `passed` case 保留。

聚合行为：

- 恢复结束且没有再次中断时，调用现有 `aggregate_compare(config, run_dir)`。
- 聚合基于最终 manifest 和目录内结果文件，跳过的 passed case 仍参与 compare 输出。

## Artifact Policy

- 已 `passed` 的 case 目录不改动。
- 待重跑 case 开始前，其 `bench.log` 会按现有逻辑以写模式打开并覆盖。
- 待重跑 case 的 `status.json`、`result.csv`、`result.xlsx` 由新 run 覆盖。
- vLLM serve 目录下的 `vllm.log` / `docker.inspect.json` / `serve_command.txt` 可能被同一
  serve profile 的恢复过程覆盖。这是可接受的，因为它们描述的是最近一次启动该 serve
  profile 的容器状态。
- `controller.log` 继续追加，便于审计 stop 和 resume 的连续历史。

## Error Handling

- active run 存在时，`resume` 返回 1，不发送信号、不清理资源。
- `config.resolved.json` 缺失或无法解析时，`resume` 返回 1，并提示用户只能使用原始
  `run --config ... --run-id ...` 重新跑。
- manifest 损坏时，`resume` 返回 1，不猜测已完成 case。
- pending case 为空时，`resume` 返回 0，打印没有需要恢复的 case，不改写 manifest。
- Docker 资源冲突继续沿用现有 ownership label 规则：只删除本 run 管理且 label 匹配的容器。
- 恢复过程中再次 `stop` 时，状态写为 `interrupted`，后续可再次 resume。

## Tests

单元测试聚焦 `vllm_standalone_bench/tests/test_auto_bench.py`：

- CLI parser 接受 `resume --results-dir --run-id`，wrapper 允许 `resume`。
- `resume` 拒绝 `starting` / `running` active run。
- `resume` 从 `interrupted` manifest 中跳过 `passed` case，只运行 pending case。
- `resume` 对 `failed` / `skipped` / 未记录 case 进行重跑。
- `resume` 完成后 manifest 只包含历史 passed row 和新结果 row，不重复旧 interrupted row。
- 某 serve group 全部 passed 时，不启动该 group 的 vLLM 容器。
- 某 serve group 部分 pending 时，只执行 pending bench case。
- pending 为空时返回 0，并保持结果文件不变。
- `config.resolved.json` 的 `results_dir` 被 CLI `--results-dir` 覆盖。
- 恢复过程收到 `StopRequested` 时历史 passed row 保留，pending row 标记为 interrupted。

验证命令：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
bash -n vllm_standalone_bench/run_auto_bench.sh
git diff --check
```

## Acceptance Criteria

- 用户可以先执行 `RUN_ID=<id> ./run_auto_bench.sh stop` 中止后台 run。
- 中止后容器和本 run 拥有的 network 仍按现有逻辑清理。
- 用户随后执行 `RUN_ID=<id> ./run_auto_bench.sh resume`，只补跑未成功 case。
- 已经 `passed` 的 case 不重新启动 bench-runner，不覆盖其 `result.csv` / `result.xlsx`。
- 恢复完成后 `status` 显示最终状态；全部 case 成功时为 `completed`。
- 同一个 run 可以多次 stop/resume，直到所有 case 成功或用户选择不再恢复。
