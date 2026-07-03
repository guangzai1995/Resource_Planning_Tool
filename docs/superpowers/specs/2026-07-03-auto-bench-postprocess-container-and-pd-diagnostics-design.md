# Auto Bench 后处理容器化与 PD 诊断设计

## 背景

远程运行 `auto_bench.vllm_pd_p2p_remote.example.json` 时出现两类问题：

- 本地 baseline case 已通过，但资源监控合并重写 `result.csv` 时报 `Permission denied`。原因是 benchmark Docker 容器通过 bind mount 生成结果文件，宿主机控制器进程再写同一文件时可能遇到文件归属或权限不一致。
- PD topology case 失败在远程 `docker run -d` 阶段，状态只记录 `topology role failed to start: p1 (125)`，没有保存 Docker stderr/stdout，导致无法判断真实原因。

## 目标

1. 后处理在 bench-runner 容器里执行，宿主机不依赖 `openpyxl`、`matplotlib` 等 Python 包。
2. 后处理容器使用宿主机 UID/GID 写结果，避免生成或重写 root-owned 结果文件。
3. 远程 topology role 启动失败时保存 masked command、stdout、stderr 和 return code，方便定位 Docker 125 的真实原因。

## 非目标

- 不在本次改动里猜测修复具体 PD 参数。Docker 125 的根因必须由远程 Docker stderr 确认。
- 不替换 benchmark runner 镜像构建流程。
- 不要求宿主机安装后处理依赖。

## 设计

新增 `auto_bench.py postprocess` 子命令，在宿主机或容器内执行纯 Python 后处理逻辑，包括：

- 对每个 case 的资源监控 summary 合并到 `result.csv/result.xlsx`。
- 聚合 `compare.csv/compare.xlsx`。
- 生成 plots。

控制器 `run` 完成后不再直接在宿主机调用 `aggregate_compare`，而是启动 `run.bench_image` 容器执行：

```text
docker run --rm --user <uid>:<gid> -v <repo>:/workspace -w /workspace <bench_image> python /workspace/vllm_standalone_bench/auto_bench.py postprocess --config <config> --results-dir <results_dir> --run-id <run_id>
```

容器执行失败时只记录 warning，不改变 benchmark case 的通过/失败结果。

`run_auto_bench.sh postprocess` 默认调用 `auto_bench.py postprocess --container`，由宿主机
启动后处理容器；容器内部执行的 Python 命令不带 `--container`，避免递归启动。

benchmark runner 容器本身也使用 `--user <uid>:<gid>` 运行。否则
`result.csv/result.xlsx` 会先由 root-owned 容器进程创建，后处理容器即使使用宿主机
UID/GID 也无法重写已有文件。

远程 topology role 启动失败时，在对应 case 目录写入：

- `commands/<role>.txt`：已有 masked command。
- `logs/<role>.start.log`：新增，包含 return code、stdout、stderr。

同时 `status.json.error` 包含简短 stderr 摘要，便于 `status` 和 manifest 直接看到关键失败信息。

## 测试策略

- 单元测试验证 postprocess 容器命令包含 `--user uid:gid`、repo mount、config/results/run-id 参数。
- 控制器测试验证 run 完成后调用容器化后处理，而不是直接调用宿主机 `aggregate_compare`。
- 单元测试验证 `postprocess` 子命令会执行资源监控合并和 compare 聚合。
- 单元测试验证远程 role 启动失败会写 `logs/<role>.start.log`，且 `status.json.error` 包含 stderr 摘要。
