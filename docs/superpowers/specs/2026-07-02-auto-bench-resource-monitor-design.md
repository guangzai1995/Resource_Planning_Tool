# Auto Bench Resource Monitor 设计

## Status

Ready for user review.

## Context

`vllm_standalone_bench/auto_bench.py` 负责离线自动化压测的控制流程：
启动 vLLM/SGLang 服务容器、等待 ready、按 bench profile 启动 bench-runner 容器、
写入 `manifest.json` / `state.json`，并把每个 case 的结果放到：

```text
<run_id>/<model>/<serve_profile>/<bench_profile>/
  result.csv
  result.xlsx
  bench.log
```

当前 `run_bench_multi.py` 已经会从服务端 `/metrics` 采集部分模型服务内部指标，例如
GPU KV cache usage、prefix cache 命中率和 speculative decoding 指标。这些指标说明的是
推理服务内部状态，但不能回答压测期间整机资源水位问题，例如宿主机 CPU、内存、网络 IO、
磁盘 IO、GPU 利用率、显存和功耗。

本设计为 `auto_bench` 增加宿主机侧轻量资源监控。采样生命周期绑定到单个 benchmark
case，保证结果可以和同一 case 的吞吐、TTFT、TPOT、E2EL 对齐分析。

## User-Approved Direction

采用宿主机侧采样方案。首版采集宿主机全局 CPU、内存、网络 IO、磁盘 IO，并通过
`nvidia-smi` 采集 NVIDIA GPU 指标。资源监控的目标是查看压测过程中的资源使用情况：

- 结果表中追加核心汇总字段，便于报告和横向比较。
- 每个 case 保存时间序列文件，便于排查资源尖峰。
- 首版不做资源阈值门禁，不因为资源采样失败改变 benchmark 成败。

## Goals

- 在每个 benchmark case 运行期间采集宿主机全局资源指标。
- 使用 Linux `/proc` 采集 CPU、内存、网络 IO 和磁盘 IO，不引入新的 Python 运行时依赖。
- 使用宿主机 `nvidia-smi` 采集 NVIDIA GPU 利用率、显存、功耗和温度。
- 为每个 case 写入 `resource_samples.csv` 时间序列和 `resource_summary.json` 汇总。
- 将核心资源汇总字段追加合并到该 case 的 `result.csv` 和 `result.xlsx`。
- 资源监控缺失或失败时，benchmark 继续执行，并在资源汇总中记录不可用原因。
- 默认启用资源监控，采样间隔默认 1 秒。

## Non-Goals

- 不实现阈值熔断或自动停止压测。
- 不采集容器级 CPU/内存/网络/磁盘指标。首版只采宿主机全局指标。
- 不适配 Ascend NPU、AMD GPU 或其他硬件后端。
- 不接入 Prometheus、Grafana 或外部监控系统。
- 不把单卡明细全部追加到 `result.csv`，避免结果表过宽。
- 不修改 `run_bench_multi.py` 的请求压测逻辑和服务端 `/metrics` 采集逻辑。

## Configuration

在 `run` 配置下新增可选段落：

```json
{
  "run": {
    "resource_monitor": {
      "enabled": true,
      "backend": "nvidia-smi",
      "interval_sec": 1.0
    }
  }
}
```

字段语义：

- `enabled`：是否启用资源监控，默认 `true`。
- `backend`：GPU 采集后端，首版只允许 `"nvidia-smi"`。
- `interval_sec`：采样间隔秒数，默认 `1.0`，必须为正数。

向后兼容规则：

- 旧配置不写 `resource_monitor` 时等价于启用默认资源监控。
- `enabled=false` 时不启动采样器，不生成资源文件，也不追加资源列。
- `nvidia-smi` 不存在、机器无 NVIDIA GPU 或命令失败时，系统资源仍正常采集；
  GPU 字段为空，`resource_summary.json` 记录 GPU backend 不可用。

## Architecture

新增独立模块：

```text
vllm_standalone_bench/resource_monitor.py
```

主要职责：

- 读取 `/proc/stat` 计算 CPU 使用率。
- 读取 `/proc/meminfo` 计算内存使用量和使用率。
- 读取 `/proc/net/dev` 计算全机网络收发速率。
- 读取 `/proc/diskstats` 计算全机磁盘读写速率。
- 调用 `nvidia-smi` 并解析 GPU CSV 输出。
- 按固定间隔采样并缓存样本。
- 停止时写入 `resource_samples.csv` 和 `resource_summary.json`。
- 提供可被 `auto_bench.py` 调用的简单接口。

接口形态：

```python
monitor = ResourceMonitor(
    output_dir=layout.bench_dir,
    interval_sec=config.run.resource_monitor.interval_sec,
    enabled=config.run.resource_monitor.enabled,
    backend=config.run.resource_monitor.backend,
)
monitor.start()
try:
    result = active_runner.run(bench_cmd, ...)
finally:
    summary = monitor.stop()
```

`auto_bench.py` 只负责配置解析、生命周期绑定和结果合并。采样、解析、汇总和文件写入都放在
`resource_monitor.py`，避免继续扩大 `auto_bench.py` 的职责。

## Execution Flow

每个 benchmark case 的执行流程调整为：

```text
vLLM/SGLang serve container ready
create bench case output dir
start ResourceMonitor
run bench-runner container
stop ResourceMonitor in finally
write resource_samples.csv
write resource_summary.json
if result.csv exists:
  append aggregate resource columns to result.csv
if result.xlsx exists and openpyxl is available:
  append aggregate resource columns to result.xlsx
record manifest row
```

采样器必须在 `finally` 中停止，因此 bench 容器失败、用户 stop、`KeyboardInterrupt` 或
`StopRequested` 都会尽量写出已有资源样本。

## Sampling Design

### CPU

读取 `/proc/stat` 第一行 `cpu` 的累计 jiffies。相邻两次采样计算：

```text
cpu_util_pct = 100 * (delta_total - delta_idle) / delta_total
```

第一次采样没有前序点，CPU 使用率可以为空或从第二次采样开始记录。

### Memory

读取 `/proc/meminfo`：

- `MemTotal`
- `MemAvailable`

计算：

```text
mem_total_mb = MemTotal / 1024
mem_available_mb = MemAvailable / 1024
mem_used_mb = mem_total_mb - mem_available_mb
mem_used_pct = 100 * mem_used_mb / mem_total_mb
```

### Network IO

读取 `/proc/net/dev`，聚合除 loopback 之外所有网卡的 `rx_bytes` 和 `tx_bytes`。相邻两次采样按
时间差计算：

```text
net_rx_mb_s = delta_rx_bytes / elapsed_s / 1024 / 1024
net_tx_mb_s = delta_tx_bytes / elapsed_s / 1024 / 1024
```

首版不做网卡白名单配置。

### Disk IO

读取 `/proc/diskstats`，聚合真实块设备的扇区读写量。相邻两次采样按 Linux 常用 512-byte sector
计算：

```text
disk_read_mb_s = delta_read_sectors * 512 / elapsed_s / 1024 / 1024
disk_write_mb_s = delta_write_sectors * 512 / elapsed_s / 1024 / 1024
```

首版跳过明显的虚拟设备和分区噪声，例如 `loop*`、`ram*`。是否聚合分区由解析逻辑保持保守：
优先聚合主块设备，避免同一 IO 在磁盘和分区上重复统计。

### NVIDIA GPU

使用宿主机命令：

```bash
nvidia-smi \
  --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits
```

解析字段：

- `gpu_index`
- `gpu_name`
- `gpu_uuid`
- `gpu_util_pct`
- `gpu_mem_used_mb`
- `gpu_mem_total_mb`
- `gpu_mem_used_pct`
- `gpu_power_w`
- `gpu_temperature_c`

`nvidia-smi` 输出 `N/A`、`[Not Supported]` 或空值时，该字段记录为空，不让单字段解析失败导致
整次采样失败。

## Output Files

每个 case 目录新增：

```text
<run_id>/<model>/<serve_profile>/<bench_profile>/
  resource_samples.csv
  resource_summary.json
```

### resource_samples.csv

时间序列 CSV 一行表示一个采样时间点，便于 Excel 直接查看趋势。系统资源和 GPU 聚合字段放在同一行。
单卡明细不展开到 CSV，避免多卡机器产生过宽表格。

建议字段：

```text
timestamp
elapsed_s
cpu_util_pct
mem_total_mb
mem_used_mb
mem_available_mb
mem_used_pct
net_rx_mb_s
net_tx_mb_s
disk_read_mb_s
disk_write_mb_s
gpu_count
gpu_util_avg_pct
gpu_util_max_pct
gpu_mem_used_avg_mb
gpu_mem_used_max_mb
gpu_mem_total_mb
gpu_mem_used_max_pct
gpu_power_avg_w
gpu_power_max_w
gpu_temp_max_c
```

### resource_summary.json

汇总 JSON 保留全机聚合和单卡明细：

```json
{
  "available": true,
  "system_available": true,
  "gpu_available": true,
  "backend": "nvidia-smi",
  "interval_sec": 1.0,
  "sample_count": 120,
  "error_count": 0,
  "aggregate": {
    "cpu_util_avg_pct": 38.2,
    "cpu_util_p95_pct": 82.4,
    "cpu_util_max_pct": 91.0,
    "mem_used_avg_mb": 182400.5,
    "mem_used_p95_mb": 190100.0,
    "mem_used_max_mb": 191200.0,
    "mem_used_max_pct": 72.8,
    "net_rx_avg_mb_s": 12.1,
    "net_rx_max_mb_s": 108.5,
    "net_tx_avg_mb_s": 9.7,
    "net_tx_max_mb_s": 95.2,
    "disk_read_avg_mb_s": 33.0,
    "disk_read_max_mb_s": 420.0,
    "disk_write_avg_mb_s": 5.1,
    "disk_write_max_mb_s": 80.0,
    "gpu_count": 8,
    "gpu_util_avg_pct": 71.2,
    "gpu_util_p95_pct": 96.0,
    "gpu_util_max_pct": 99.0,
    "gpu_mem_used_avg_mb": 52110.5,
    "gpu_mem_used_p95_mb": 76000.0,
    "gpu_mem_used_max_mb": 77320.0,
    "gpu_mem_total_mb": 81559.0,
    "gpu_mem_used_max_pct": 94.8,
    "gpu_power_avg_w": 418.6,
    "gpu_power_p95_w": 580.0,
    "gpu_power_max_w": 612.0,
    "gpu_temp_max_c": 76.0
  },
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA H200",
      "uuid": "GPU-...",
      "gpu_util_avg_pct": 70.1,
      "gpu_util_p95_pct": 96.0,
      "gpu_util_max_pct": 99.0,
      "gpu_mem_used_avg_mb": 52000.0,
      "gpu_mem_used_p95_mb": 76000.0,
      "gpu_mem_used_max_mb": 77320.0,
      "gpu_mem_total_mb": 81559.0,
      "gpu_power_avg_w": 410.0,
      "gpu_power_p95_w": 575.0,
      "gpu_power_max_w": 600.0,
      "gpu_temp_max_c": 74.0
    }
  ]
}
```

`available` 表示是否至少有一种资源采样成功。`system_available` 和 `gpu_available` 分别表示系统
资源和 GPU 资源是否有有效样本。

## Result Table Columns

`result.csv` 和 `result.xlsx` 只追加全机汇总字段：

```text
resource_monitor_available
resource_sample_count
cpu_util_avg_pct
cpu_util_p95_pct
cpu_util_max_pct
mem_used_avg_mb
mem_used_p95_mb
mem_used_max_mb
mem_used_max_pct
net_rx_avg_mb_s
net_rx_max_mb_s
net_tx_avg_mb_s
net_tx_max_mb_s
disk_read_avg_mb_s
disk_read_max_mb_s
disk_write_avg_mb_s
disk_write_max_mb_s
gpu_count
gpu_util_avg_pct
gpu_util_p95_pct
gpu_util_max_pct
gpu_mem_used_avg_mb
gpu_mem_used_p95_mb
gpu_mem_used_max_mb
gpu_mem_total_mb
gpu_mem_used_max_pct
gpu_power_avg_w
gpu_power_p95_w
gpu_power_max_w
gpu_temp_max_c
```

合并规则：

- 如果 `result.csv` 有多行，每行追加相同的 case 级资源汇总。当前一个 bench profile 可能包含多个
  input/output/concurrency 组合，但资源监控生命周期绑定整个 bench profile case，因此首版资源汇总是
  case 级而不是单个矩阵行级。
- CSV 使用现有 UTF-8-BOM 写法，保留原始列顺序，在末尾追加资源列。
- XLSX 保留前两行英文/中文表头，在末尾追加英文列名和中文列名。
- 资源监控不可用时，`resource_monitor_available=false`，数值列留空。

## Error Handling

- `enabled=false`：不启动采样器，不生成资源文件，不追加资源列。
- `/proc` 文件读取失败：记录系统资源错误，继续尝试 GPU 采样。
- `nvidia-smi` 不存在：记录 `gpu_available=false` 和错误信息，继续系统资源采样。
- `nvidia-smi` 单次执行失败：增加错误计数，继续下一轮采样。
- 单个字段无法解析：该字段留空，不丢弃整条样本。
- 全程无有效样本：写 `resource_summary.json`，其中 `available=false`，benchmark 状态不受影响。
- bench 容器异常退出：仍在 `finally` 中停止采样并写出已有资源文件。
- 资源文件写入失败：记录 warning，不把原本成功的 benchmark 改为 failed。

## Tests

新增或扩展测试集中在 `vllm_standalone_bench/tests`：

- 配置解析：旧配置默认启用 resource monitor；显式 `enabled=false` 可禁用；非法 backend 或
  非正 `interval_sec` 报 `ConfigError`。
- `/proc/stat` CPU 解析：根据两次累计值计算正确利用率。
- `/proc/meminfo` 解析：正确计算 total、available、used 和 used percent。
- `/proc/net/dev` 解析：跳过 loopback，按两次采样计算 MB/s。
- `/proc/diskstats` 解析：跳过虚拟设备，按 sector 差值计算 MB/s。
- `nvidia-smi` CSV 解析：支持单卡、多卡、`N/A`、`[Not Supported]` 和空字段。
- 汇总计算：avg、p95、max 对空值和多样本处理正确。
- 采样器降级：`nvidia-smi` 不存在时系统资源可用，benchmark 不失败。
- `auto_bench` 集成：bench 成功时写资源文件并追加结果列。
- `auto_bench` 集成：bench 失败或中断时仍调用 `monitor.stop()`。
- 结果合并：CSV/XLSX 原有列保留，新资源列追加到末尾。

验证命令：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
git diff --check
```

## Acceptance Criteria

- 每个启用资源监控的 benchmark case 都生成 `resource_samples.csv` 和
  `resource_summary.json`。
- `resource_samples.csv` 可以直接用 Excel 查看随时间变化的 CPU、内存、网络、磁盘和 GPU 聚合资源。
- `resource_summary.json` 包含全机聚合和单卡 GPU 明细。
- `result.csv` 和 `result.xlsx` 末尾包含资源汇总字段。
- 没有 NVIDIA GPU 或 `nvidia-smi` 不可用时，系统资源仍采集，GPU 字段留空。
- 资源监控异常不会把 benchmark 成功结果改成失败。
- 默认配置无需修改即可获得资源监控结果。
