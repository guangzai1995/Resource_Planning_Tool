# Auto Bench Resource Monitor 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `auto_bench` 的每个 benchmark case 增加宿主机全局 CPU、内存、网络 IO、磁盘 IO 和 NVIDIA GPU 资源监控，并把时间序列与汇总指标落盘到 case 结果目录。

**架构：** 新增 `vllm_standalone_bench/resource_monitor.py` 承担采样、解析、汇总、文件写入和结果表合并；`auto_bench.py` 只新增配置解析、生命周期绑定和调用点。采样失败不影响 benchmark 成败，资源汇总以 case 级别追加到 `result.csv/result.xlsx`。

**技术栈：** Python 标准库、Linux `/proc`、宿主机 `nvidia-smi`、pytest、openpyxl（项目已有可选依赖）。

---

## 文件结构

- 创建：`vllm_standalone_bench/resource_monitor.py`
  - 定义 `ResourceReaders`、采样数据结构、`ResourceMonitor`、解析函数、汇总函数、CSV/JSON 写入函数、`append_summary_to_result_files()`。
  - 不依赖 Docker，不依赖 `auto_bench.py`，只使用标准库和可选 `openpyxl`。
- 创建：`vllm_standalone_bench/tests/test_resource_monitor.py`
  - 覆盖 `/proc` 解析、`nvidia-smi` 解析、汇总、降级、文件输出、CSV/XLSX 合并。
- 修改：`vllm_standalone_bench/auto_bench.py`
  - 新增 `ResourceMonitorRunConfig` dataclass。
  - 扩展 `RunConfig`、`_parse_run()`、`config_to_dict()`。
  - 在每个非 dry-run benchmark case 前后启动/停止资源监控，并合并结果表。
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`
  - 覆盖配置默认值、显式禁用、非法配置、run_controller 集成、bench 失败仍 stop monitor。
- 修改：`vllm_standalone_bench/configs/auto_bench.example.json`
  - 展示 `run.resource_monitor` 默认推荐配置。
- 修改：`vllm_standalone_bench/README.md`
  - 说明资源监控输出文件、字段范围和 `nvidia-smi` 降级行为。

---

### 任务 1：实现资源解析与汇总核心

**文件：**
- 创建：`vllm_standalone_bench/resource_monitor.py`
- 创建：`vllm_standalone_bench/tests/test_resource_monitor.py`

- [ ] **步骤 1：编写失败的解析与汇总测试**

在 `vllm_standalone_bench/tests/test_resource_monitor.py` 写入首批测试：

```python
import pytest

import resource_monitor as rm


def test_parse_cpu_stat_and_compute_utilization():
    first = rm.parse_proc_stat("cpu  100 0 50 850 0 0 0 0 0 0\n")
    second = rm.parse_proc_stat("cpu  180 0 70 950 0 0 0 0 0 0\n")

    assert rm.cpu_utilization_pct(first, second) == pytest.approx(50.0)


def test_parse_meminfo_uses_mem_available():
    sample = "\n".join([
        "MemTotal:       1024000 kB",
        "MemFree:         100000 kB",
        "MemAvailable:    256000 kB",
    ])

    memory = rm.parse_meminfo(sample)

    assert memory["mem_total_mb"] == pytest.approx(1000.0)
    assert memory["mem_available_mb"] == pytest.approx(250.0)
    assert memory["mem_used_mb"] == pytest.approx(750.0)
    assert memory["mem_used_pct"] == pytest.approx(75.0)


def test_parse_net_dev_skips_loopback_and_computes_rates():
    before = """
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000 0 0 0 0 0 0 0 2000 0 0 0 0 0 0 0
  eth0: 1048576 0 0 0 0 0 0 0 2097152 0 0 0 0 0 0 0
"""
    after = """
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 999999 0 0 0 0 0 0 0 999999 0 0 0 0 0 0 0
  eth0: 3145728 0 0 0 0 0 0 0 5242880 0 0 0 0 0 0 0
"""

    rates = rm.network_rates_mb_s(
        rm.parse_net_dev(before),
        rm.parse_net_dev(after),
        elapsed_s=2.0,
    )

    assert rates["net_rx_mb_s"] == pytest.approx(1.0)
    assert rates["net_tx_mb_s"] == pytest.approx(1.5)


def test_parse_nvidia_smi_csv_handles_multi_gpu_and_empty_values():
    text = "\n".join([
        "0, NVIDIA H200, GPU-0, 98, 77320, 81559, 612.50, 76",
        "1, NVIDIA H200, GPU-1, N/A, 12000, 81559, [Not Supported], ",
    ])

    gpus = rm.parse_nvidia_smi_csv(text)

    assert gpus[0]["gpu_index"] == 0
    assert gpus[0]["gpu_util_pct"] == 98.0
    assert gpus[0]["gpu_mem_used_pct"] == pytest.approx(94.8001, rel=0.001)
    assert gpus[1]["gpu_util_pct"] is None
    assert gpus[1]["gpu_power_w"] is None
    assert gpus[1]["gpu_temperature_c"] is None


def test_summarize_samples_computes_avg_p95_and_max():
    samples = [
        {"cpu_util_pct": 10.0, "mem_used_mb": 100.0, "gpu_util_avg_pct": 20.0},
        {"cpu_util_pct": 20.0, "mem_used_mb": 200.0, "gpu_util_avg_pct": 40.0},
        {"cpu_util_pct": 30.0, "mem_used_mb": 300.0, "gpu_util_avg_pct": 60.0},
    ]

    summary = rm.summarize_samples(samples, gpu_details=[])

    assert summary["available"] is True
    assert summary["sample_count"] == 3
    assert summary["aggregate"]["cpu_util_avg_pct"] == pytest.approx(20.0)
    assert summary["aggregate"]["cpu_util_p95_pct"] == pytest.approx(30.0)
    assert summary["aggregate"]["mem_used_max_mb"] == pytest.approx(300.0)
    assert summary["aggregate"]["gpu_util_max_pct"] == pytest.approx(60.0)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
```

预期：FAIL，导入 `resource_monitor` 或被测函数不存在。

- [ ] **步骤 3：实现最少核心代码**

创建 `vllm_standalone_bench/resource_monitor.py`，包含这些公开函数和常量：

```python
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _parse_float(value: str) -> float | None:
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[N/A]", "[Not Supported]", "Not Supported"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_proc_stat(text: str) -> dict[str, int]:
    for line in text.splitlines():
        if line.startswith("cpu "):
            values = [int(part) for part in line.split()[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return {"idle": idle, "total": sum(values)}
    raise ValueError("/proc/stat missing aggregate cpu line")


def cpu_utilization_pct(previous: Mapping[str, int], current: Mapping[str, int]) -> float | None:
    total_delta = int(current["total"]) - int(previous["total"])
    idle_delta = int(current["idle"]) - int(previous["idle"])
    if total_delta <= 0:
        return None
    return _round((total_delta - idle_delta) * 100.0 / total_delta)


def parse_meminfo(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        parts = rest.split()
        if parts:
            values[name] = float(parts[0])
    total_mb = values["MemTotal"] / 1024.0
    available_mb = values.get("MemAvailable", values.get("MemFree", 0.0)) / 1024.0
    used_mb = max(total_mb - available_mb, 0.0)
    used_pct = used_mb * 100.0 / total_mb if total_mb > 0 else 0.0
    return {
        "mem_total_mb": _round(total_mb),
        "mem_available_mb": _round(available_mb),
        "mem_used_mb": _round(used_mb),
        "mem_used_pct": _round(used_pct),
    }


def parse_net_dev(text: str) -> dict[str, int]:
    rx_total = 0
    tx_total = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        if iface.strip() == "lo":
            continue
        fields = rest.split()
        if len(fields) >= 16:
            rx_total += int(fields[0])
            tx_total += int(fields[8])
    return {"rx_bytes": rx_total, "tx_bytes": tx_total}


def network_rates_mb_s(previous: Mapping[str, int], current: Mapping[str, int], *, elapsed_s: float) -> dict[str, float | None]:
    if elapsed_s <= 0:
        return {"net_rx_mb_s": None, "net_tx_mb_s": None}
    rx = max(int(current["rx_bytes"]) - int(previous["rx_bytes"]), 0)
    tx = max(int(current["tx_bytes"]) - int(previous["tx_bytes"]), 0)
    return {
        "net_rx_mb_s": _round(rx / elapsed_s / 1024.0 / 1024.0),
        "net_tx_mb_s": _round(tx / elapsed_s / 1024.0 / 1024.0),
    }


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 8:
            continue
        mem_used = _parse_float(parts[4])
        mem_total = _parse_float(parts[5])
        mem_pct = (
            _round(mem_used * 100.0 / mem_total)
            if mem_used is not None and mem_total not in (None, 0)
            else None
        )
        rows.append({
            "gpu_index": int(parts[0]),
            "gpu_name": parts[1],
            "gpu_uuid": parts[2],
            "gpu_util_pct": _parse_float(parts[3]),
            "gpu_mem_used_mb": mem_used,
            "gpu_mem_total_mb": mem_total,
            "gpu_mem_used_pct": mem_pct,
            "gpu_power_w": _parse_float(parts[6]),
            "gpu_temperature_c": _parse_float(parts[7]),
        })
    return rows
```

在同一文件中实现 `summarize_samples(samples, gpu_details)`，至少支持测试中使用的字段：

```python
def _percentile(values: list[float], percentile: float) -> float | None:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    index = math.ceil(percentile / 100.0 * len(clean)) - 1
    return clean[max(0, min(index, len(clean) - 1))]


def _values(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(sample[key])
        for sample in samples
        if sample.get(key) is not None
    ]


def _add_stats(aggregate: dict[str, Any], *, avg_key: str, p95_key: str, max_key: str, values: list[float]) -> None:
    if not values:
        return
    aggregate[avg_key] = _round(sum(values) / len(values))
    aggregate[p95_key] = _round(_percentile(values, 95))
    aggregate[max_key] = _round(max(values))


def summarize_samples(samples: list[dict[str, Any]], *, gpu_details: list[dict[str, Any]], error_count: int = 0, interval_sec: float = 1.0, backend: str = "nvidia-smi") -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    _add_stats(aggregate, avg_key="cpu_util_avg_pct", p95_key="cpu_util_p95_pct", max_key="cpu_util_max_pct", values=_values(samples, "cpu_util_pct"))
    _add_stats(aggregate, avg_key="mem_used_avg_mb", p95_key="mem_used_p95_mb", max_key="mem_used_max_mb", values=_values(samples, "mem_used_mb"))
    _add_stats(aggregate, avg_key="mem_used_avg_pct", p95_key="mem_used_p95_pct", max_key="mem_used_max_pct", values=_values(samples, "mem_used_pct"))
    _add_stats(aggregate, avg_key="net_rx_avg_mb_s", p95_key="net_rx_p95_mb_s", max_key="net_rx_max_mb_s", values=_values(samples, "net_rx_mb_s"))
    _add_stats(aggregate, avg_key="net_tx_avg_mb_s", p95_key="net_tx_p95_mb_s", max_key="net_tx_max_mb_s", values=_values(samples, "net_tx_mb_s"))
    _add_stats(aggregate, avg_key="disk_read_avg_mb_s", p95_key="disk_read_p95_mb_s", max_key="disk_read_max_mb_s", values=_values(samples, "disk_read_mb_s"))
    _add_stats(aggregate, avg_key="disk_write_avg_mb_s", p95_key="disk_write_p95_mb_s", max_key="disk_write_max_mb_s", values=_values(samples, "disk_write_mb_s"))
    _add_stats(aggregate, avg_key="gpu_util_avg_pct", p95_key="gpu_util_p95_pct", max_key="gpu_util_max_pct", values=_values(samples, "gpu_util_avg_pct"))
    _add_stats(aggregate, avg_key="gpu_mem_used_avg_mb", p95_key="gpu_mem_used_p95_mb", max_key="gpu_mem_used_max_mb", values=_values(samples, "gpu_mem_used_avg_mb"))
    _add_stats(aggregate, avg_key="gpu_power_avg_w", p95_key="gpu_power_p95_w", max_key="gpu_power_max_w", values=_values(samples, "gpu_power_avg_w"))
    _add_stats(aggregate, avg_key="gpu_temp_avg_c", p95_key="gpu_temp_p95_c", max_key="gpu_temp_max_c", values=_values(samples, "gpu_temp_max_c"))
    if samples:
        aggregate["gpu_count"] = max((int(sample.get("gpu_count") or 0) for sample in samples), default=0)
        mem_total_values = _values(samples, "gpu_mem_total_mb")
        if mem_total_values:
            aggregate["gpu_mem_total_mb"] = _round(max(mem_total_values))
        max_pct_values = _values(samples, "gpu_mem_used_max_pct")
        if max_pct_values:
            aggregate["gpu_mem_used_max_pct"] = _round(max(max_pct_values))
    return {
        "available": bool(samples),
        "system_available": any(sample.get("cpu_util_pct") is not None or sample.get("mem_used_mb") is not None for sample in samples),
        "gpu_available": any((sample.get("gpu_count") or 0) > 0 for sample in samples),
        "backend": backend,
        "interval_sec": interval_sec,
        "sample_count": len(samples),
        "error_count": error_count,
        "aggregate": aggregate,
        "gpus": gpu_details,
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
```

预期：PASS，5 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/resource_monitor.py vllm_standalone_bench/tests/test_resource_monitor.py
git commit -m "feat(bench): add resource monitor parsing core"
```

---

### 任务 2：实现采样器、文件输出和结果表合并

**文件：**
- 修改：`vllm_standalone_bench/resource_monitor.py`
- 修改：`vllm_standalone_bench/tests/test_resource_monitor.py`

- [ ] **步骤 1：编写失败的采样、落盘和合并测试**

追加这些测试到 `vllm_standalone_bench/tests/test_resource_monitor.py`：

```python
import csv
import json
from pathlib import Path


def test_resource_monitor_writes_samples_and_summary(tmp_path):
    monitor = rm.ResourceMonitor(
        output_dir=tmp_path,
        interval_sec=1.0,
        enabled=True,
        backend="nvidia-smi",
        readers=rm.ResourceReaders(
            proc_stat=lambda: "cpu  100 0 50 850 0 0 0 0 0 0\n",
            meminfo=lambda: "MemTotal: 1024000 kB\nMemAvailable: 512000 kB\n",
            net_dev=lambda: "",
            diskstats=lambda: "",
            nvidia_smi=lambda: "0, NVIDIA H200, GPU-0, 50, 1000, 2000, 300, 70\n",
        ),
    )

    monitor.sample_once(now=100.0)
    summary = monitor.stop()

    assert (tmp_path / "resource_samples.csv").is_file()
    assert (tmp_path / "resource_summary.json").is_file()
    loaded = json.loads((tmp_path / "resource_summary.json").read_text(encoding="utf-8"))
    assert loaded["available"] is True
    assert loaded["gpu_available"] is True
    assert summary["aggregate"]["gpu_count"] == 1


def test_resource_monitor_degrades_when_nvidia_smi_missing(tmp_path):
    def missing_gpu():
        raise FileNotFoundError("nvidia-smi")

    monitor = rm.ResourceMonitor(
        output_dir=tmp_path,
        interval_sec=1.0,
        enabled=True,
        backend="nvidia-smi",
        readers=rm.ResourceReaders(
            proc_stat=lambda: "cpu  100 0 50 850 0 0 0 0 0 0\n",
            meminfo=lambda: "MemTotal: 1024000 kB\nMemAvailable: 512000 kB\n",
            net_dev=lambda: "",
            diskstats=lambda: "",
            nvidia_smi=missing_gpu,
        ),
    )

    monitor.sample_once(now=100.0)
    summary = monitor.stop()

    assert summary["available"] is True
    assert summary["system_available"] is True
    assert summary["gpu_available"] is False
    assert summary["error_count"] >= 1


def test_append_summary_to_result_csv_adds_resource_columns(tmp_path):
    result_csv = tmp_path / "result.csv"
    result_csv.write_text("model,throughput_tok_s\nm,12.5\n", encoding="utf-8-sig")
    summary = {
        "available": True,
        "sample_count": 2,
        "aggregate": {
            "cpu_util_avg_pct": 50.0,
            "gpu_count": 1,
            "gpu_util_max_pct": 99.0,
        },
    }

    rm.append_summary_to_result_files(tmp_path, summary)

    with result_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "m"
    assert rows[0]["resource_monitor_available"] == "true"
    assert rows[0]["resource_sample_count"] == "2"
    assert rows[0]["cpu_util_avg_pct"] == "50.0"
    assert rows[0]["gpu_util_max_pct"] == "99.0"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
```

预期：FAIL，`ResourceMonitor`、`ResourceReaders` 或 `append_summary_to_result_files` 不存在。

- [ ] **步骤 3：实现采样器和文件合并**

在 `resource_monitor.py` 增加：

```python
SAMPLE_HEADERS = [
    "timestamp", "elapsed_s",
    "cpu_util_pct",
    "mem_total_mb", "mem_used_mb", "mem_available_mb", "mem_used_pct",
    "net_rx_mb_s", "net_tx_mb_s",
    "disk_read_mb_s", "disk_write_mb_s",
    "gpu_count", "gpu_util_avg_pct", "gpu_util_max_pct",
    "gpu_mem_used_avg_mb", "gpu_mem_used_max_mb", "gpu_mem_total_mb",
    "gpu_mem_used_max_pct", "gpu_power_avg_w", "gpu_power_max_w",
    "gpu_temp_max_c",
]

RESOURCE_RESULT_COLUMNS = [
    "resource_monitor_available", "resource_sample_count",
    "cpu_util_avg_pct", "cpu_util_p95_pct", "cpu_util_max_pct",
    "mem_used_avg_mb", "mem_used_p95_mb", "mem_used_max_mb", "mem_used_max_pct",
    "net_rx_avg_mb_s", "net_rx_max_mb_s", "net_tx_avg_mb_s", "net_tx_max_mb_s",
    "disk_read_avg_mb_s", "disk_read_max_mb_s",
    "disk_write_avg_mb_s", "disk_write_max_mb_s",
    "gpu_count", "gpu_util_avg_pct", "gpu_util_p95_pct", "gpu_util_max_pct",
    "gpu_mem_used_avg_mb", "gpu_mem_used_p95_mb", "gpu_mem_used_max_mb",
    "gpu_mem_total_mb", "gpu_mem_used_max_pct",
    "gpu_power_avg_w", "gpu_power_p95_w", "gpu_power_max_w",
    "gpu_temp_max_c",
]


@dataclass(frozen=True)
class ResourceReaders:
    proc_stat: Callable[[], str]
    meminfo: Callable[[], str]
    net_dev: Callable[[], str]
    diskstats: Callable[[], str]
    nvidia_smi: Callable[[], str]


def default_readers() -> ResourceReaders:
    return ResourceReaders(
        proc_stat=lambda: Path("/proc/stat").read_text(encoding="utf-8"),
        meminfo=lambda: Path("/proc/meminfo").read_text(encoding="utf-8"),
        net_dev=lambda: Path("/proc/net/dev").read_text(encoding="utf-8"),
        diskstats=lambda: Path("/proc/diskstats").read_text(encoding="utf-8"),
        nvidia_smi=lambda: subprocess.run(
            NVIDIA_SMI_QUERY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
    )
```

实现 `ResourceMonitor`：

```python
class ResourceMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        interval_sec: float = 1.0,
        enabled: bool = True,
        backend: str = "nvidia-smi",
        readers: ResourceReaders | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.interval_sec = float(interval_sec)
        self.enabled = enabled
        self.backend = backend
        self.readers = readers or default_readers()
        self.samples: list[dict[str, Any]] = []
        self.gpu_samples: list[dict[str, Any]] = []
        self.error_count = 0
        self._started_at: float | None = None
        self._previous_cpu: dict[str, int] | None = None
        self._previous_net: dict[str, int] | None = None
        self._previous_disk: dict[str, int] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.enabled:
            return
        self.sample_once()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.sample_once()

    def sample_once(self, now: float | None = None) -> None:
        if not self.enabled:
            return
        timestamp = time.time() if now is None else now
        if self._started_at is None:
            self._started_at = timestamp
        sample: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "elapsed_s": _round(timestamp - self._started_at),
        }
        self._sample_system(sample, timestamp)
        self._sample_gpu(sample)
        self.samples.append(sample)

    def _sample_system(self, sample: dict[str, Any], timestamp: float) -> None:
        try:
            current_cpu = parse_proc_stat(self.readers.proc_stat())
            sample["cpu_util_pct"] = (
                cpu_utilization_pct(self._previous_cpu, current_cpu)
                if self._previous_cpu is not None else None
            )
            self._previous_cpu = current_cpu
        except Exception:
            self.error_count += 1
        try:
            sample.update(parse_meminfo(self.readers.meminfo()))
        except Exception:
            self.error_count += 1
        try:
            current_net = parse_net_dev(self.readers.net_dev())
            sample.update(
                network_rates_mb_s(self._previous_net, current_net, elapsed_s=self.interval_sec)
                if self._previous_net is not None
                else {"net_rx_mb_s": None, "net_tx_mb_s": None}
            )
            self._previous_net = current_net
        except Exception:
            self.error_count += 1
        try:
            current_disk = parse_diskstats(self.readers.diskstats())
            sample.update(
                disk_rates_mb_s(self._previous_disk, current_disk, elapsed_s=self.interval_sec)
                if self._previous_disk is not None
                else {"disk_read_mb_s": None, "disk_write_mb_s": None}
            )
            self._previous_disk = current_disk
        except Exception:
            self.error_count += 1

    def _sample_gpu(self, sample: dict[str, Any]) -> None:
        if self.backend != "nvidia-smi":
            return
        try:
            gpus = parse_nvidia_smi_csv(self.readers.nvidia_smi())
        except Exception:
            self.error_count += 1
            gpus = []
        self.gpu_samples.extend(gpus)
        apply_gpu_aggregate(sample, gpus)

    def stop(self) -> dict[str, Any]:
        if not self.enabled:
            return {"available": False, "sample_count": 0, "aggregate": {}}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec, 1.0) + 1.0)
        summary = summarize_samples(
            self.samples,
            gpu_details=summarize_gpus(self.gpu_samples),
            error_count=self.error_count,
            interval_sec=self.interval_sec,
            backend=self.backend,
        )
        write_samples_csv(self.output_dir / "resource_samples.csv", self.samples)
        write_summary_json(self.output_dir / "resource_summary.json", summary)
        return summary
```

补齐 `parse_diskstats()`、`disk_rates_mb_s()`、`apply_gpu_aggregate()`、`summarize_gpus()`、`write_samples_csv()`、`write_summary_json()`、`append_summary_to_result_files()`。`append_summary_to_result_files()` 至少实现 CSV 合并，并用 `try/except ImportError` 方式实现 XLSX 合并：

```python
def append_summary_to_result_files(output_dir: Path, summary: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    values = flatten_summary_for_result(summary)
    csv_path = output_dir / "result.csv"
    if csv_path.exists():
        append_summary_to_csv(csv_path, values)
    xlsx_path = output_dir / "result.xlsx"
    if xlsx_path.exists():
        append_summary_to_xlsx(xlsx_path, values)
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
```

预期：PASS，资源监控模块测试全部通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/resource_monitor.py vllm_standalone_bench/tests/test_resource_monitor.py
git commit -m "feat(bench): write resource monitor artifacts"
```

---

### 任务 3：把资源监控配置接入 auto_bench

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的配置测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 的配置测试区域追加：

```python
def test_resource_monitor_defaults_enabled(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))

    assert config.run.resource_monitor.enabled is True
    assert config.run.resource_monitor.backend == "nvidia-smi"
    assert config.run.resource_monitor.interval_sec == 1.0


def test_resource_monitor_can_be_disabled(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {"enabled": False}

    config = ab.load_config(write_config(tmp_path, data))

    assert config.run.resource_monitor.enabled is False
    assert config.run.resource_monitor.backend == "nvidia-smi"


def test_resource_monitor_rejects_unsupported_backend(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {
        "enabled": True,
        "backend": "dcmi",
        "interval_sec": 1.0,
    }

    with pytest.raises(ab.ConfigError, match="resource_monitor.backend"):
        ab.load_config(write_config(tmp_path, data))


def test_resource_monitor_interval_must_be_positive(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {
        "enabled": True,
        "backend": "nvidia-smi",
        "interval_sec": 0,
    }

    with pytest.raises(ab.ConfigError, match="resource_monitor.interval_sec"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py \
  -k "resource_monitor_defaults_enabled or resource_monitor_can_be_disabled or resource_monitor_rejects_unsupported_backend or resource_monitor_interval_must_be_positive" -q
```

预期：FAIL，`RunConfig` 没有 `resource_monitor` 字段或解析函数不存在。

- [ ] **步骤 3：实现配置 dataclass 和解析**

在 `auto_bench.py` 顶部导入附近加入：

```python
from resource_monitor import ResourceMonitor, append_summary_to_result_files
```

在 `VllmCacheConfig` 后新增：

```python
@dataclass(frozen=True)
class ResourceMonitorRunConfig:
    enabled: bool = True
    backend: str = "nvidia-smi"
    interval_sec: float = 1.0
```

在 `RunConfig` 中加入字段：

```python
resource_monitor: ResourceMonitorRunConfig = field(default_factory=ResourceMonitorRunConfig)
```

新增解析函数：

```python
def _parse_resource_monitor(run: dict[str, Any]) -> ResourceMonitorRunConfig:
    raw = run.get("resource_monitor")
    if raw is None:
        return ResourceMonitorRunConfig()
    data = _require_mapping(raw, "run.resource_monitor")
    enabled = _bool(data.get("enabled", True), "run.resource_monitor.enabled")
    backend = _string(data.get("backend", "nvidia-smi"), "run.resource_monitor.backend")
    if backend != "nvidia-smi":
        raise ConfigError("run.resource_monitor.backend only supports nvidia-smi")
    interval_sec = _finite_float(
        data.get("interval_sec", 1.0),
        "run.resource_monitor.interval_sec",
    )
    if interval_sec <= 0:
        raise ConfigError("run.resource_monitor.interval_sec must be > 0")
    return ResourceMonitorRunConfig(
        enabled=enabled,
        backend=backend,
        interval_sec=interval_sec,
    )
```

在 `_parse_run()` 返回 `RunConfig(...)` 时加入：

```python
resource_monitor=_parse_resource_monitor(run),
```

在 `config_to_dict()` 的 run 部分加入：

```python
"resource_monitor": {
    "enabled": config.run.resource_monitor.enabled,
    "backend": config.run.resource_monitor.backend,
    "interval_sec": config.run.resource_monitor.interval_sec,
},
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py \
  -k "resource_monitor_defaults_enabled or resource_monitor_can_be_disabled or resource_monitor_rejects_unsupported_backend or resource_monitor_interval_must_be_positive" -q
```

预期：PASS，4 个配置测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): parse resource monitor config"
```

---

### 任务 4：把采样生命周期接入 run_controller

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 run_controller 集成测试**

在 `test_auto_bench.py` 的 `run_controller` 测试区域追加：

```python
class FakeResourceMonitor:
    instances = []

    def __init__(self, *, output_dir, interval_sec, enabled, backend):
        self.output_dir = Path(output_dir)
        self.interval_sec = interval_sec
        self.enabled = enabled
        self.backend = backend
        self.started = False
        self.stopped = False
        FakeResourceMonitor.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        summary = {
            "available": True,
            "sample_count": 1,
            "aggregate": {"cpu_util_avg_pct": 12.5},
        }
        (self.output_dir / "resource_summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        return summary


def test_run_controller_starts_and_stops_resource_monitor(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    FakeResourceMonitor.instances = []
    summaries = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)
    monkeypatch.setattr(ab, "append_summary_to_result_files", lambda output_dir, summary: summaries.append((Path(output_dir), summary)))

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 0
    assert len(FakeResourceMonitor.instances) == 1
    monitor = FakeResourceMonitor.instances[0]
    assert monitor.started is True
    assert monitor.stopped is True
    assert monitor.interval_sec == 1.0
    assert monitor.backend == "nvidia-smi"
    assert summaries[0][1]["aggregate"]["cpu_util_avg_pct"] == 12.5


def test_run_controller_stops_resource_monitor_when_bench_fails(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    config = ab.load_config(write_config(tmp_path, data))
    runner = FakeRunner(failures={"docker run --rm": 7})
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)
    monkeypatch.setattr(ab, "append_summary_to_result_files", lambda *args, **kwargs: None)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    assert len(FakeResourceMonitor.instances) == 1
    assert FakeResourceMonitor.instances[0].stopped is True


def test_run_controller_does_not_start_resource_monitor_for_dry_run(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=True)

    assert result == 0
    assert FakeResourceMonitor.instances == []


def test_run_controller_does_not_start_resource_monitor_when_disabled(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {"enabled": False}
    config = ab.load_config(write_config(tmp_path, data))
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert result == 0
    assert FakeResourceMonitor.instances == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py \
  -k "resource_monitor and run_controller" -q
```

预期：FAIL，`run_controller` 没有创建 `ResourceMonitor`。

- [ ] **步骤 3：实现生命周期接入**

在 `run_controller()` 内每个 bench case 的非 dry-run 分支，围绕 `active_runner.run(bench_cmd, ...)` 加入：

```python
monitor = (
    ResourceMonitor(
        output_dir=layout.bench_dir,
        interval_sec=config.run.resource_monitor.interval_sec,
        enabled=True,
        backend=config.run.resource_monitor.backend,
    )
    if config.run.resource_monitor.enabled
    else None
)
resource_summary: dict[str, Any] | None = None
try:
    if monitor is not None:
        monitor.start()
    with (layout.bench_dir / "bench.log").open(
        "w",
        encoding="utf-8",
    ) as log:
        result = active_runner.run(
            bench_cmd,
            check=False,
            capture=False,
            stdout=log,
            stderr=log,
        )
finally:
    if monitor is not None:
        try:
            resource_summary = monitor.stop()
        except Exception as exc:
            logger.warning("resource monitor stop failed: %s", exc)
        if resource_summary is not None:
            try:
                append_summary_to_result_files(layout.bench_dir, resource_summary)
            except Exception as exc:
                logger.warning("resource monitor result merge failed: %s", exc)
```

保持原有 `bench_interrupted` 逻辑不变：如果 `active_runner.run()` 抛 `StopRequested` 或 `KeyboardInterrupt`，异常仍继续向外传播；`monitor.stop()` 和 bench 容器清理都要执行。

关键约束：

- dry-run 分支不创建 monitor。
- `enabled=false` 时不创建 monitor，不生成资源文件，不追加资源列。
- `append_summary_to_result_files()` 的异常只记 warning，不改变 `status`。
- monitor 的异常不能跳过 `cleanup_bench_container_if_owned()`。

- [ ] **步骤 4：运行集成测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py \
  -k "resource_monitor and run_controller" -q
```

预期：PASS。

- [ ] **步骤 5：运行关键 auto_bench 回归测试**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
```

预期：PASS，`test_auto_bench.py` 全部通过。

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): monitor resources during auto bench cases"
```

---

### 任务 5：文档、示例配置和全量验证

**文件：**
- 修改：`vllm_standalone_bench/configs/auto_bench.example.json`
- 修改：`vllm_standalone_bench/README.md`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`
- 修改：`docs/superpowers/plans/2026-07-02-auto-bench-resource-monitor.md`

- [x] **步骤 1：编写失败的文档/配置检查测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 追加：

```python
def test_example_config_includes_resource_monitor():
    path = CONFIG_DIR / "auto_bench.example.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["run"]["resource_monitor"] == {
        "enabled": True,
        "backend": "nvidia-smi",
        "interval_sec": 1.0,
    }
```

- [x] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_example_config_includes_resource_monitor -q
```

预期：FAIL，示例配置还没有 `run.resource_monitor`。

- [x] **步骤 3：更新示例配置**

在 `vllm_standalone_bench/configs/auto_bench.example.json` 的 `run` 对象中加入：

```json
"resource_monitor": {
  "enabled": true,
  "backend": "nvidia-smi",
  "interval_sec": 1.0
}
```

注意 JSON 逗号位置，保持文件可被 `json.loads()` 解析。

- [x] **步骤 4：更新 README**

在 `vllm_standalone_bench/README.md` 的离线自动化压测章节后加入：

```markdown
### 资源监控

`auto_bench.py` 默认在每个 benchmark case 期间采集宿主机全局资源：
CPU、内存、网络 IO、磁盘 IO，以及可用时的 NVIDIA GPU 指标。GPU 采集使用宿主机
`nvidia-smi`，不要求 bench-runner 镜像安装监控依赖。

每个 case 目录会新增：

```text
resource_samples.csv
resource_summary.json
```

`resource_samples.csv` 是按采样时间点记录的趋势数据；`resource_summary.json`
包含 avg、p95、max 汇总和单卡 GPU 明细。`result.csv` / `result.xlsx`
末尾会追加 case 级资源汇总列。没有 NVIDIA GPU 或 `nvidia-smi` 不可用时，
系统资源仍会采集，GPU 字段留空，benchmark 成败不受资源监控影响。

可在配置中显式调整：

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
```

- [x] **步骤 5：运行文档/配置测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_example_config_includes_resource_monitor -q
```

预期：PASS。

- [x] **步骤 6：运行全量验证**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
pytest -q
git diff --check
```

预期：

- `test_resource_monitor.py` 全部 PASS。
- `test_auto_bench.py` 全部 PASS。
- 仓库全量 `pytest -q` PASS，允许既有 warnings。
- `git diff --check` 无输出，退出码 0。

- [x] **步骤 7：Commit**

```bash
git add \
  vllm_standalone_bench/configs/auto_bench.example.json \
  vllm_standalone_bench/README.md \
  vllm_standalone_bench/tests/test_auto_bench.py \
  docs/superpowers/plans/2026-07-02-auto-bench-resource-monitor.md
git commit -m "docs(bench): document auto bench resource monitoring"
```

---

## 最终核对清单

- 规格文件 `docs/superpowers/specs/2026-07-02-auto-bench-resource-monitor-design.md` 的 Goals 均有对应任务。
- 系统资源采集使用 `/proc`，没有新增 Python 依赖。
- GPU 只支持 `nvidia-smi`，不可用时降级。
- 资源监控失败不改变 benchmark 成败。
- 每个 case 有 `resource_samples.csv` 和 `resource_summary.json`。
- `result.csv/result.xlsx` 追加全机汇总字段，不追加单卡明细。
- 任务 1-5 均遵循红灯测试、绿灯实现、验证、commit。
