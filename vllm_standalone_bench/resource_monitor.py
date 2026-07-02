import csv
import math
from io import StringIO


NVIDIA_SMI_QUERY = (
    "index,name,uuid,utilization.gpu,memory.used,memory.total,"
    "power.draw,temperature.gpu"
)

_MB = 1024.0 * 1024.0


def parse_proc_stat(text):
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != "cpu":
            continue

        values = [int(value) for value in parts[1:]]
        idle = values[3] if len(values) > 3 else 0
        iowait = values[4] if len(values) > 4 else 0
        return {
            "total": sum(values),
            "idle": idle + iowait,
            "values": values,
        }

    raise ValueError("missing aggregate cpu line")


def cpu_utilization_pct(previous, current):
    total_delta = current["total"] - previous["total"]
    idle_delta = current["idle"] - previous["idle"]
    if total_delta <= 0:
        return 0.0

    busy_delta = max(0, total_delta - idle_delta)
    return busy_delta / total_delta * 100.0


def parse_meminfo(text):
    values_kb = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        if not parts:
            continue
        values_kb[key] = float(parts[0])

    total_mb = values_kb.get("MemTotal", 0.0) / 1024.0
    available_kb = values_kb.get("MemAvailable", values_kb.get("MemFree", 0.0))
    available_mb = available_kb / 1024.0
    used_mb = max(0.0, total_mb - available_mb)
    used_pct = used_mb / total_mb * 100.0 if total_mb else 0.0

    return {
        "mem_total_mb": total_mb,
        "mem_available_mb": available_mb,
        "mem_used_mb": used_mb,
        "mem_used_pct": used_pct,
    }


def parse_net_dev(text):
    rx_bytes = 0
    tx_bytes = 0
    interfaces = {}

    for line in text.splitlines():
        if ":" not in line:
            continue

        iface, raw_stats = line.split(":", 1)
        iface = iface.strip()
        if iface == "lo":
            continue

        fields = raw_stats.split()
        if len(fields) < 16:
            continue

        rx = int(fields[0])
        tx = int(fields[8])
        interfaces[iface] = {"rx_bytes": rx, "tx_bytes": tx}
        rx_bytes += rx
        tx_bytes += tx

    return {
        "rx_bytes": rx_bytes,
        "tx_bytes": tx_bytes,
        "interfaces": interfaces,
    }


def network_rates_mb_s(previous, current, *, elapsed_s):
    if elapsed_s <= 0:
        return {"net_rx_mb_s": 0.0, "net_tx_mb_s": 0.0}

    rx_delta = max(0, current["rx_bytes"] - previous["rx_bytes"])
    tx_delta = max(0, current["tx_bytes"] - previous["tx_bytes"])
    return {
        "net_rx_mb_s": rx_delta / _MB / elapsed_s,
        "net_tx_mb_s": tx_delta / _MB / elapsed_s,
    }


def parse_nvidia_smi_csv(text):
    gpus = []
    for row in csv.reader(StringIO(text)):
        if not row or not any(cell.strip() for cell in row):
            continue

        padded = [cell.strip() for cell in row] + [""] * 8
        gpu_index, name, uuid, util, mem_used, mem_total, power, temperature = padded[:8]
        mem_used_mb = _parse_float(mem_used)
        mem_total_mb = _parse_float(mem_total)
        mem_used_pct = (
            mem_used_mb / mem_total_mb * 100.0
            if mem_used_mb is not None and mem_total_mb
            else None
        )

        gpus.append({
            "gpu_index": int(gpu_index),
            "gpu_name": name,
            "gpu_uuid": uuid,
            "gpu_util_pct": _parse_float(util),
            "gpu_mem_used_mb": mem_used_mb,
            "gpu_mem_total_mb": mem_total_mb,
            "gpu_mem_used_pct": mem_used_pct,
            "gpu_power_w": _parse_float(power),
            "gpu_temperature_c": _parse_float(temperature),
        })

    return gpus


def summarize_samples(
    samples,
    *,
    gpu_details,
    error_count=0,
    interval_sec=1.0,
    backend="nvidia-smi",
):
    sample_count = len(samples)
    aggregate = {
        "cpu_util_avg_pct": _average(_sample_values(samples, "cpu_util_pct")),
        "cpu_util_p95_pct": _p95(_sample_values(samples, "cpu_util_pct")),
        "cpu_util_max_pct": _maximum(_sample_values(samples, "cpu_util_pct")),
        "mem_used_avg_mb": _average(_sample_values(samples, "mem_used_mb")),
        "mem_used_p95_mb": _p95(_sample_values(samples, "mem_used_mb")),
        "mem_used_max_mb": _maximum(_sample_values(samples, "mem_used_mb")),
        "gpu_util_avg_pct": _average(_sample_values(samples, "gpu_util_avg_pct")),
        "gpu_util_p95_pct": _p95(_sample_values(samples, "gpu_util_avg_pct")),
        "gpu_util_max_pct": _maximum(_sample_values(samples, "gpu_util_avg_pct")),
    }

    return {
        "available": sample_count > 0,
        "backend": backend,
        "interval_sec": interval_sec,
        "sample_count": sample_count,
        "error_count": error_count,
        "aggregate": aggregate,
        "gpu_details": gpu_details,
    }


def _parse_float(value):
    value = value.strip()
    if not value or value.upper() == "N/A" or value.startswith("["):
        return None
    return float(value)


def _sample_values(samples, key):
    return [
        float(sample[key])
        for sample in samples
        if sample.get(key) is not None
    ]


def _average(values):
    if not values:
        return None
    return sum(values) / len(values)


def _maximum(values):
    if not values:
        return None
    return max(values)


def _p95(values):
    if not values:
        return None
    sorted_values = sorted(values)
    index = math.ceil(0.95 * len(sorted_values)) - 1
    return sorted_values[index]
