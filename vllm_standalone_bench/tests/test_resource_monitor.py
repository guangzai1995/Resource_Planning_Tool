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
