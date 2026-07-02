import pytest

import resource_monitor as rm


def test_nvidia_smi_query_is_subprocess_argv():
    assert rm.NVIDIA_SMI_QUERY == [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]


def test_parse_cpu_stat_and_compute_utilization():
    first = rm.parse_proc_stat("cpu  100 0 50 850 0 0 0 0 0 0\n")
    second = rm.parse_proc_stat("cpu  180 0 70 950 0 0 0 0 0 0\n")

    assert rm.cpu_utilization_pct(first, second) == pytest.approx(50.0)
    assert rm.cpu_utilization_pct(first, first) is None
    assert rm.cpu_utilization_pct(second, first) is None


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

    invalid_rates = rm.network_rates_mb_s(
        rm.parse_net_dev(before),
        rm.parse_net_dev(after),
        elapsed_s=0.0,
    )
    assert invalid_rates["net_rx_mb_s"] is None
    assert invalid_rates["net_tx_mb_s"] is None


def test_parse_nvidia_smi_csv_handles_multi_gpu_and_empty_values():
    text = "\n".join([
        "0, NVIDIA H200, GPU-0, 98, 77320, 81559, 612.50, 76",
        "1, NVIDIA H200, GPU-1, N/A, 12000, 81559, Not Supported, Not Supported",
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
        {
            "cpu_util_pct": 10.0,
            "mem_used_mb": 100.0,
            "mem_used_pct": 25.0,
            "net_rx_mb_s": 1.0,
            "net_tx_mb_s": 0.5,
            "disk_read_mb_s": 10.0,
            "disk_write_mb_s": 4.0,
            "gpu_util_avg_pct": 20.0,
            "gpu_mem_used_mb": 1000.0,
            "gpu_mem_total_mb": 8000.0,
            "gpu_mem_used_pct": 12.5,
            "gpu_power_w": 100.0,
            "gpu_temperature_c": 50.0,
        },
        {
            "cpu_util_pct": 20.0,
            "mem_used_mb": 200.0,
            "mem_used_pct": 50.0,
            "net_rx_mb_s": 2.0,
            "net_tx_mb_s": 1.0,
            "disk_read_mb_s": 20.0,
            "disk_write_mb_s": 8.0,
            "gpu_util_avg_pct": 40.0,
            "gpu_mem_used_mb": 2000.0,
            "gpu_mem_total_mb": 8000.0,
            "gpu_mem_used_pct": 25.0,
            "gpu_power_w": 200.0,
            "gpu_temperature_c": 60.0,
        },
        {
            "cpu_util_pct": 30.0,
            "mem_used_mb": 300.0,
            "mem_used_pct": 75.0,
            "net_rx_mb_s": 3.0,
            "net_tx_mb_s": 1.5,
            "disk_read_mb_s": 30.0,
            "disk_write_mb_s": 12.0,
            "gpu_util_avg_pct": 60.0,
            "gpu_mem_used_mb": 3000.0,
            "gpu_mem_total_mb": 8000.0,
            "gpu_mem_used_pct": 37.5,
            "gpu_power_w": 300.0,
            "gpu_temperature_c": 70.0,
        },
    ]
    gpu_details = [{"gpu_index": 0}, {"gpu_index": 1}]

    summary = rm.summarize_samples(samples, gpu_details=gpu_details)

    assert summary["system_available"] is True
    assert summary["gpu_available"] is True
    assert summary["gpus"] == gpu_details
    assert "gpu_details" not in summary
    assert summary["sample_count"] == 3

    assert set(summary["aggregate"]) == {
        "cpu_util_avg_pct",
        "cpu_util_p95_pct",
        "cpu_util_max_pct",
        "mem_used_avg_mb",
        "mem_used_p95_mb",
        "mem_used_max_mb",
        "mem_used_avg_pct",
        "mem_used_p95_pct",
        "mem_used_max_pct",
        "net_rx_avg_mb_s",
        "net_rx_p95_mb_s",
        "net_rx_max_mb_s",
        "net_tx_avg_mb_s",
        "net_tx_p95_mb_s",
        "net_tx_max_mb_s",
        "disk_read_avg_mb_s",
        "disk_read_p95_mb_s",
        "disk_read_max_mb_s",
        "disk_write_avg_mb_s",
        "disk_write_p95_mb_s",
        "disk_write_max_mb_s",
        "gpu_count",
        "gpu_util_avg_pct",
        "gpu_util_p95_pct",
        "gpu_util_max_pct",
        "gpu_mem_used_avg_mb",
        "gpu_mem_used_p95_mb",
        "gpu_mem_used_max_mb",
        "gpu_mem_total_mb",
        "gpu_mem_used_max_pct",
        "gpu_power_avg_w",
        "gpu_power_p95_w",
        "gpu_power_max_w",
        "gpu_temp_avg_c",
        "gpu_temp_p95_c",
        "gpu_temp_max_c",
    }
    aggregate = summary["aggregate"]
    assert aggregate["cpu_util_avg_pct"] == pytest.approx(20.0)
    assert aggregate["cpu_util_p95_pct"] == pytest.approx(30.0)
    assert aggregate["cpu_util_max_pct"] == pytest.approx(30.0)
    assert aggregate["mem_used_avg_mb"] == pytest.approx(200.0)
    assert aggregate["mem_used_p95_mb"] == pytest.approx(300.0)
    assert aggregate["mem_used_max_mb"] == pytest.approx(300.0)
    assert aggregate["mem_used_avg_pct"] == pytest.approx(50.0)
    assert aggregate["mem_used_p95_pct"] == pytest.approx(75.0)
    assert aggregate["mem_used_max_pct"] == pytest.approx(75.0)
    assert aggregate["net_rx_avg_mb_s"] == pytest.approx(2.0)
    assert aggregate["net_rx_p95_mb_s"] == pytest.approx(3.0)
    assert aggregate["net_rx_max_mb_s"] == pytest.approx(3.0)
    assert aggregate["net_tx_avg_mb_s"] == pytest.approx(1.0)
    assert aggregate["net_tx_p95_mb_s"] == pytest.approx(1.5)
    assert aggregate["net_tx_max_mb_s"] == pytest.approx(1.5)
    assert aggregate["disk_read_avg_mb_s"] == pytest.approx(20.0)
    assert aggregate["disk_read_p95_mb_s"] == pytest.approx(30.0)
    assert aggregate["disk_read_max_mb_s"] == pytest.approx(30.0)
    assert aggregate["disk_write_avg_mb_s"] == pytest.approx(8.0)
    assert aggregate["disk_write_p95_mb_s"] == pytest.approx(12.0)
    assert aggregate["disk_write_max_mb_s"] == pytest.approx(12.0)
    assert aggregate["gpu_count"] == 2
    assert aggregate["gpu_util_avg_pct"] == pytest.approx(40.0)
    assert aggregate["gpu_util_p95_pct"] == pytest.approx(60.0)
    assert aggregate["gpu_util_max_pct"] == pytest.approx(60.0)
    assert aggregate["gpu_mem_used_avg_mb"] == pytest.approx(2000.0)
    assert aggregate["gpu_mem_used_p95_mb"] == pytest.approx(3000.0)
    assert aggregate["gpu_mem_used_max_mb"] == pytest.approx(3000.0)
    assert aggregate["gpu_mem_total_mb"] == pytest.approx(8000.0)
    assert aggregate["gpu_mem_used_max_pct"] == pytest.approx(37.5)
    assert aggregate["gpu_power_avg_w"] == pytest.approx(200.0)
    assert aggregate["gpu_power_p95_w"] == pytest.approx(300.0)
    assert aggregate["gpu_power_max_w"] == pytest.approx(300.0)
    assert aggregate["gpu_temp_avg_c"] == pytest.approx(60.0)
    assert aggregate["gpu_temp_p95_c"] == pytest.approx(70.0)
    assert aggregate["gpu_temp_max_c"] == pytest.approx(70.0)
