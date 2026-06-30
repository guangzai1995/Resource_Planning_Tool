import pytest

from scripts.estimate_deepseek_v4_flash_ascend import (
    DEFAULT_GPU_ASSUMPTIONS,
    DEFAULT_MODEL_ASSUMPTION,
    DEFAULT_RUNTIME_SCENARIOS,
    BaselineRow,
    estimate_row,
    required_memory_per_card_gb,
)


def test_required_memory_per_card_changes_with_weight_precision():
    bf16 = required_memory_per_card_gb(
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=2.0),
        gpu_count=8,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        framework_overhead_gb=3.0,
    )
    int8 = required_memory_per_card_gb(
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=1.0),
        gpu_count=8,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        framework_overhead_gb=3.0,
    )
    assert bf16 > int8
    assert int8 > 3.0


def test_910c_compute_scale_uses_target_hardware_ratio():
    baseline = BaselineRow(
        excel_row=2,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        throughput_tokens_s=100.0,
        ttft_p90_ms=400.0,
        ttft_p99_ms=500.0,
        ttft_max_ms=550.0,
        ttft_mean_ms=300.0,
        decode_latency_p90_ms=40.0,
        decode_latency_p99_ms=45.0,
        decode_latency_max_ms=60.0,
        decode_latency_mean_ms=35.0,
    )
    result = estimate_row(
        baseline=baseline,
        target_gpu=DEFAULT_GPU_ASSUMPTIONS["910C"],
        base_gpu=DEFAULT_GPU_ASSUMPTIONS["P800"],
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=1.0),
        runtime=DEFAULT_RUNTIME_SCENARIOS["base"],
    )
    assert result["target_gpu"] == "910C"
    assert result["gpu_count"] == 8
    assert result["compute_scale"] == pytest.approx(780 / 280, rel=1e-6)
    assert result["decode_scale"] == pytest.approx(
        min(result["compute_scale"], result["bandwidth_scale"]),
        rel=1e-6,
    )
    assert result["输出tokens总吞吐"] == pytest.approx(100.0 * result["decode_scale"])
    assert 0.0 <= result["confidence"] <= 1.0
