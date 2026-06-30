"""Estimate DeepSeek v4 flash 8-card Ascend benchmark data from P800 baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_SHEET = "671B-P800-8测试数据"
OUTPUT_FILENAME = "deepseek_v4_flash_910b_910c_8card_estimate.xlsx"
GPU_COUNT = 8
MEMORY_UTILIZATION = 0.90
FRAMEWORK_OVERHEAD_GB = 3.0
ATTENTION_FACTOR = 2.0

SOURCE_COLUMNS = [
    "输入长度",
    "输出长度",
    "并发数",
    "输出tokens总吞吐",
    "首tokens时延TP90（ms）",
    "首tokens时延TP99（ms）",
    "最大首tokens时延（ms）",
    "平均首tokens时延（ms）",
    "增量时延TP90（ms）",
    "增量时延TP99（ms）",
    "最大增量时延（ms）",
    "平均增量时延（ms）",
]

AUDIT_COLUMNS = [
    "base_sheet",
    "base_row",
    "target_gpu",
    "gpu_count",
    "runtime_efficiency",
    "compute_scale",
    "bandwidth_scale",
    "decode_scale",
    "bottleneck",
    "required_per_card_gb",
    "confidence",
    "assumption_notes",
]


@dataclass(frozen=True)
class GpuAssumption:
    name: str
    memory_gb: float
    bandwidth_gbs: float
    bf16_tflops: float
    spec_source: str


@dataclass(frozen=True)
class ModelAssumption:
    name: str
    total_params_b: float
    active_params_b: float
    layers: int
    hidden_size: int
    num_kv_heads: int
    head_size: int
    weight_bytes: float
    kv_bytes: float
    spec_source: str

    def with_updates(self, **kwargs: Any) -> "ModelAssumption":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RuntimeScenario:
    name: str
    base_efficiency: float
    target_efficiency: float
    label: str


@dataclass(frozen=True)
class BaselineRow:
    excel_row: int
    input_tokens: float
    output_tokens: float
    concurrency: float
    throughput_tokens_s: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    ttft_max_ms: float
    ttft_mean_ms: float
    decode_latency_p90_ms: float
    decode_latency_p99_ms: float
    decode_latency_max_ms: float
    decode_latency_mean_ms: float


DEFAULT_GPU_ASSUMPTIONS = {
    "P800": GpuAssumption("P800", 64.0, 2000.0, 280.0, "project_seed_data"),
    "910B": GpuAssumption("910B", 64.0, 2000.0, 280.0, "p800_equivalent_assumption"),
    "910C": GpuAssumption("910C", 128.0, 4000.0, 780.0, "public_cloudmatrix384_assumption"),
}

DEFAULT_MODEL_ASSUMPTION = ModelAssumption(
    name="DeepSeek v4 flash proxy",
    total_params_b=671.0,
    active_params_b=37.0,
    layers=61,
    hidden_size=7168,
    num_kv_heads=128,
    head_size=128,
    weight_bytes=1.0,
    kv_bytes=2.0,
    spec_source="DeepSeek-V3/R1 proxy",
)

DEFAULT_RUNTIME_SCENARIOS = {
    "conservative": RuntimeScenario("conservative", 0.70, 0.55, "保守"),
    "base": RuntimeScenario("base", 0.70, 0.70, "基准"),
    "optimistic": RuntimeScenario("optimistic", 0.70, 0.85, "乐观"),
}


def required_memory_per_card_gb(
    *,
    model: ModelAssumption,
    gpu_count: int,
    input_tokens: float,
    output_tokens: float,
    concurrency: float,
    framework_overhead_gb: float,
) -> float:
    weights_gb = model.total_params_b * 1e9 * model.weight_bytes / 1024**3
    kv_cache_gb = (
        2
        * model.layers
        * model.num_kv_heads
        * model.head_size
        * model.kv_bytes
        * (input_tokens + output_tokens)
        * concurrency
        / 1024**3
    )
    return weights_gb / gpu_count + kv_cache_gb / gpu_count + framework_overhead_gb


def decode_bytes_per_token(
    *,
    model: ModelAssumption,
    input_tokens: float,
    output_tokens: float,
) -> float:
    generated_tokens_so_far = output_tokens / 2
    weight_bytes = model.active_params_b * 1e9 * model.weight_bytes
    kv_bytes = (
        2
        * model.layers
        * model.num_kv_heads
        * model.head_size
        * model.kv_bytes
        * (input_tokens + generated_tokens_so_far)
    )
    return weight_bytes + kv_bytes


def prefill_flops(*, model: ModelAssumption, input_tokens: float) -> float:
    return (
        2 * model.active_params_b * 1e9 * input_tokens
        + ATTENTION_FACTOR * model.layers * input_tokens**2 * model.hidden_size
    )


def confidence_for(target_gpu: GpuAssumption) -> float:
    hardware_score = 0.70 if target_gpu.name == "910B" else 0.60
    return round(
        0.30 * 1.00
        + 0.25 * hardware_score
        + 0.20 * 0.65
        + 0.15 * 0.60
        + 0.10 * 0.90,
        2,
    )


def estimate_row(
    *,
    baseline: BaselineRow,
    target_gpu: GpuAssumption,
    base_gpu: GpuAssumption,
    model: ModelAssumption,
    runtime: RuntimeScenario,
) -> dict[str, Any]:
    target_model = model
    base_model = model
    required_gb = required_memory_per_card_gb(
        model=target_model,
        gpu_count=GPU_COUNT,
        input_tokens=baseline.input_tokens,
        output_tokens=baseline.output_tokens,
        concurrency=baseline.concurrency,
        framework_overhead_gb=FRAMEWORK_OVERHEAD_GB,
    )
    feasible = required_gb <= target_gpu.memory_gb * MEMORY_UTILIZATION

    compute_scale = (
        (target_gpu.bf16_tflops * runtime.target_efficiency)
        / (base_gpu.bf16_tflops * runtime.base_efficiency)
        * (base_model.active_params_b / target_model.active_params_b)
    )
    bandwidth_scale = (
        (target_gpu.bandwidth_gbs * runtime.target_efficiency)
        / (base_gpu.bandwidth_gbs * runtime.base_efficiency)
        * (
            decode_bytes_per_token(
                model=base_model,
                input_tokens=baseline.input_tokens,
                output_tokens=baseline.output_tokens,
            )
            / decode_bytes_per_token(
                model=target_model,
                input_tokens=baseline.input_tokens,
                output_tokens=baseline.output_tokens,
            )
        )
    )
    decode_scale = min(compute_scale, bandwidth_scale)
    prefill_compute_scale = compute_scale * (
        prefill_flops(model=base_model, input_tokens=baseline.input_tokens)
        / prefill_flops(model=target_model, input_tokens=baseline.input_tokens)
    )
    bottleneck = "compute" if compute_scale <= bandwidth_scale else "memory_bandwidth"
    if not feasible:
        bottleneck = "memory_infeasible"

    def scaled_decode(value: float) -> float | None:
        return round(value / decode_scale, 4) if feasible else None

    def scaled_ttft(value: float) -> float | None:
        return round(value / prefill_compute_scale, 4) if feasible else None

    return {
        "输入长度": baseline.input_tokens,
        "输出长度": baseline.output_tokens,
        "并发数": baseline.concurrency,
        "输出tokens总吞吐": round(baseline.throughput_tokens_s * decode_scale, 4) if feasible else None,
        "首tokens时延TP90（ms）": scaled_ttft(baseline.ttft_p90_ms),
        "首tokens时延TP99（ms）": scaled_ttft(baseline.ttft_p99_ms),
        "最大首tokens时延（ms）": scaled_ttft(baseline.ttft_max_ms),
        "平均首tokens时延（ms）": scaled_ttft(baseline.ttft_mean_ms),
        "增量时延TP90（ms）": scaled_decode(baseline.decode_latency_p90_ms),
        "增量时延TP99（ms）": scaled_decode(baseline.decode_latency_p99_ms),
        "最大增量时延（ms）": scaled_decode(baseline.decode_latency_max_ms),
        "平均增量时延（ms）": scaled_decode(baseline.decode_latency_mean_ms),
        "base_sheet": SOURCE_SHEET,
        "base_row": baseline.excel_row,
        "target_gpu": target_gpu.name,
        "gpu_count": GPU_COUNT,
        "runtime_efficiency": runtime.target_efficiency,
        "compute_scale": round(compute_scale, 6),
        "bandwidth_scale": round(bandwidth_scale, 6),
        "decode_scale": round(decode_scale, 6),
        "bottleneck": bottleneck,
        "required_per_card_gb": round(required_gb, 4),
        "confidence": confidence_for(target_gpu),
        "assumption_notes": f"{model.spec_source}; {target_gpu.spec_source}; runtime={runtime.label}",
    }


def load_p800_baseline(workbook_path: Path) -> list[BaselineRow]:
    df = pd.read_excel(workbook_path, sheet_name=SOURCE_SHEET)
    missing = [col for col in SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required baseline columns: {missing}")
    rows: list[BaselineRow] = []
    for index, row in df.dropna(subset=["输入长度", "并发数"]).iterrows():
        rows.append(
            BaselineRow(
                excel_row=int(index) + 2,
                input_tokens=float(row["输入长度"]),
                output_tokens=float(row["输出长度"]),
                concurrency=float(row["并发数"]),
                throughput_tokens_s=float(row["输出tokens总吞吐"]),
                ttft_p90_ms=float(row["首tokens时延TP90（ms）"]),
                ttft_p99_ms=float(row["首tokens时延TP99（ms）"]),
                ttft_max_ms=float(row["最大首tokens时延（ms）"]),
                ttft_mean_ms=float(row["平均首tokens时延（ms）"]),
                decode_latency_p90_ms=float(row["增量时延TP90（ms）"]),
                decode_latency_p99_ms=float(row["增量时延TP99（ms）"]),
                decode_latency_max_ms=float(row["最大增量时延（ms）"]),
                decode_latency_mean_ms=float(row["平均增量时延（ms）"]),
            )
        )
    return rows


def build_estimates(
    baseline_rows: list[BaselineRow],
    *,
    model: ModelAssumption = DEFAULT_MODEL_ASSUMPTION,
    runtime: RuntimeScenario = DEFAULT_RUNTIME_SCENARIOS["base"],
) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []
    base_gpu = DEFAULT_GPU_ASSUMPTIONS["P800"]
    for baseline in baseline_rows:
        for gpu_name in ("910B", "910C"):
            output_rows.append(
                estimate_row(
                    baseline=baseline,
                    target_gpu=DEFAULT_GPU_ASSUMPTIONS[gpu_name],
                    base_gpu=base_gpu,
                    model=model,
                    runtime=runtime,
                )
            )
    return pd.DataFrame(output_rows, columns=SOURCE_COLUMNS + AUDIT_COLUMNS)
