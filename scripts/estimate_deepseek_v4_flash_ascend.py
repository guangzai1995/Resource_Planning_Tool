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

REFERENCE_COLUMNS = [
    "source_file",
    "gpu_family",
    "model",
    "gpu_count",
    "rows",
    "mean_throughput_tokens_s",
    "max_throughput_tokens_s",
    "min_decode_latency_mean_ms",
    "mean_ttft_ms",
    "note",
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
    "P800": GpuAssumption("P800", 96.0, 2000.0, 280.0, "user_confirmed_p800_96gb"),
    "910B": GpuAssumption("910B", 64.0, 2000.0, 280.0, "p800_equivalent_assumption"),
    "910C": GpuAssumption("910C", 128.0, 4000.0, 780.0, "public_cloudmatrix384_assumption"),
}

DEFAULT_BASE_MODEL_ASSUMPTION = ModelAssumption(
    name="DeepSeek R1 INT8 baseline proxy",
    total_params_b=671.0,
    active_params_b=37.0,
    layers=61,
    hidden_size=7168,
    num_kv_heads=128,
    head_size=128,
    weight_bytes=1.0,
    kv_bytes=2.0,
    spec_source="DeepSeek-R1/P800 INT8 baseline proxy",
)

DEFAULT_TARGET_MODEL_ASSUMPTION = ModelAssumption(
    name="DeepSeek V4 Flash INT8 public proxy",
    total_params_b=284.0,
    active_params_b=13.0,
    layers=61,
    hidden_size=7168,
    num_kv_heads=128,
    head_size=128,
    weight_bytes=1.0,
    kv_bytes=2.0,
    spec_source="DeepSeek-V4-Flash public 284B/13B; R1 proxy architecture",
)

DEFAULT_MODEL_ASSUMPTION = DEFAULT_TARGET_MODEL_ASSUMPTION

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
    base_model: ModelAssumption = DEFAULT_BASE_MODEL_ASSUMPTION,
    target_model: ModelAssumption = DEFAULT_TARGET_MODEL_ASSUMPTION,
    model: ModelAssumption | None = None,
    runtime: RuntimeScenario,
) -> dict[str, Any]:
    if model is not None:
        target_model = model
    required_gb = required_memory_per_card_gb(
        model=target_model,
        gpu_count=GPU_COUNT,
        input_tokens=baseline.input_tokens,
        output_tokens=baseline.output_tokens,
        concurrency=baseline.concurrency,
        framework_overhead_gb=FRAMEWORK_OVERHEAD_GB,
    )
    feasible = required_gb <= target_gpu.memory_gb * MEMORY_UTILIZATION

    hardware_compute_scale = (
        (target_gpu.bf16_tflops * runtime.target_efficiency)
        / (base_gpu.bf16_tflops * runtime.base_efficiency)
    )
    compute_scale = (
        hardware_compute_scale
        * (base_model.active_params_b / target_model.active_params_b)
    )
    hardware_bandwidth_scale = (
        (target_gpu.bandwidth_gbs * runtime.target_efficiency)
        / (base_gpu.bandwidth_gbs * runtime.base_efficiency)
    )
    bandwidth_scale = (
        hardware_bandwidth_scale
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
    prefill_compute_scale = hardware_compute_scale * (
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
        "assumption_notes": (
            f"{base_model.spec_source} -> {target_model.spec_source}; "
            f"{target_gpu.spec_source}; runtime={runtime.label}"
        ),
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
    base_model: ModelAssumption = DEFAULT_BASE_MODEL_ASSUMPTION,
    target_model: ModelAssumption = DEFAULT_TARGET_MODEL_ASSUMPTION,
    model: ModelAssumption | None = None,
    runtime: RuntimeScenario = DEFAULT_RUNTIME_SCENARIOS["base"],
) -> pd.DataFrame:
    if model is not None:
        target_model = model
    output_rows: list[dict[str, Any]] = []
    base_gpu = DEFAULT_GPU_ASSUMPTIONS["P800"]
    for baseline in baseline_rows:
        for gpu_name in ("910B", "910C"):
            output_rows.append(
                estimate_row(
                    baseline=baseline,
                    target_gpu=DEFAULT_GPU_ASSUMPTIONS[gpu_name],
                    base_gpu=base_gpu,
                    base_model=base_model,
                    target_model=target_model,
                    runtime=runtime,
                )
            )
    return pd.DataFrame(output_rows, columns=SOURCE_COLUMNS + AUDIT_COLUMNS)


def summarize_910b_reference(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    csv_paths = list(data_dir.glob("910B*/**/*.csv")) + list(data_dir.glob("910B*/**/*.CSV"))
    for csv_path in sorted(csv_paths):
        rel_parts = csv_path.relative_to(data_dir).parts
        if len(rel_parts) < 3:
            continue
        gpu_family, model_name, filename = rel_parts[0], rel_parts[1], rel_parts[2]
        gpu_count = Path(filename).stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rows.append(
            {
                "source_file": str(csv_path),
                "gpu_family": gpu_family,
                "model": model_name,
                "gpu_count": gpu_count,
                "rows": len(df),
                "mean_throughput_tokens_s": round(float(df["输出tokens总吞吐"].mean()), 4),
                "max_throughput_tokens_s": round(float(df["输出tokens总吞吐"].max()), 4),
                "min_decode_latency_mean_ms": round(float(df["平均增量时延（ms）"].min()), 4),
                "mean_ttft_ms": round(float(df["平均首tokens时延（ms）"].mean()), 4),
                "note": "低利用率参考，不参与估算校准",
            }
        )
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


def assumptions_dataframe() -> pd.DataFrame:
    rows = []
    for gpu in DEFAULT_GPU_ASSUMPTIONS.values():
        rows.append(
            {
                "category": "gpu",
                "name": gpu.name,
                "memory_gb": gpu.memory_gb,
                "bandwidth_gbs": gpu.bandwidth_gbs,
                "bf16_tflops_or_efficiency": gpu.bf16_tflops,
                "source": gpu.spec_source,
            }
        )
    for runtime in DEFAULT_RUNTIME_SCENARIOS.values():
        rows.append(
            {
                "category": "runtime",
                "name": runtime.name,
                "memory_gb": None,
                "bandwidth_gbs": None,
                "bf16_tflops_or_efficiency": runtime.target_efficiency,
                "source": runtime.label,
            }
        )
    rows.append(
        {
            "category": "model_base",
            "name": DEFAULT_BASE_MODEL_ASSUMPTION.name,
            "memory_gb": DEFAULT_BASE_MODEL_ASSUMPTION.total_params_b,
            "bandwidth_gbs": DEFAULT_BASE_MODEL_ASSUMPTION.active_params_b,
            "bf16_tflops_or_efficiency": DEFAULT_BASE_MODEL_ASSUMPTION.weight_bytes,
            "source": DEFAULT_BASE_MODEL_ASSUMPTION.spec_source,
        }
    )
    rows.append(
        {
            "category": "model_target",
            "name": DEFAULT_TARGET_MODEL_ASSUMPTION.name,
            "memory_gb": DEFAULT_TARGET_MODEL_ASSUMPTION.total_params_b,
            "bandwidth_gbs": DEFAULT_TARGET_MODEL_ASSUMPTION.active_params_b,
            "bf16_tflops_or_efficiency": DEFAULT_TARGET_MODEL_ASSUMPTION.weight_bytes,
            "source": DEFAULT_TARGET_MODEL_ASSUMPTION.spec_source,
        }
    )
    return pd.DataFrame(rows)


def explanation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["版本", "v1"],
            ["主基线", SOURCE_SHEET],
            ["估算口径", "P800实测基线 + R1 INT8基线模型 -> V4 Flash INT8目标模型 + 硬件Roofline比例"],
            ["P800显存口径", "按用户确认的实测P800 96GB写入假设表；未采用项目旧seed中的64GB"],
            ["目标模型口径", "DeepSeek V4-Flash按公开Hugging Face模型卡284B总参数、13B激活参数；如内部口径为283B/14B，可替换假设后重算"],
            ["显存公式", "required_per_card_gb = total_params_b*1e9*weight_bytes/1024^3/gpu_count + KV_cache_gb/gpu_count + framework_overhead_gb"],
            ["Decode公式", "decode_scale = min(compute_scale, bandwidth_scale); throughput_target = throughput_base * decode_scale"],
            ["compute_scale", "(target_tflops*target_eff)/(base_tflops*base_eff) * (base_active_params_b/target_active_params_b)"],
            ["bandwidth_scale", "(target_bw*target_eff)/(base_bw*base_eff) * (base_decode_bytes_per_token/target_decode_bytes_per_token)"],
            ["TTFT公式", "ttft_target = ttft_base / prefill_compute_scale"],
            ["增量时延公式", "decode_latency_target = decode_latency_base / decode_scale"],
            ["910B参考数据", "只进入参考sheet，不参与校准"],
            ["联网参考数据", "公开硬件规格进入公式假设；公开吞吐/benchmark只进入参考sheet，不参与校准"],
            ["适用边界", "估算不能替代实测；实际部署需用910B/910C复测"],
        ],
        columns=["项目", "说明"],
    )


def public_reference_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_title": "A-IO: Adaptive Inference Orchestration for Memory-Bound NPUs",
                "url": "https://arxiv.org/abs/2604.09752",
                "hardware": "Ascend 910B",
                "model_or_scenario": "OpenPangu 1B/7B, HuggingFace + PyTorch",
                "metric_summary": "Reports OpenPangu 1B/7B TPS under 2K workloads and 32K accuracy on Ascend 910B.",
                "comparability_note": "Single-card small-model paper data; not used to calibrate DeepSeek 8-card MoE estimates.",
            },
            {
                "source_title": "An Empirical Study of OpenPangu Quantization on Ascend NPUs",
                "url": "https://arxiv.org/abs/2606.21257",
                "hardware": "Ascend 910B1",
                "model_or_scenario": "OpenPangu 1B/7B post-training quantization",
                "metric_summary": "Documents Ascend 910B1 64GB HBM environment and quantization behavior.",
                "comparability_note": "Quantization and accuracy reference; not a throughput calibration dataset.",
            },
            {
                "source_title": "Serving Large Language Models on Huawei CloudMatrix384",
                "url": "https://arxiv.org/abs/2506.12708",
                "hardware": "384 x Ascend 910C",
                "model_or_scenario": "DeepSeek-R1 on CloudMatrix-Infer",
                "metric_summary": "Reports 6688 prefill tokens/s/NPU, 1943 decode tokens/s/NPU, and 538 tokens/s/NPU at 15 ms TPOT.",
                "comparability_note": "384-card CloudMatrix-Infer result; upper-bound reference, not directly comparable to 8-card deployment.",
            },
            {
                "source_title": "Tom's Hardware CloudMatrix384 report",
                "url": "https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power",
                "hardware": "Ascend 910C",
                "model_or_scenario": "Public hardware/spec reporting",
                "metric_summary": "Reports 910C at 780 BF16 TFLOPS, 128GB HBM, 3.2TB/s HBM bandwidth, and CloudMatrix384 at 300 BF16 PFLOPS.",
                "comparability_note": "Non-vendor public report; used only as a configurable default assumption/reference.",
            },
        ]
    )


def build_workbook(*, source_workbook: Path, data_dir: Path, output_path: Path) -> Path:
    baseline_rows = load_p800_baseline(source_workbook)
    estimates = build_estimates(baseline_rows)
    baseline_df = pd.read_excel(source_workbook, sheet_name=SOURCE_SHEET)
    reference_df = summarize_910b_reference(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        explanation_dataframe().to_excel(writer, sheet_name="00_估算说明", index=False, startrow=2)
        assumptions_dataframe().to_excel(writer, sheet_name="01_假设表", index=False)
        estimates[estimates["target_gpu"] == "910B"].to_excel(writer, sheet_name="02_910B_8卡估算", index=False)
        estimates[estimates["target_gpu"] == "910C"].to_excel(writer, sheet_name="03_910C_8卡估算", index=False)
        baseline_df.to_excel(writer, sheet_name="04_P800基线", index=False)
        reference_df.to_excel(writer, sheet_name="05_910B低利用率参考", index=False)
        public_reference_dataframe().to_excel(writer, sheet_name="06_联网参考数据", index=False)

        ws = writer.book["00_估算说明"]
        ws["A1"] = "DeepSeek v4 flash 910B/910C 8卡估算"

    return output_path


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workbook", type=Path, default=Path("资源规划工具.xlsx"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs") / OUTPUT_FILENAME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_workbook(
        source_workbook=args.source_workbook,
        data_dir=args.data_dir,
        output_path=args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
