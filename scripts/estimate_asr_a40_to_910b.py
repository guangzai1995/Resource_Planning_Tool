"""Estimate single-card Ascend 910B ASR benchmark rows from A40 results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE = Path("vllm_standalone_bench/results/asr_180s_concurrency_sweep.csv")
OUTPUT_CSV = Path("vllm_standalone_bench/results/asr_180s_concurrency_sweep_910b_single_estimate.csv")
OUTPUT_XLSX = Path("vllm_standalone_bench/results/asr_180s_concurrency_sweep_910b_single_estimate.xlsx")

A40_MEMORY_GB = 48.0
A40_BANDWIDTH_GBS = 696.0
A40_LOW_PRECISION_TENSOR_TFLOPS = 149.7

ASCEND_910B_MEMORY_GB = 64.0
ASCEND_910B_BANDWIDTH_GBS = 2000.0
ASCEND_910B_BF16_TFLOPS = 280.0


def scale_latency(series: pd.Series, scale: float) -> pd.Series:
    return (series / scale).round(4)


def scale_e2e(df: pd.DataFrame, e2e_col: str, ttft_col: str, prefill_scale: float, decode_scale: float) -> pd.Series:
    decode_component = (df[e2e_col] - df[ttft_col]).clip(lower=0)
    return (df[ttft_col] / prefill_scale + decode_component / decode_scale).round(4)


def build_estimate(df: pd.DataFrame) -> pd.DataFrame:
    compute_scale = ASCEND_910B_BF16_TFLOPS / A40_LOW_PRECISION_TENSOR_TFLOPS
    bandwidth_scale = ASCEND_910B_BANDWIDTH_GBS / A40_BANDWIDTH_GBS
    decode_scale = min(compute_scale, bandwidth_scale)
    prefill_scale = compute_scale
    memory_usage_scale = A40_MEMORY_GB / ASCEND_910B_MEMORY_GB

    out = df.copy()
    out["backend"] = "estimated-910b-single-card"

    for col in ["throughput_tok_s", "decode_effective_tok_s"]:
        out[col] = (out[col] * decode_scale).round(4)
    for col in ["input_throughput_tok_s", "prefill_effective_tok_s"]:
        out[col] = (out[col] * prefill_scale).round(4)

    out["throughput_req_s"] = (
        out["throughput_tok_s"] / out["avg_output_tokens"].where(out["avg_output_tokens"] > 0)
    ).fillna(out["throughput_req_s"] * decode_scale).round(4)

    for col in ["ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms"]:
        out[col] = scale_latency(out[col], prefill_scale)
    for col in ["tpot_mean_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms"]:
        out[col] = scale_latency(out[col], decode_scale)

    for suffix in ["mean", "p50", "p90", "p99"]:
        out[f"e2el_{suffix}_ms"] = scale_e2e(
            df,
            f"e2el_{suffix}_ms",
            f"ttft_{suffix}_ms",
            prefill_scale,
            decode_scale,
        )

    for col in ["avg_gpu_kv_cache_usage", "peak_gpu_kv_cache_usage"]:
        out[col] = (out[col] * memory_usage_scale).round(4)

    out["duration_s"] = (out["n_success"] / out["throughput_req_s"].where(out["throughput_req_s"] > 0)).round(4)
    out["rtfx"] = (out["audio_duration_s_total"] / out["duration_s"].where(out["duration_s"] > 0)).round(4)

    out["source_gpu"] = "A40"
    out["source_gpu_count"] = 1
    out["target_gpu"] = "910B"
    out["target_gpu_count"] = 1
    out["a40_memory_gb"] = A40_MEMORY_GB
    out["a40_bandwidth_gbs"] = A40_BANDWIDTH_GBS
    out["a40_low_precision_tensor_tflops"] = A40_LOW_PRECISION_TENSOR_TFLOPS
    out["target_memory_gb"] = ASCEND_910B_MEMORY_GB
    out["target_bandwidth_gbs"] = ASCEND_910B_BANDWIDTH_GBS
    out["target_bf16_tflops"] = ASCEND_910B_BF16_TFLOPS
    out["compute_scale"] = round(compute_scale, 6)
    out["bandwidth_scale"] = round(bandwidth_scale, 6)
    out["decode_scale"] = round(decode_scale, 6)
    out["prefill_scale"] = round(prefill_scale, 6)
    out["memory_usage_scale"] = round(memory_usage_scale, 6)
    out["bottleneck"] = "compute" if compute_scale <= bandwidth_scale else "memory_bandwidth"
    out["confidence"] = 0.62
    out["assumption_notes"] = (
        "A40 measured single-card ASR rows; A40 48GB/696GBps and 149.7 low-precision "
        "Tensor TFLOPS from NVIDIA A40 data sheet; "
        "910B uses project assumption 64GB/2000GBps/280 BF16 TFLOPS; "
        "same model, same concurrency, same success/failure counts; "
        "throughput scaled by min(compute_scale, bandwidth_scale), latency inverse-scaled."
    )
    return out


def main() -> int:
    df = pd.read_csv(SOURCE)
    out = build_estimate(df)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="910B单卡估算", index=False)
        df.to_excel(writer, sheet_name="A40单卡基线", index=False)
        pd.DataFrame(
            [
                ["source", str(SOURCE)],
                ["target_csv", str(OUTPUT_CSV)],
                ["target_xlsx", str(OUTPUT_XLSX)],
                ["a40_source_url", "https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf"],
                ["910b_assumption_source", "scripts/estimate_deepseek_v4_flash_ascend.py"],
                ["compute_scale", out["compute_scale"].iloc[0]],
                ["bandwidth_scale", out["bandwidth_scale"].iloc[0]],
                ["decode_scale", out["decode_scale"].iloc[0]],
                ["formula", "target_throughput = base_throughput * min(compute_scale, bandwidth_scale)"],
                ["formula", "target_latency = base_latency / matching_scale"],
            ],
            columns=["item", "value"],
        ).to_excel(writer, sheet_name="估算说明", index=False)
    print(OUTPUT_CSV)
    print(OUTPUT_XLSX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
