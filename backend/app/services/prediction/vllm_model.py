"""
vLLM 参数建模推算引擎：当无实测数据时，基于 Roofline 模型进行理论推算
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUSpec:
    name: str
    memory_gb: float
    memory_bw_gbps: float
    bf16_tflops: float


@dataclass
class ModelSpec:
    name: str
    size_b: float
    is_moe: bool
    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_size: int
    quantization: str | None = None


def _get_quant_bytes(quantization: str | None) -> float:
    return {None: 2.0, "AWQ": 0.5, "GPTQ": 0.5, "FP8": 1.0}.get(quantization, 2.0)


def estimate_ttft(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    input_tokens: int,
    concurrency: int,
) -> float:
    """估算首 Token 延迟（ms）"""
    d = model.hidden_size
    n_layers = model.num_layers

    # Prefill FLOPs (Attention + FFN)
    # Attention: QKV proj + attention scores + output proj
    attn_flops_per_layer = (
        4 * input_tokens * d * d  # QKV + O proj
        + 2 * input_tokens * input_tokens * d  # attention scores
    )
    # FFN SwiGLU: ~2.67x hidden  
    ffn_hidden = int(d * 8 / 3)
    ffn_flops_per_layer = 2 * input_tokens * d * ffn_hidden * 2  # gate + up + down
    total_flops = n_layers * (attn_flops_per_layer + ffn_flops_per_layer)

    # 有效算力
    tp_eff = settings.tp_efficiency_table.get(gpu_count, 0.85)
    effective_tflops = gpu.bf16_tflops * gpu_count * tp_eff * settings.GPU_COMPUTE_UTILIZATION
    if effective_tflops <= 0:
        return 999999.0

    compute_ms = total_flops / (effective_tflops * 1e12) * 1000

    # 并发等待系数（chunked prefill 近似）
    queue_factor = 1.0 + (concurrency / settings.MAX_NUM_SEQS) * 0.4

    ttft_ms = compute_ms * queue_factor + settings.SCHEDULING_OVERHEAD_MS
    return max(0.0, ttft_ms)


def estimate_decode_throughput(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    concurrency: int,
    avg_seq_len: int,
) -> float:
    """估算系统级 Decode 吞吐量（tokens/s）"""
    quant_bytes = _get_quant_bytes(model.quantization)
    # 模型权重读取字节数 / step
    weight_bytes = model.size_b * 1e9 * quant_bytes / gpu_count

    # KV Cache 读写字节数
    kv_bytes_per_token = (
        2 * model.num_kv_heads * model.head_size * 2  # bf16
        * model.num_layers / gpu_count
    )
    kv_bytes = concurrency * avg_seq_len * kv_bytes_per_token

    bytes_per_step = weight_bytes + kv_bytes

    # 有效带宽
    effective_bw = gpu.memory_bw_gbps * 1e9 * gpu_count * settings.BW_EFFICIENCY
    if effective_bw <= 0:
        return 0.0

    step_time_s = bytes_per_step / effective_bw
    if step_time_s <= 0:
        return 0.0

    return concurrency / step_time_s


def estimate_max_concurrency(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    input_tokens: int,
    output_tokens: int,
    max_ttft_ms: float,
    min_tput_per_user: float,
) -> tuple[int, str]:
    """
    二分搜索满足约束的最大并发数
    返回 (max_concurrency, bottleneck)
    """
    lo, hi = 1, 512
    result = 0
    bottleneck = "none"

    for conc in range(lo, hi + 1, 1):
        ttft = estimate_ttft(model, gpu, gpu_count, input_tokens, conc)
        tput = estimate_decode_throughput(model, gpu, gpu_count, conc,
                                          input_tokens + output_tokens // 2)
        per_user = tput / conc if conc > 0 else 0

        if ttft > max_ttft_ms:
            bottleneck = "compute"
            break
        if per_user < min_tput_per_user:
            bottleneck = "memory"
            break
        result = conc

    return result, bottleneck


def predict(
    model_spec: ModelSpec,
    gpu_spec: GPUSpec,
    gpu_count: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> dict:
    """vLLM 建模推算，返回预测结果"""
    ttft = estimate_ttft(model_spec, gpu_spec, gpu_count, input_tokens, concurrency)
    avg_seq_len = input_tokens + output_tokens // 2
    throughput = estimate_decode_throughput(
        model_spec, gpu_spec, gpu_count, concurrency, avg_seq_len
    )

    # TTFT P90 估算（加 20% 抖动）
    ttft_p90 = ttft * 1.2

    return {
        "throughput_tokens_s": round(throughput, 2),
        "ttft_mean_ms": round(ttft, 1),
        "ttft_p90_ms": round(ttft_p90, 1),
        "ttft_p99_ms": round(ttft * 1.5, 1),
        "decode_latency_mean_ms": round(1000.0 / (throughput / concurrency) if throughput > 0 else 0, 1),
        "decode_latency_p90_ms": round(1000.0 / (throughput / concurrency) * 1.2 if throughput > 0 else 0, 1),
        "confidence": 0.55,
        "data_count": 0,
        "is_extrapolation": False,
    }
