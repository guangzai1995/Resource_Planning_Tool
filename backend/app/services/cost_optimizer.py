"""
成本优化器：遍历所有 GPU × 卡数 组合，返回满足约束的 TOP-K 方案
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.services.prediction.ensemble import predict as ensemble_predict
from app.services.prediction.vllm_model import estimate_max_concurrency, GPUSpec, ModelSpec

logger = get_logger(__name__)

# GPU 卡数候选
CANDIDATE_GPU_COUNTS = [1, 2, 4, 8]


def optimize(
    db: Session,
    target_concurrency: int,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    max_ttft_ms: float,
    min_throughput_per_user: float,
    top_k: int = 5,
) -> list[dict]:
    from app.models.gpu_spec import GpuSpec
    from app.models.model import Model

    gpus = db.query(GpuSpec).all()
    model_row = db.query(Model).filter_by(name=model_name).first()

    candidates = []
    rank = 0

    for gpu in gpus:
        for gpu_count in CANDIDATE_GPU_COUNTS:
            # 预测该组合在 target_concurrency 下的性能
            pred = ensemble_predict(
                db,
                gpu_name=gpu.name,
                model_name=model_name,
                gpu_count=gpu_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                concurrency=target_concurrency,
            )

            if pred["source"] == "unavailable":
                continue

            ttft = pred.get("ttft_mean_ms") or 0.0
            throughput = pred.get("output_throughput") or 0.0
            per_user_tput = throughput / target_concurrency if target_concurrency > 0 else 0

            # 计算最大可承载并发数（通过 vLLM 建模估算）
            max_conc = target_concurrency  # 默认
            if model_row and gpu.memory_bw_gbps and gpu.bf16_tflops:
                model_spec = ModelSpec(
                    name=model_row.name,
                    size_b=model_row.size_b,
                    is_moe=bool(model_row.is_moe),
                    num_layers=model_row.num_layers or 32,
                    hidden_size=model_row.hidden_size or 4096,
                    num_kv_heads=model_row.num_kv_heads or 8,
                    head_size=model_row.head_size or 128,
                    quantization=model_row.quantization,
                )
                gpu_spec = GPUSpec(
                    name=gpu.name,
                    memory_gb=gpu.memory_gb,
                    memory_bw_gbps=gpu.memory_bw_gbps,
                    bf16_tflops=gpu.bf16_tflops,
                )
                max_conc, _ = estimate_max_concurrency(
                    model_spec, gpu_spec, gpu_count,
                    input_tokens, output_tokens,
                    max_ttft_ms, min_throughput_per_user,
                )

            utilization = target_concurrency / max_conc if max_conc > 0 else 999.0
            price_per_hour = gpu.price_per_hour * gpu_count

            # 成本/百万 token = 价格/h / (吞吐量 tokens/s * 3600 / 1e6)
            if throughput > 0:
                tokens_per_hour = throughput * 3600
                cost_per_1m = price_per_hour / (tokens_per_hour / 1e6)
            else:
                cost_per_1m = None

            warnings = list(pred.get("warnings", []))
            if utilization > 1.0:
                warnings.append(f"目标并发({target_concurrency})超过估算最大并发({max_conc})，建议增加卡数")

            candidates.append({
                "gpu_name": gpu.name,
                "gpu_count": gpu_count,
                "price_per_hour": price_per_hour,
                "max_concurrency": max_conc,
                "utilization_rate": round(utilization, 3),
                "cost_per_1m_tokens": round(cost_per_1m, 2) if cost_per_1m else None,
                "source": pred["source"],
                "confidence": pred["confidence"],
                "ttft_mean_ms": ttft,
                "throughput": throughput,
                "warnings": warnings,
                "sort_key": (
                    0 if utilization <= 1.0 else 1,   # 满足约束优先
                    cost_per_1m if cost_per_1m else 9999,  # 低成本优先
                ),
            })

    # 按排序键升序
    candidates.sort(key=lambda x: x["sort_key"])

    # 分配 rank
    result = []
    for i, c in enumerate(candidates[:top_k]):
        c["rank"] = i + 1
        result.append(c)

    return result
