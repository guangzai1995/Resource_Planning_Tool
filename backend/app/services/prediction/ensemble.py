"""
混合预测引擎：插值优先 + vLLM 建模兜底
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.services.prediction import interpolation as interp_engine
from app.services.prediction import vllm_model as model_engine

logger = get_logger(__name__)


def _get_model_and_gpu(db: Session, gpu_name: str, model_name: str):
    """从数据库加载 ModelSpec 和 GPUSpec"""
    from app.models.gpu_spec import GpuSpec
    from app.models.model import Model
    from app.services.prediction.vllm_model import GPUSpec, ModelSpec

    gpu_row = db.query(GpuSpec).filter_by(name=gpu_name).first()
    model_row = db.query(Model).filter_by(name=model_name).first()

    gpu_spec = None
    if gpu_row:
        gpu_spec = GPUSpec(
            name=gpu_row.name,
            memory_gb=gpu_row.memory_gb or 80.0,
            memory_bw_gbps=gpu_row.memory_bandwidth_gbps or 2000.0,
            bf16_tflops=gpu_row.tflops_bf16 or 300.0,
        )

    model_spec = None
    if model_row:
        model_spec = ModelSpec(
            name=model_row.name,
            size_b=model_row.parameter_b,
            is_moe=bool(model_row.is_moe),
            num_layers=model_row.num_layers or 32,
            hidden_size=model_row.hidden_size or 4096,
            num_kv_heads=model_row.num_kv_heads or 8,
            head_size=model_row.head_size or 128,
            quantization=model_row.quantization,
        )

    return gpu_spec, model_spec


def predict(
    db: Session,
    gpu_name: str,
    model_name: str,
    gpu_count: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> dict:
    """
    统一预测入口
    - 插值数据充足 → interpolation
    - 插值数据稀疏 → ensemble
    - 无数据 → model_based
    """
    # 1. 尝试插值
    interp_result = interp_engine.predict(
        db, gpu_name, model_name, gpu_count,
        input_tokens, output_tokens, concurrency
    )

    if interp_result and interp_result.get("data_count", 0) >= 4:
        return {
            "source": "interpolation",
            "confidence": interp_result["confidence"],
            "data_points_used": interp_result["data_count"],
            "throughput_tokens_s": interp_result.get("throughput_tokens_s"),
            "ttft_mean_ms": interp_result.get("ttft_mean_ms"),
            "ttft_p90_ms": interp_result.get("ttft_p90_ms"),
            "ttft_p99_ms": interp_result.get("ttft_p99_ms"),
            "decode_latency_mean_ms": interp_result.get("decode_latency_mean_ms"),
            "decode_latency_p90_ms": interp_result.get("decode_latency_p90_ms"),
            "is_extrapolation": interp_result.get("is_extrapolation", False),
            "warnings": ["外推预测，误差可能偏大"] if interp_result.get("is_extrapolation") else [],
        }

    # 2. 尝试 vLLM 建模
    gpu_spec, model_spec = _get_model_and_gpu(db, gpu_name, model_name)

    if gpu_spec is None or model_spec is None:
        # 无数据且无规格信息，返回不可用
        return {
            "source": "unavailable",
            "confidence": 0.0,
            "data_points_used": 0,
            "output_throughput": None,
            "ttft_mean_ms": None,
            "ttft_p90_ms": None,
            "ttft_p99_ms": None,
            "decode_latency_mean_ms": None,
            "decode_latency_p90_ms": None,
            "is_extrapolation": True,
            "warnings": [f"GPU {gpu_name} 或模型 {model_name} 无规格数据，无法预测"],
        }

    model_result = model_engine.predict(
        model_spec, gpu_spec, gpu_count,
        input_tokens, output_tokens, concurrency
    )

    # 如果有部分插值数据，做混合
    if interp_result and interp_result.get("data_count", 0) >= 1:
        # 加权混合：插值占 0.6，建模占 0.4
        w_i, w_m = 0.6, 0.4
        blended: dict = {}
        for key in ["throughput_tokens_s", "ttft_mean_ms", "ttft_p90_ms"]:
            iv = interp_result.get(key)
            mv = model_result.get(key)
            if iv is not None and mv is not None:
                blended[key] = round(iv * w_i + mv * w_m, 2)
            else:
                blended[key] = iv or mv
        return {
            "source": "ensemble",
            "confidence": round(interp_result["confidence"] * w_i + model_result["confidence"] * w_m, 2),
            "data_points_used": interp_result["data_count"],
            "throughput_tokens_s": blended.get("throughput_tokens_s"),
            "ttft_mean_ms": blended.get("ttft_mean_ms"),
            "ttft_p90_ms": blended.get("ttft_p90_ms"),
            "ttft_p99_ms": model_result.get("ttft_p99_ms"),
            "decode_latency_mean_ms": model_result.get("decode_latency_mean_ms"),
            "decode_latency_p90_ms": model_result.get("decode_latency_p90_ms"),
            "is_extrapolation": interp_result.get("is_extrapolation", True),
            "warnings": ["数据点较少，使用插值+建模混合预测"],
        }

    model_result.update({
        "source": "model_based",
        "data_points_used": 0,
        "is_extrapolation": True,
        "warnings": [f"该组合无实测数据，为 vLLM 参数建模理论估算（置信度 {model_result['confidence']:.0%}）"],
    })
    return model_result
