import math
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import cache_key, get_prediction, set_prediction
from app.schemas.predict import PredictRequest, PredictResponse, PredictResult
from app.services.prediction.ensemble import predict as ensemble_predict

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """性能预测接口：插值优先，vLLM 建模兜底"""
    key = cache_key(
        "pred", req.gpu_name, req.model_name, req.gpu_count,
        req.input_tokens, req.output_tokens, req.concurrency
    )
    cached = get_prediction(key)
    if cached:
        return cached

    result = ensemble_predict(
        db,
        gpu_name=req.gpu_name,
        model_name=req.model_name,
        gpu_count=req.gpu_count,
        input_tokens=req.input_tokens,
        output_tokens=req.output_tokens,
        concurrency=req.concurrency,
    )

    # 推算最大安全并发数
    max_safe_conc = None
    recommended_conc = None
    if req.max_ttft_ms and result.get("ttft_mean_ms"):
        # 简单估算：当前并发下的 TTFT 若超限则减小
        if result["ttft_mean_ms"] <= req.max_ttft_ms:
            max_safe_conc = int(req.concurrency * (req.max_ttft_ms / result["ttft_mean_ms"]))
            recommended_conc = min(max_safe_conc, req.concurrency)

    response = PredictResponse(
        source=result.get("source", "unavailable"),
        confidence=result.get("confidence", 0.0),
        data_points_used=result.get("data_points_used", 0),
        result=PredictResult(
            predicted_ttft_mean_ms=result.get("ttft_mean_ms"),
            predicted_ttft_p90_ms=result.get("ttft_p90_ms"),
            predicted_throughput_tokens_s=result.get("throughput_tokens_s"),
            max_safe_concurrency=max_safe_conc,
            recommended_concurrency=recommended_conc,
        ),
        warnings=result.get("warnings", []),
        metadata={
            "gpu_name": req.gpu_name,
            "model_name": req.model_name,
            "gpu_count": req.gpu_count,
        },
    )
    set_prediction(key, response)
    return response


@router.get("/sweep")
def predict_sweep(
    gpu_name: str = Query(...),
    model_name: str = Query(...),
    gpu_count: int = Query(1, ge=1, le=128),
    input_tokens: int = Query(..., ge=1),
    output_tokens: int = Query(256, ge=1),
    max_concurrency: int = Query(128, ge=1, le=2048),
    db: Session = Depends(get_db),
):
    """返回从 1 到 max_concurrency 的并发数扫描预测（用于前端绘制曲线）"""
    # 生成对数等间距的并发点（约 20 个）
    n_points = 20
    conc_set: set[int] = {1, max_concurrency}
    for i in range(1, n_points - 1):
        c = max(1, round(math.exp(math.log(max(max_concurrency, 2)) * i / (n_points - 1))))
        conc_set.add(c)

    results = []
    for conc in sorted(conc_set):
        ck = cache_key("sweep", gpu_name, model_name, gpu_count, input_tokens, output_tokens, conc)
        cached = get_prediction(ck)
        if cached:
            results.append(cached)
            continue
        try:
            r = ensemble_predict(
                db,
                gpu_name=gpu_name,
                model_name=model_name,
                gpu_count=gpu_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                concurrency=conc,
            )
            throughput = r.get("throughput_tokens_s")
            ttft_ms = r.get("ttft_mean_ms")
            point = {
                "concurrency": conc,
                "throughput_tokens_s": round(throughput, 2) if throughput is not None else None,
                "throughput_per_user_tokens_s": round(throughput / conc, 2) if throughput and conc > 0 else None,
                "ttft_mean_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                "ttft_s": round(ttft_ms / 1000, 4) if ttft_ms is not None else None,
                "decode_latency_mean_ms": round(r.get("decode_latency_mean_ms"), 2) if r.get("decode_latency_mean_ms") is not None else None,
                "source": r.get("source"),
                "confidence": r.get("confidence"),
                "is_extrapolation": r.get("is_extrapolation", False),
            }
            set_prediction(ck, point)
            results.append(point)
        except Exception:
            pass

    return results
