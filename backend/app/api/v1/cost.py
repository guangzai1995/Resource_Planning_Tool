from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.cost import CostOptimizeQuery, CostOptimizeResponse, CostRecommendation
from app.services.cost_optimizer import optimize

router = APIRouter(prefix="/cost", tags=["cost"])


@router.get("/optimize", response_model=CostOptimizeResponse)
def cost_optimize(
    target_concurrency: int = Query(..., ge=1),
    model_name: str = Query(...),
    input_tokens: int = Query(..., ge=1),
    output_tokens: int = Query(256, ge=1),
    max_ttft_ms: float = Query(3000.0),
    min_throughput_per_user: float = Query(1.0),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """成本优化：枚举所有 GPU×卡数 组合，返回最优方案"""
    candidates = optimize(
        db=db,
        target_concurrency=target_concurrency,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        max_ttft_ms=max_ttft_ms,
        min_throughput_per_user=min_throughput_per_user,
        top_k=top_k,
    )

    recommendations = [
        CostRecommendation(
            rank=c["rank"],
            gpu_name=c["gpu_name"],
            gpu_count=c["gpu_count"],
            price_per_hour=c["price_per_hour"],
            max_concurrency=c.get("max_concurrency"),
            utilization_rate=c.get("utilization_rate"),
            cost_per_1m_tokens=c.get("cost_per_1m_tokens"),
            source=c["source"],
            confidence=c["confidence"],
            warnings=c.get("warnings", []),
        )
        for c in candidates
    ]

    return CostOptimizeResponse(
        recommendations=recommendations,
        query_params={
            "target_concurrency": target_concurrency,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_ttft_ms": max_ttft_ms,
        },
    )
