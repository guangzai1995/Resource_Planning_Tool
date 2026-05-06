from pydantic import BaseModel, Field
from typing import Optional


class CostOptimizeQuery(BaseModel):
    target_concurrency: int = Field(..., ge=1)
    model_name: str
    input_tokens: int = Field(..., ge=1)
    output_tokens: int = Field(..., ge=1)
    max_ttft_ms: float = Field(3000.0, ge=0)
    min_throughput_per_user: float = Field(1.0, ge=0)
    top_k: int = Field(5, ge=1, le=20)


class CostRecommendation(BaseModel):
    rank: int
    gpu_name: str
    gpu_count: int
    price_per_hour: float
    max_concurrency: Optional[int]
    utilization_rate: Optional[float]
    cost_per_1m_tokens: Optional[float]
    source: str
    confidence: float
    warnings: list[str] = []


class CostOptimizeResponse(BaseModel):
    recommendations: list[CostRecommendation]
    query_params: dict
