from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    gpu_name: str = Field(..., examples=["H200"])
    model_name: str = Field(..., examples=["72B"])
    gpu_count: int = Field(..., ge=1, le=32)
    input_tokens: int = Field(..., ge=1, le=128000)
    output_tokens: int = Field(..., ge=1, le=8192)
    concurrency: int = Field(..., ge=1, le=2048)
    max_ttft_ms: Optional[float] = Field(None, ge=0)
    min_throughput_per_user: Optional[float] = Field(None, ge=0)


class PredictResult(BaseModel):
    predicted_ttft_mean_ms: Optional[float]
    predicted_ttft_p90_ms: Optional[float]
    predicted_throughput_tokens_s: Optional[float]
    max_safe_concurrency: Optional[int]
    recommended_concurrency: Optional[int]


class PredictResponse(BaseModel):
    source: str  # interpolation | model_based | ensemble
    confidence: float
    data_points_used: int
    result: PredictResult
    warnings: list[str] = []
    metadata: dict = {}
