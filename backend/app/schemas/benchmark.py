from pydantic import BaseModel, Field
from typing import Literal, Optional


class BenchmarkConfig(BaseModel):
    """直接测试已运行的 OpenAI 兼容 API，无需拉起 vLLM 服务"""

    api_base_url: str = Field(..., examples=["http://0.0.0.0:9999/v1"])
    api_key: str = Field("token-abc", description="Bearer 鉴权 token")
    backend_type: Literal["openai", "openai-chat"] = Field(
        "openai-chat",
        description="openai → /v1/completions；openai-chat → /v1/chat/completions",
    )

    # 仅用于存储/显示，不影响请求逻辑
    model_name: str = Field(..., description="模型名称（与数据库中一致）")
    gpu_name: str = Field(..., description="GPU 型号（与数据库中一致）")
    gpu_count: int = Field(1, ge=1, description="GPU 卡数")

    input_tokens_list: list[int] = Field(
        default=[512, 2048], description="输入 token 数列表，每个值独立测试"
    )
    output_tokens: int = Field(512, ge=1, description="输出 token 数")
    concurrency_list: list[int] = Field(
        default=[1, 4, 8, 16, 32], description="并发数列表"
    )
    epochs: int = Field(
        5, ge=1,
        description="每个（输入tokens, 并发数）测试点的轮数。每轮同时发 concurrency 个请求，"
                    "总请求数 = epochs × concurrency。轮数越多统计越稳定",
    )
    max_ttft_ms: Optional[float] = Field(
        None, ge=0,
        description="首 token 延时上限（ms，均值）。超出则跳过该输入长度下更高并发测试。None 表示不限制",
    )
    min_throughput_per_user: Optional[float] = Field(
        None, ge=0,
        description="单用户吞吐下限（tokens/s）。低于则跳过该输入长度下更高并发测试。None 表示不限制",
    )
    tokenizer_path: Optional[str] = Field(
        None,
        description="分词器路径或 modelscope/huggingface 模型名（用于精确计算 token 数）。"
                    "留空则自动尝试 model_name，失败则回退到 usage.completion_tokens 或 SSE chunk 计数",
    )


class BenchmarkTaskStatus(BaseModel):
    task_id: str
    status: str  # pending | running | done | failed
    gpu_name: str
    model_name: str
    gpu_count: int
    created_at: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    error_message: Optional[str]


class BenchmarkPointResult(BaseModel):
    input_tokens: int
    output_tokens: int
    concurrency: int
    throughput_tokens_s: Optional[float] = None
    throughput_per_user_tokens_s: Optional[float] = None
    ttft_mean_ms: Optional[float] = None
    ttft_p90_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    ttft_max_ms: Optional[float] = None
    decode_latency_mean_ms: Optional[float] = None
    decode_latency_p90_ms: Optional[float] = None
    decode_latency_p99_ms: Optional[float] = None
    decode_latency_max_ms: Optional[float] = None


class GpuSpecSchema(BaseModel):
    id: Optional[int] = None
    name: str
    vendor: Optional[str] = None
    memory_gb: float
    memory_bandwidth_gbps: Optional[float] = None
    tflops_bf16: Optional[float] = None
    price_per_hour: float = 0.0
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class ModelSchema(BaseModel):
    id: Optional[int] = None
    name: str
    parameter_b: float
    model_type: Optional[str] = "dense"
    default_model_path: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True
