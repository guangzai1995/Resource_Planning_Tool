from sqlalchemy import Column, Integer, Text, Float, UniqueConstraint, Index
from app.core.database import Base


class BenchmarkData(Base):
    __tablename__ = "benchmark_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer)
    gpu_name = Column(Text, nullable=False)
    model_name = Column(Text, nullable=False)
    gpu_count = Column(Integer, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    concurrency = Column(Integer, nullable=False)
    throughput_tokens_s = Column(Float)            # system-level tokens/s
    throughput_per_user_tokens_s = Column(Float)   # per-user tokens/s
    ttft_mean_ms = Column(Float)
    ttft_p90_ms = Column(Float)
    ttft_p99_ms = Column(Float)
    ttft_max_ms = Column(Float)
    decode_latency_mean_ms = Column(Float)
    decode_latency_p90_ms = Column(Float)
    decode_latency_p99_ms = Column(Float)
    decode_latency_max_ms = Column(Float)
    gpu_utilization_pct = Column(Float)
    memory_used_gb = Column(Float)
    recorded_at = Column(Text, server_default="(datetime('now'))")

    __table_args__ = (
        UniqueConstraint(
            "gpu_name", "model_name", "gpu_count",
            "input_tokens", "output_tokens", "concurrency",
            name="uq_benchmark_data",
        ),
        Index("idx_bdata_lookup", "gpu_name", "model_name", "gpu_count",
              "input_tokens", "concurrency"),
    )
