from sqlalchemy import Column, Integer, Text
from app.core.database import Base


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_uuid = Column(Text, nullable=False, unique=True)
    gpu_name = Column(Text, nullable=False)
    model_name = Column(Text, nullable=False)
    gpu_count = Column(Integer, nullable=False)
    vllm_args = Column(Text)
    benchmark_args = Column(Text)
    status = Column(Text, default="pending")
    source = Column(Text, default="benchmark")
    started_at = Column(Text)
    finished_at = Column(Text)
    error_message = Column(Text)
    created_at = Column(Text, server_default="(datetime('now'))")
