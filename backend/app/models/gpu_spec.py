from sqlalchemy import Column, Integer, Text, Float
from app.core.database import Base


class GpuSpec(Base):
    __tablename__ = "gpu_specs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, unique=True)
    vendor = Column(Text)
    memory_gb = Column(Float, nullable=False)
    memory_bandwidth_gbps = Column(Float)
    tflops_bf16 = Column(Float)
    price_per_hour = Column(Float, nullable=False, default=0.0)
    notes = Column(Text)
    created_at = Column(Text, server_default="(datetime('now'))")
