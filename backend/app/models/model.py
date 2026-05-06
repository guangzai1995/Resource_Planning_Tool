from sqlalchemy import Column, Integer, Text, Float
from app.core.database import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, unique=True)
    parameter_b = Column(Float, nullable=False)          # renamed from size_b
    model_type = Column(Text, default="dense")           # new: dense | moe
    default_model_path = Column(Text)                    # new
    is_moe = Column(Integer, default=0)                  # kept for vLLM calc
    num_layers = Column(Integer)
    hidden_size = Column(Integer)
    num_kv_heads = Column(Integer)
    head_size = Column(Integer)
    quantization = Column(Text)
    notes = Column(Text)
    created_at = Column(Text, server_default="(datetime('now'))")
