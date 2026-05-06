from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=["backend/.env", ".env"],   # 兼容从项目根目录或 backend/ 启动
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 数据库
    SQLITE_PATH: str = "./data/rpt.db"

    # 数据源
    EXCEL_DATA_PATH: str = "./资源规划工具.xlsx"
    BENCHMARK_TOOLS_DIR: str = "./benchmark_tools"
    MODEL_DIR: str = "./model"

    # 缓存 TTL（秒）
    PREDICTION_CACHE_TTL: int = 3600
    INTERP_CACHE_TTL: int = 1800
    META_CACHE_TTL: int = 600

    # vLLM 建模经验系数
    GPU_MEMORY_UTILIZATION: float = 0.9
    FRAMEWORK_OVERHEAD_GB: float = 1.5
    GPU_COMPUTE_UTILIZATION: float = 0.70
    BW_EFFICIENCY: float = 0.80
    SCHEDULING_OVERHEAD_MS: float = 5.0
    MAX_NUM_SEQS: int = 256

    @property
    def sqlite_url(self) -> str:
        path = Path(self.SQLITE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.resolve()}"

    @property
    def tp_efficiency_table(self) -> dict[int, float]:
        return {1: 1.0, 2: 0.92, 4: 0.88, 8: 0.82}


settings = Settings()
