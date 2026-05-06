"""
Excel 导入服务：解析 资源规划工具.xlsx → SQLite
Sheet 命名规则：{model_size}-{GPU}-{gpu_count}测试数据
"""
import re
from pathlib import Path

import openpyxl
import pandas as pd
from sqlalchemy.orm import Session

from app.core.logging import get_logger

logger = get_logger(__name__)

SHEET_PATTERN = re.compile(
    r"^(\d+B(?:-A\d+B)?)"   # model_size: 4B / 235B-A22B / 671B
    r"-(H200|H20|P800)"      # gpu_name
    r"-(\d+)"                # gpu_count
    r"\s*测试\s*数据$"        # 后缀
)

COLUMN_MAPPING = {
    "输入长度": "input_tokens",
    "输出长度": "output_tokens",
    "并发数": "concurrency",
    "输出tokens总吞吐": "throughput_tokens_s",
    "首tokens时延TP90（ms）": "ttft_p90_ms",
    "首tokens时延TP99（ms）": "ttft_p99_ms",
    "最大首tokens时延（ms）": "ttft_max_ms",
    "平均首tokens时延（ms）": "ttft_mean_ms",
    "增量时延TP90（ms）": "decode_latency_p90_ms",
    "增量时延TP99（ms）": "decode_latency_p99_ms",
    "最大增量时延（ms）": "decode_latency_max_ms",
    "平均增量时延（ms）": "decode_latency_mean_ms",
}

# 列名别名（宽松匹配）
COLUMN_ALIASES: dict[str, str] = {
    "输入长度": "input_tokens",
    "输出长度": "output_tokens",
    "并发数": "concurrency",
    "输出tokens总吞吐": "throughput_tokens_s",
    "首tokens时延tp90": "ttft_p90_ms",
    "首tokens时延tp99": "ttft_p99_ms",
    "最大首tokens时延": "ttft_max_ms",
    "平均首tokens时延": "ttft_mean_ms",
    "增量时延tp90": "decode_latency_p90_ms",
    "增量时延tp99": "decode_latency_p99_ms",
    "最大增量时延": "decode_latency_max_ms",
    "平均增量时延": "decode_latency_mean_ms",
}


def _normalize_col(col: str) -> str:
    """标准化列名：去掉括号、单位、大小写"""
    col = str(col).strip().lower()
    col = re.sub(r"[（(][^）)]*[）)]", "", col)  # 去掉括号内容
    col = col.strip()
    return col


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """尝试将 DataFrame 列名映射到标准字段名"""
    rename = {}
    for raw_col in df.columns:
        norm = _normalize_col(str(raw_col))
        for alias, standard in COLUMN_ALIASES.items():
            if norm.startswith(alias) or alias in norm:
                rename[raw_col] = standard
                break
        # 精确匹配
        for exact, standard in COLUMN_MAPPING.items():
            if str(raw_col).strip() == exact:
                rename[raw_col] = standard
                break
    return df.rename(columns=rename)


def import_excel(excel_path: Path, db: Session) -> dict:
    """解析 Excel 所有测试 Sheet，批量写入 SQLite（UPSERT）"""
    from app.models.benchmark_data import BenchmarkData
    from app.models.benchmark_run import BenchmarkRun
    import uuid
    from datetime import datetime

    stats: dict = {"sheets": 0, "rows": 0, "skipped": []}

    wb = openpyxl.load_workbook(excel_path, data_only=True, read_only=True)

    for sheet_name in wb.sheetnames:
        m = SHEET_PATTERN.match(sheet_name)
        if not m:
            stats["skipped"].append(sheet_name)
            logger.info("skip_sheet", sheet=sheet_name)
            continue

        model_name = m.group(1)
        gpu_name = m.group(2)
        gpu_count = int(m.group(3))

        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        except Exception as e:
            logger.warning("read_sheet_error", sheet=sheet_name, error=str(e))
            stats["skipped"].append(sheet_name)
            continue

        # 只保留前 12 列有效数据
        df = df.iloc[:, :12]

        # 映射列名
        df = _map_columns(df)

        # 过滤空行
        required = ["input_tokens", "concurrency"]
        missing_req = [c for c in required if c not in df.columns]
        if missing_req:
            logger.warning("missing_columns", sheet=sheet_name, missing=missing_req)
            stats["skipped"].append(sheet_name)
            continue

        df = df.dropna(subset=["input_tokens", "concurrency"])
        if df.empty:
            stats["skipped"].append(sheet_name)
            continue

        # 类型转换
        try:
            df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0).astype(int)
            df["output_tokens"] = pd.to_numeric(df.get("output_tokens", 256), errors="coerce").fillna(256).astype(int)
            df["concurrency"] = pd.to_numeric(df["concurrency"], errors="coerce").fillna(0).astype(int)
            df = df[df["input_tokens"] > 0]
            df = df[df["concurrency"] > 0]
        except Exception as e:
            logger.warning("type_convert_error", sheet=sheet_name, error=str(e))
            stats["skipped"].append(sheet_name)
            continue

        df["gpu_name"] = gpu_name
        df["model_name"] = model_name
        df["gpu_count"] = gpu_count

        # 创建导入批次记录
        run_uuid = str(uuid.uuid4())
        run = BenchmarkRun(
            run_uuid=run_uuid,
            gpu_name=gpu_name,
            model_name=model_name,
            gpu_count=gpu_count,
            status="done",
            source="excel_import",
            started_at=datetime.now().isoformat(),
            finished_at=datetime.now().isoformat(),
        )
        db.add(run)
        db.flush()

        # UPSERT 数据行
        numeric_cols = [
            "throughput_tokens_s", "ttft_p90_ms", "ttft_p99_ms", "ttft_max_ms",
            "ttft_mean_ms", "decode_latency_p90_ms", "decode_latency_p99_ms",
            "decode_latency_max_ms", "decode_latency_mean_ms",
        ]
        inserted = 0
        for _, row in df.iterrows():
            # 检查是否已存在（UPSERT 逻辑）
            existing = db.query(BenchmarkData).filter_by(
                gpu_name=row["gpu_name"],
                model_name=row["model_name"],
                gpu_count=int(row["gpu_count"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                concurrency=int(row["concurrency"]),
            ).first()

            data_kwargs = {
                "run_id": run.id,
                "gpu_name": str(row["gpu_name"]),
                "model_name": str(row["model_name"]),
                "gpu_count": int(row["gpu_count"]),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "concurrency": int(row["concurrency"]),
            }
            for col in numeric_cols:
                if col in df.columns and col in row:
                    val = row.get(col)
                    try:
                        data_kwargs[col] = float(val) if val is not None and str(val) != "nan" else None
                    except (ValueError, TypeError):
                        data_kwargs[col] = None

            if existing:
                for k, v in data_kwargs.items():
                    setattr(existing, k, v)
            else:
                db.add(BenchmarkData(**data_kwargs))
                inserted += 1

        db.flush()
        stats["sheets"] += 1
        stats["rows"] += inserted
        logger.info(
            "sheet_imported",
            sheet=sheet_name,
            gpu=gpu_name,
            model=model_name,
            gpu_count=gpu_count,
            rows=inserted,
        )

    db.commit()
    return stats
