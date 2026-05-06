import io
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import clear_prediction_cache
from app.core.logging import get_logger

router = APIRouter(prefix="/data", tags=["data"])
logger = get_logger(__name__)


@router.post("/import")
async def import_data(
    file: UploadFile = File(...),
    gpu_name: str = "",
    model_name: str = "",
    gpu_count: int = 1,
    db: Session = Depends(get_db),
):
    """上传 Excel 或 CSV 批量导入数据"""
    filename = file.filename or ""
    content = await file.read()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        # Excel 导入
        import tempfile, os
        from app.services.excel_importer import import_excel

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            stats = import_excel(Path(tmp_path), db)
        finally:
            os.unlink(tmp_path)
        clear_prediction_cache()
        return {"imported": stats["rows"], "sheets": stats["sheets"], "skipped": stats["skipped"]}

    elif filename.endswith(".csv"):
        # CSV 导入
        import pandas as pd
        from app.services.excel_importer import _map_columns
        from app.models.benchmark_data import BenchmarkData
        from app.models.benchmark_run import BenchmarkRun
        import uuid
        from datetime import datetime

        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        df = _map_columns(df)
        df["gpu_name"] = gpu_name
        df["model_name"] = model_name
        df["gpu_count"] = gpu_count

        run_uuid = str(uuid.uuid4())
        run = BenchmarkRun(
            run_uuid=run_uuid,
            gpu_name=gpu_name,
            model_name=model_name,
            gpu_count=gpu_count,
            status="done",
            source="csv_import",
            started_at=datetime.now().isoformat(),
            finished_at=datetime.now().isoformat(),
        )
        db.add(run)
        db.flush()

        inserted = 0
        for _, row in df.iterrows():
            try:
                bd = BenchmarkData(
                    run_id=run.id,
                    gpu_name=str(row.get("gpu_name", gpu_name)),
                    model_name=str(row.get("model_name", model_name)),
                    gpu_count=int(row.get("gpu_count", gpu_count)),
                    input_tokens=int(row.get("input_tokens", 0)),
                    output_tokens=int(row.get("output_tokens", 256)),
                    concurrency=int(row.get("concurrency", 1)),
                )
                for field in ["throughput_tokens_s", "ttft_mean_ms", "ttft_p90_ms",
                              "ttft_p99_ms", "ttft_max_ms", "decode_latency_mean_ms",
                              "decode_latency_p90_ms", "decode_latency_p99_ms",
                              "decode_latency_max_ms"]:
                    val = row.get(field)
                    setattr(bd, field, float(val) if val is not None and str(val) != "nan" else None)
                db.merge(bd)
                inserted += 1
            except Exception:
                pass
        db.commit()
        clear_prediction_cache()
        return {"imported": inserted, "message": "CSV 导入完成"}

    raise HTTPException(status_code=400, detail="仅支持 .xlsx 或 .csv 文件")


@router.post("/reimport")
def reimport_excel(db: Session = Depends(get_db)):
    """清空并重新从 Excel 导入数据"""
    from app.core.config import settings
    from app.services.excel_importer import import_excel
    from app.models.benchmark_data import BenchmarkData

    db.query(BenchmarkData).delete()
    db.commit()
    clear_prediction_cache()

    excel_path = Path(settings.EXCEL_DATA_PATH)
    if not excel_path.exists():
        raise HTTPException(status_code=404, detail=f"Excel 文件不存在: {excel_path}")

    stats = import_excel(excel_path, db)
    return {"imported": stats["rows"], "sheets": stats["sheets"], "skipped": stats["skipped"]}


@router.get("/coverage")
def get_coverage(db: Session = Depends(get_db)):
    """返回数据覆盖热力图数据"""
    from app.models.benchmark_data import BenchmarkData

    rows = (
        db.query(
            BenchmarkData.gpu_name,
            BenchmarkData.model_name,
            BenchmarkData.gpu_count,
            func.count(BenchmarkData.id).label("data_count"),
            func.min(BenchmarkData.input_tokens).label("min_input"),
            func.max(BenchmarkData.input_tokens).label("max_input"),
            func.min(BenchmarkData.concurrency).label("min_concurrency"),
            func.max(BenchmarkData.concurrency).label("max_concurrency"),
        )
        .group_by(BenchmarkData.gpu_name, BenchmarkData.model_name, BenchmarkData.gpu_count)
        .all()
    )

    items = [
        {
            "gpu_name": r.gpu_name,
            "model_name": r.model_name,
            "gpu_count": r.gpu_count,
            "data_count": r.data_count,
            "min_input": r.min_input,
            "max_input": r.max_input,
            "min_concurrency": r.min_concurrency,
            "max_concurrency": r.max_concurrency,
        }
        for r in rows
    ]
    total_rows = sum(i["data_count"] for i in items)
    return {"total_rows": total_rows, "items": items}


@router.get("/options")
def get_options(db: Session = Depends(get_db)):
    """返回联动下拉框所需的可用组合及各参数离散值（用于前端级联选择）"""
    from app.models.benchmark_data import BenchmarkData

    rows = (
        db.query(
            BenchmarkData.gpu_name,
            BenchmarkData.model_name,
            BenchmarkData.gpu_count,
            BenchmarkData.input_tokens,
            BenchmarkData.output_tokens,
        )
        .distinct()
        .order_by(
            BenchmarkData.gpu_name,
            BenchmarkData.model_name,
            BenchmarkData.gpu_count,
            BenchmarkData.input_tokens,
        )
        .all()
    )

    combos: dict = {}
    for r in rows:
        key = (r.gpu_name, r.model_name, r.gpu_count)
        if key not in combos:
            combos[key] = {"input_tokens": set(), "output_tokens": set()}
        combos[key]["input_tokens"].add(r.input_tokens)
        combos[key]["output_tokens"].add(r.output_tokens)

    return [
        {
            "gpu_name": k[0],
            "model_name": k[1],
            "gpu_count": k[2],
            "input_tokens": sorted(v["input_tokens"]),
            "output_tokens": sorted(v["output_tokens"]),
        }
        for k, v in combos.items()
    ]


@router.get("/detail")
def get_detail(
    gpu_name: str,
    model_name: str,
    gpu_count: int,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
):
    """获取特定组合的详细数据"""
    from app.models.benchmark_data import BenchmarkData

    total = (
        db.query(func.count(BenchmarkData.id))
        .filter_by(gpu_name=gpu_name, model_name=model_name, gpu_count=gpu_count)
        .scalar()
    )
    rows = (
        db.query(BenchmarkData)
        .filter_by(gpu_name=gpu_name, model_name=model_name, gpu_count=gpu_count)
        .order_by(BenchmarkData.input_tokens, BenchmarkData.concurrency)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": [
            {
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "concurrency": r.concurrency,
                "throughput_tokens_s": r.throughput_tokens_s,
                "ttft_mean_ms": r.ttft_mean_ms,
                "ttft_p90_ms": r.ttft_p90_ms,
                "decode_latency_mean_ms": r.decode_latency_mean_ms,
            }
            for r in rows
        ],
    }


# ── 数据记录 CRUD ──────────────────────────────────────────────

class _RecordUpdate(BaseModel):
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


def _row_to_dict(r) -> dict:
    return {
        "id": r.id,
        "gpu_name": r.gpu_name,
        "model_name": r.model_name,
        "gpu_count": r.gpu_count,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "concurrency": r.concurrency,
        "throughput_tokens_s": r.throughput_tokens_s,
        "throughput_per_user_tokens_s": r.throughput_per_user_tokens_s,
        "ttft_mean_ms": r.ttft_mean_ms,
        "ttft_p90_ms": r.ttft_p90_ms,
        "ttft_p99_ms": r.ttft_p99_ms,
        "ttft_max_ms": r.ttft_max_ms,
        "decode_latency_mean_ms": r.decode_latency_mean_ms,
        "decode_latency_p90_ms": r.decode_latency_p90_ms,
        "decode_latency_p99_ms": r.decode_latency_p99_ms,
        "decode_latency_max_ms": r.decode_latency_max_ms,
        "recorded_at": r.recorded_at,
    }


@router.get("/records")
def list_records(
    gpu_name: Optional[str] = None,
    model_name: Optional[str] = None,
    gpu_count: Optional[int] = None,
    input_tokens: Optional[int] = None,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
):
    """分页列出数据记录，支持按 GPU/模型/卡数/输入长度过滤"""
    from app.models.benchmark_data import BenchmarkData

    q = db.query(BenchmarkData)
    if gpu_name:     q = q.filter(BenchmarkData.gpu_name == gpu_name)
    if model_name:   q = q.filter(BenchmarkData.model_name == model_name)
    if gpu_count:    q = q.filter(BenchmarkData.gpu_count == gpu_count)
    if input_tokens: q = q.filter(BenchmarkData.input_tokens == input_tokens)

    total = q.count()
    rows = (
        q.order_by(
            BenchmarkData.gpu_name, BenchmarkData.model_name,
            BenchmarkData.gpu_count, BenchmarkData.input_tokens,
            BenchmarkData.concurrency,
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {"total": total, "page": page, "page_size": page_size, "items": [_row_to_dict(r) for r in rows]}


@router.put("/records/{record_id}")
def update_record(record_id: int, body: _RecordUpdate, db: Session = Depends(get_db)):
    """更新单条数据记录的性能指标"""
    from app.models.benchmark_data import BenchmarkData

    rec = db.query(BenchmarkData).filter_by(id=record_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found")
    for field, val in body.model_dump(exclude_none=True).items():
        setattr(rec, field, val)
    db.commit()
    clear_prediction_cache()
    return _row_to_dict(rec)


@router.delete("/records")
def delete_records(
    ids: List[int] = Query(..., description="要删除的记录 ID 列表"),
    db: Session = Depends(get_db),
):
    """批量删除数据记录（支持单条或多条）"""
    from app.models.benchmark_data import BenchmarkData

    deleted = db.query(BenchmarkData).filter(BenchmarkData.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    clear_prediction_cache()
    return {"deleted": deleted}
