import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.benchmark import BenchmarkConfig, BenchmarkTaskStatus, BenchmarkPointResult
from app.tasks.benchmark_runner import run_benchmark, get_task_status, get_task_queue, get_task_results, cancel_task

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.post("/run")
def submit_benchmark(
    config: BenchmarkConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """提交压测任务"""
    from app.models.benchmark_run import BenchmarkRun

    task_id = str(uuid.uuid4())
    run = BenchmarkRun(
        run_uuid=task_id,
        gpu_name=config.gpu_name,
        model_name=config.model_name,
        gpu_count=config.gpu_count,
        benchmark_args=config.model_dump_json(),
        status="pending",
        source="benchmark",
        created_at=datetime.now().isoformat(),
    )
    db.add(run)
    db.commit()

    background_tasks.add_task(run_benchmark, task_id, config)
    return {"task_id": task_id, "status": "pending"}


@router.get("/{task_id}/status", response_model=BenchmarkTaskStatus)
def get_status(task_id: str, db: Session = Depends(get_db)):
    from app.models.benchmark_run import BenchmarkRun

    run = db.query(BenchmarkRun).filter_by(run_uuid=task_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Task not found")

    # 优先用内存状态（更实时）
    mem_status = get_task_status(task_id)
    status = mem_status if mem_status not in ("unknown",) else run.status

    return BenchmarkTaskStatus(
        task_id=task_id,
        status=status,
        gpu_name=run.gpu_name,
        model_name=run.model_name,
        gpu_count=run.gpu_count,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_message=run.error_message,
    )


@router.get("/list")
def list_tasks(db: Session = Depends(get_db), limit: int = 20):
    from app.models.benchmark_run import BenchmarkRun

    runs = (
        db.query(BenchmarkRun)
        .filter(BenchmarkRun.source == "benchmark")
        .order_by(BenchmarkRun.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "task_id": r.run_uuid,
            "status": get_task_status(r.run_uuid) if get_task_status(r.run_uuid) != "unknown" else r.status,
            "gpu_name": r.gpu_name,
            "model_name": r.model_name,
            "gpu_count": r.gpu_count,
            "created_at": r.created_at,
            "finished_at": r.finished_at,
        }
        for r in runs
    ]


@router.websocket("/{task_id}/stream")
async def stream_log(websocket: WebSocket, task_id: str):
    """WebSocket 实时推流压测日志"""
    await websocket.accept()
    queue = get_task_queue(task_id)

    if queue is None:
        # 历史任务，直接返回状态
        from app.core.database import db_session
        from app.models.benchmark_run import BenchmarkRun

        with db_session() as db:
            run = db.query(BenchmarkRun).filter_by(run_uuid=task_id).first()
            status = run.status if run else "not_found"
        await websocket.send_json({"type": "end", "status": status})
        await websocket.close()
        return

    try:
        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            if line is None:  # 结束信号
                status = get_task_status(task_id)
                await websocket.send_json({"type": "end", "status": status})
                break
            await websocket.send_json({"type": "log", "content": line})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@router.post("/{task_id}/cancel")
def cancel_benchmark(task_id: str, db: Session = Depends(get_db)):
    """请求停止正在运行的压测任务"""
    from app.models.benchmark_run import BenchmarkRun

    run = db.query(BenchmarkRun).filter_by(run_uuid=task_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Task not found")
    ok = cancel_task(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail=f"Task is not running (status={run.status})")
    return {"task_id": task_id, "status": "cancelling"}


@router.get("/{task_id}/results", response_model=list[BenchmarkPointResult])
def get_results(task_id: str, db: Session = Depends(get_db)):
    """获取已完成任务的详细测试结果（内存缓存，服务重启后丢失）"""
    from app.models.benchmark_run import BenchmarkRun

    run = db.query(BenchmarkRun).filter_by(run_uuid=task_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Task not found")

    rows = get_task_results(task_id)
    return [BenchmarkPointResult(**r) for r in rows]
