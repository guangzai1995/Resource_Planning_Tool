from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Resource Planning Tool API",
    version="2.1.0",
    description="LLM 推理资源规划工具 - 内部使用",
)

# ── 中间件 ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=elapsed,
    )
    return response


# ── 路由 ────────────────────────────────────────────────────
from app.api.v1.router import router as v1_router  # noqa: E402
app.include_router(v1_router)


# ── 健康检查 ─────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": "2.1.0"}


# ── 启动事件 ─────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    from app.core.database import create_all_tables, db_session
    from app.models.benchmark_data import BenchmarkData
    from app.seed_data import seed_initial_data
    from app.services.excel_importer import import_excel

    # 1. 创建表
    create_all_tables()
    logger.info("database_tables_created")

    with db_session() as db:
        # 2. 填充初始 GPU/模型规格
        seed_initial_data(db)
        logger.info("seed_data_loaded")

        # 3. 若数据库为空，自动导入 Excel
        count = db.query(BenchmarkData).count()
        if count == 0:
            excel_path = Path(settings.EXCEL_DATA_PATH)
            if excel_path.exists():
                logger.info("auto_importing_excel", path=str(excel_path))
                stats = import_excel(excel_path, db)
                logger.info(
                    "excel_imported",
                    sheets=stats["sheets"],
                    rows=stats["rows"],
                    skipped=stats["skipped"],
                )
            else:
                logger.warning("excel_not_found", path=str(excel_path))
        else:
            logger.info("database_has_data", count=count)


# ── 前端静态文件（生产构建后）──────────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    # 只挂载 /assets 目录，避免 StaticFiles 拦截所有路由
    _assets_dir = _frontend_dist / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")
    logger.info("frontend_static_mounted", path=str(_frontend_dist))


# ── SPA 兜底路由（必须在所有 API 路由之后注册）──────────────────
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """
    对所有非 API 路径：
    1. 若是真实存在的静态文件（favicon、robots.txt 等）直接返回
    2. 否则返回 index.html，让前端 Router 处理
    """
    if _frontend_dist.exists():
        file_path = _frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        index = _frontend_dist / "index.html"
        if index.exists():
            return FileResponse(str(index))
    return JSONResponse({"error": "not found"}, status_code=404)
