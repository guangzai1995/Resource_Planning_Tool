from fastapi import APIRouter

from app.api.v1 import predict, cost, benchmark, data, meta

router = APIRouter(prefix="/api/v1")

router.include_router(predict.router)
router.include_router(cost.router)
router.include_router(benchmark.router)
router.include_router(data.router)
router.include_router(meta.router)
