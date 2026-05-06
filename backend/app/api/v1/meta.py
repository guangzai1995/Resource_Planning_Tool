from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import clear_prediction_cache
from app.schemas.benchmark import GpuSpecSchema, ModelSchema

router = APIRouter(tags=["meta"])


# ── GPU Specs ──────────────────────────────────────────────
@router.get("/gpus", response_model=list[GpuSpecSchema])
def list_gpus(db: Session = Depends(get_db)):
    from app.models.gpu_spec import GpuSpec
    return db.query(GpuSpec).order_by(GpuSpec.name).all()


@router.get("/gpus/{gpu_id}", response_model=GpuSpecSchema)
def get_gpu(gpu_id: int, db: Session = Depends(get_db)):
    from app.models.gpu_spec import GpuSpec
    gpu = db.query(GpuSpec).filter(GpuSpec.id == gpu_id).first()
    if not gpu:
        raise HTTPException(status_code=404, detail="GPU not found")
    return gpu


@router.post("/gpus", response_model=GpuSpecSchema)
def create_gpu(spec: GpuSpecSchema, db: Session = Depends(get_db)):
    from app.models.gpu_spec import GpuSpec

    existing = db.query(GpuSpec).filter_by(name=spec.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"GPU '{spec.name}' already exists")

    gpu = GpuSpec(**spec.model_dump(exclude={"id"}))
    db.add(gpu)
    db.commit()
    db.refresh(gpu)
    clear_prediction_cache()
    return gpu


@router.put("/gpus/{gpu_id}", response_model=GpuSpecSchema)
def update_gpu(gpu_id: int, spec: GpuSpecSchema, db: Session = Depends(get_db)):
    from app.models.gpu_spec import GpuSpec

    gpu = db.query(GpuSpec).filter(GpuSpec.id == gpu_id).first()
    if not gpu:
        raise HTTPException(status_code=404, detail="GPU not found")
    for k, v in spec.model_dump(exclude={"id"}, exclude_none=True).items():
        setattr(gpu, k, v)
    db.commit()
    db.refresh(gpu)
    clear_prediction_cache()
    return gpu


@router.delete("/gpus/{gpu_id}")
def delete_gpu(gpu_id: int, db: Session = Depends(get_db)):
    from app.models.gpu_spec import GpuSpec

    gpu = db.query(GpuSpec).filter(GpuSpec.id == gpu_id).first()
    if not gpu:
        raise HTTPException(status_code=404, detail="GPU not found")
    db.delete(gpu)
    db.commit()
    return {"deleted": gpu_id}


# ── Models ─────────────────────────────────────────────────
@router.get("/models", response_model=list[ModelSchema])
def list_models(db: Session = Depends(get_db)):
    from app.models.model import Model
    return db.query(Model).order_by(Model.parameter_b).all()


@router.get("/models/{model_id}", response_model=ModelSchema)
def get_model(model_id: int, db: Session = Depends(get_db)):
    from app.models.model import Model
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/models", response_model=ModelSchema)
def create_model(schema: ModelSchema, db: Session = Depends(get_db)):
    from app.models.model import Model

    existing = db.query(Model).filter_by(name=schema.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Model '{schema.name}' already exists")

    model = Model(**schema.model_dump(exclude={"id"}))
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.put("/models/{model_id}", response_model=ModelSchema)
def update_model(model_id: int, schema: ModelSchema, db: Session = Depends(get_db)):
    from app.models.model import Model

    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    for k, v in schema.model_dump(exclude={"id"}, exclude_none=True).items():
        setattr(model, k, v)
    db.commit()
    db.refresh(model)
    return model


@router.delete("/models/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    from app.models.model import Model

    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    db.delete(model)
    db.commit()
    return {"deleted": model_id}
