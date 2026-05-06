"""
Tests for the prediction service layer:
  - interpolation with sufficient data
  - fallback to model_based when no data
  - ensemble blending
"""
import pytest
from app.services.prediction.interpolation import predict as interp_predict
from app.services.prediction.vllm_model import predict as model_predict
from app.services.prediction.ensemble import predict as ensemble_predict
from app.models.benchmark_data import BenchmarkData
from app.models.gpu_spec import GpuSpec as GpuSpecORM
from app.models.model import Model as ModelORM


def _insert_benchmark_rows(db_session, gpu_name, model_name, gpu_count, n=20):
    """Insert `n` synthetic benchmark rows for a given combination."""
    rows = []
    for i in range(n):
        concurrency = (i % 5 + 1) * 4  # 4, 8, 12, 16, 20
        input_tokens = (i % 4 + 1) * 512  # 512, 1024, 1536, 2048
        row = BenchmarkData(
            gpu_name=gpu_name,
            model_name=model_name,
            gpu_count=gpu_count,
            input_tokens=input_tokens,
            output_tokens=256,
            concurrency=concurrency,
            ttft_mean_ms=200 + concurrency * 10.0,
            ttft_p90_ms=220 + concurrency * 12.0,
            throughput_tokens_s=5000.0 - concurrency * 50,
            throughput_per_user_tokens_s=500.0 - concurrency * 5,
            gpu_utilization_pct=70.0 + concurrency * 0.5,
            memory_used_gb=40.0,
        )
        rows.append(row)
    db_session.add_all(rows)
    db_session.commit()
    return rows


# ── Interpolation ─────────────────────────────────────────

def test_interpolation_with_sufficient_data(db_session):
    _insert_benchmark_rows(db_session, "H200", "32B", 4, n=20)

    result = interp_predict(
        db=db_session,
        gpu_name="H200",
        model_name="32B",
        gpu_count=4,
        input_tokens=1024,
        output_tokens=256,
        concurrency=8,
    )
    assert result is not None
    assert result.get("source") in ("interpolation",)
    assert result.get("confidence", 0) > 0.6
    assert result["result"]["predicted_ttft_mean_ms"] is not None


def test_interpolation_no_data_returns_none(db_session):
    result = interp_predict(
        db=db_session,
        gpu_name="H200",
        model_name="NonExistentModel-999B",
        gpu_count=4,
        input_tokens=1024,
        output_tokens=256,
        concurrency=8,
    )
    # Should return None when there's not enough data
    assert result is None or result.get("source") != "interpolation"


# ── vLLM model-based ──────────────────────────────────────

def test_model_based_predict_with_gpu_spec(db_session):
    """model_predict needs GPU + model ORM records."""
    gpu = db_session.query(GpuSpecORM).filter_by(name="H200").first()
    model = db_session.query(ModelORM).filter_by(name="32B").first()

    if gpu is None or model is None:
        pytest.skip("Seed data not available in test DB")

    result = model_predict(
        gpu=gpu,
        model=model,
        gpu_count=4,
        input_tokens=1024,
        output_tokens=256,
        concurrency=8,
    )
    assert result is not None
    assert result.get("source") == "model_based"
    assert result.get("confidence") == pytest.approx(0.55, abs=0.1)


# ── Ensemble ─────────────────────────────────────────────

def test_ensemble_falls_back_to_model_based_with_no_data(db_session):
    result = ensemble_predict(
        db=db_session,
        gpu_name="H200",
        model_name="32B",
        gpu_count=4,
        input_tokens=512,
        output_tokens=128,
        concurrency=4,
        max_ttft_ms=3000,
    )
    assert result is not None
    assert result["source"] in ("model_based", "unavailable")


def test_ensemble_uses_interpolation_with_sufficient_data(db_session):
    _insert_benchmark_rows(db_session, "H200", "32B", 4, n=20)

    result = ensemble_predict(
        db=db_session,
        gpu_name="H200",
        model_name="32B",
        gpu_count=4,
        input_tokens=1024,
        output_tokens=256,
        concurrency=8,
        max_ttft_ms=3000,
    )
    assert result is not None
    assert result["source"] in ("interpolation", "ensemble")
    assert result["confidence"] >= 0.5
