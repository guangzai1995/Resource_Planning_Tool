"""
Tests for core API endpoints:
  /healthz
  GET /api/v1/gpus
  GET /api/v1/models
  POST /api/v1/predict
  GET /api/v1/cost/optimize
  GET /api/v1/data/coverage
"""
import pytest


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


# ── GPU metadata ──────────────────────────────────────────

def test_list_gpus_returns_seeded_data(client):
    resp = client.get("/api/v1/gpus")
    assert resp.status_code == 200
    gpus = resp.json()
    assert isinstance(gpus, list)
    assert len(gpus) >= 1
    # Check required fields
    for g in gpus:
        assert "name" in g
        assert "memory_gb" in g


def test_create_and_get_gpu(client):
    payload = {
        "name": "TestGPU",
        "memory_gb": 80,
        "memory_bandwidth_gbps": 2000,
        "tflops_bf16": 500,
        "price_per_hour": 10.0,
    }
    r = client.post("/api/v1/gpus", json=payload)
    assert r.status_code in (200, 201)
    created = r.json()
    assert created["name"] == "TestGPU"

    # verify it appears in list
    r2 = client.get("/api/v1/gpus")
    names = [g["name"] for g in r2.json()]
    assert "TestGPU" in names


def test_delete_gpu(client):
    # Create first
    payload = {"name": "TempGPU", "memory_gb": 40, "memory_bandwidth_gbps": 900, "tflops_bf16": 200, "price_per_hour": 5.0}
    r = client.post("/api/v1/gpus", json=payload)
    gpu_id = r.json()["id"]

    r_del = client.delete(f"/api/v1/gpus/{gpu_id}")
    assert r_del.status_code in (200, 204)


# ── Model metadata ────────────────────────────────────────

def test_list_models_returns_seeded_data(client):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    assert len(models) >= 1


def test_create_model(client):
    payload = {"name": "TestModel-7B", "parameter_b": 7.0, "model_type": "dense"}
    r = client.post("/api/v1/models", json=payload)
    assert r.status_code in (200, 201)
    assert r.json()["name"] == "TestModel-7B"


# ── Predict ───────────────────────────────────────────────

def test_predict_returns_response(client):
    """With no benchmark data, should fall back to model_based or unavailable."""
    gpus = client.get("/api/v1/gpus").json()
    models = client.get("/api/v1/models").json()

    if not gpus or not models:
        pytest.skip("No seeded GPU/model data")

    payload = {
        "gpu_name": gpus[0]["name"],
        "model_name": models[0]["name"],
        "gpu_count": 4,
        "input_tokens": 1024,
        "output_tokens": 256,
        "concurrency": 8,
        "max_ttft_ms": 3000,
    }
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "source" in data
    assert data["source"] in ("interpolation", "model_based", "ensemble", "unavailable")
    assert "confidence" in data
    assert "result" in data
    assert "warnings" in data


def test_predict_missing_fields(client):
    r = client.post("/api/v1/predict", json={})
    assert r.status_code == 422


# ── Cost optimize ─────────────────────────────────────────

def test_cost_optimize_returns_recommendations(client):
    models = client.get("/api/v1/models").json()
    if not models:
        pytest.skip("No seeded model data")

    params = {
        "model_name": models[0]["name"],
        "target_concurrency": 8,
        "input_tokens": 1024,
        "output_tokens": 256,
        "max_ttft_ms": 3000,
        "min_throughput_per_user": 1.0,
        "top_k": 3,
    }
    r = client.get("/api/v1/cost/optimize", params=params)
    assert r.status_code == 200
    data = r.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


# ── Data coverage ─────────────────────────────────────────

def test_data_coverage_empty_db(client):
    r = client.get("/api/v1/data/coverage")
    assert r.status_code == 200
    data = r.json()
    assert "total_rows" in data
    assert "items" in data
    assert data["total_rows"] == 0
