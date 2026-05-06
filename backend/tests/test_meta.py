"""
Tests for metadata CRUD endpoints: /api/v1/gpus, /api/v1/models
"""
import pytest


# ── GPU CRUD ──────────────────────────────────────────────

def test_create_gpu(client):
    r = client.post("/api/v1/gpus", json={
        "name": "A100",
        "memory_gb": 80,
        "memory_bandwidth_gbps": 2000,
        "tflops_bf16": 312,
        "price_per_hour": 25.0,
    })
    assert r.status_code in (200, 201)
    body = r.json()
    assert body["name"] == "A100"
    assert body["id"] is not None


def test_get_gpu_by_id(client):
    # create
    r = client.post("/api/v1/gpus", json={
        "name": "A100X",
        "memory_gb": 80,
        "memory_bandwidth_gbps": 2000,
        "tflops_bf16": 312,
        "price_per_hour": 25.0,
    })
    gpu_id = r.json()["id"]

    r2 = client.get(f"/api/v1/gpus/{gpu_id}")
    assert r2.status_code == 200
    assert r2.json()["name"] == "A100X"


def test_update_gpu(client):
    r = client.post("/api/v1/gpus", json={
        "name": "GPUToUpdate",
        "memory_gb": 40,
        "memory_bandwidth_gbps": 1000,
        "tflops_bf16": 150,
        "price_per_hour": 8.0,
    })
    gpu_id = r.json()["id"]

    r2 = client.put(f"/api/v1/gpus/{gpu_id}", json={"price_per_hour": 12.0})
    assert r2.status_code == 200
    assert r2.json()["price_per_hour"] == 12.0


def test_delete_gpu_not_found(client):
    r = client.delete("/api/v1/gpus/99999")
    assert r.status_code == 404


def test_create_gpu_duplicate_name(client):
    payload = {
        "name": "DuplicateGPU",
        "memory_gb": 40,
        "memory_bandwidth_gbps": 900,
        "tflops_bf16": 100,
        "price_per_hour": 5.0,
    }
    r1 = client.post("/api/v1/gpus", json=payload)
    r2 = client.post("/api/v1/gpus", json=payload)
    # Second creation should fail with 409
    assert r1.status_code in (200, 201)
    assert r2.status_code == 409


# ── Model CRUD ────────────────────────────────────────────

def test_create_model(client):
    r = client.post("/api/v1/models", json={
        "name": "TestLLM-14B",
        "parameter_b": 14.0,
        "model_type": "dense",
    })
    assert r.status_code in (200, 201)
    assert r.json()["name"] == "TestLLM-14B"


def test_update_model(client):
    r = client.post("/api/v1/models", json={
        "name": "ModelToUpdate",
        "parameter_b": 7.0,
    })
    model_id = r.json()["id"]

    r2 = client.put(f"/api/v1/models/{model_id}", json={"default_model_path": "/model/updated"})
    assert r2.status_code == 200
    assert r2.json()["default_model_path"] == "/model/updated"


def test_delete_model(client):
    r = client.post("/api/v1/models", json={"name": "DeleteMe", "parameter_b": 3.0})
    model_id = r.json()["id"]

    r_del = client.delete(f"/api/v1/models/{model_id}")
    assert r_del.status_code in (200, 204)

    r2 = client.get(f"/api/v1/models/{model_id}")
    assert r2.status_code == 404
