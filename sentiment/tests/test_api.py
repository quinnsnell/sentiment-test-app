"""Unit tests for the sentiment app's HTTP surface.

These tests do NOT hit the classroom LLM (which needs VPN + a live
LiteLLM) or load the real HF pipeline. Real integration is checked at
deploy time via `/health` — Coolify polls it, and a failure marks the
deploy unhealthy.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import schemas
from main import app


@pytest.fixture
def client():
    """FastAPI TestClient — lifespan runs but SKIP_LOCAL_MODEL=1 (from
    conftest.py) skips the real HF load."""
    with TestClient(app) as c:
        yield c


def test_ready_returns_ok(client):
    """/ready is the cheap liveness probe — no dependencies exercised."""
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert "version" in body


def test_gpu_endpoint_reports_state(client):
    """/gpu returns a valid JSON shape whether or not torch is installed."""
    r = client.get("/gpu")
    assert r.status_code == 200
    body = r.json()
    for key in ("device_setting", "using_gpu", "torch_installed",
                "cuda_available", "device_count"):
        assert key in body, f"missing key: {key}"
    assert isinstance(body["devices"], list)


def test_analyze_requires_text(client):
    """POST /analyze without a text field should fail pydantic validation."""
    r = client.post("/analyze", json={})
    assert r.status_code == 422


def test_analyze_rejects_wrong_type(client):
    """POST /analyze with a non-string text field should fail validation."""
    r = client.post("/analyze", json={"text": 12345})
    assert r.status_code == 422


def test_analyze_combines_both_models_when_they_agree(client):
    """/analyze calls both classifiers and returns agreement=True when they match."""
    fake_llm = schemas.LLMResult(
        sentiment="positive", confidence=0.9,
        reasoning="clearly enthusiastic", model="mock-llm",
    )
    fake_local = schemas.LocalResult(
        sentiment="positive", confidence=0.95,
        model="mock-local", device="cpu",
    )

    with patch("main.llm_client.classify_llm", return_value=fake_llm), \
         patch("main.local_classifier.classify_local", return_value=fake_local):
        r = client.post("/analyze", json={"text": "I loved it!"})

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "I loved it!"
    assert body["llm"]["sentiment"] == "positive"
    assert body["local"]["sentiment"] == "positive"
    assert body["agreement"] is True


def test_analyze_reports_disagreement(client):
    """When the two models disagree, /analyze returns agreement=False."""
    fake_llm = schemas.LLMResult(
        sentiment="positive", confidence=0.6,
        reasoning="mildly positive", model="mock-llm",
    )
    fake_local = schemas.LocalResult(
        sentiment="neutral", confidence=0.55,
        model="mock-local", device="cpu",
    )

    with patch("main.llm_client.classify_llm", return_value=fake_llm), \
         patch("main.local_classifier.classify_local", return_value=fake_local):
        r = client.post("/analyze", json={"text": "it's fine i guess"})

    assert r.status_code == 200
    assert r.json()["agreement"] is False
