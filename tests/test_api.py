"""Unit tests for the sentiment app.

These test in-isolation behavior and DO NOT hit the classroom LLM (which needs
VPN + a live LiteLLM) or load the real HF pipeline. The live-integration check
is what `/health` does at deploy time — Coolify polls it, and a failure marks
the deploy unhealthy.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from main import LocalResult, app


@pytest.fixture
def client():
    """FastAPI TestClient with the lifespan disabled (SKIP_LOCAL_MODEL=1 in conftest)."""
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
    """/gpu returns a JSON shape regardless of whether torch is installed."""
    r = client.get("/gpu")
    assert r.status_code == 200
    body = r.json()
    # Fields that must always be present, regardless of GPU availability
    assert "device_setting" in body
    assert "using_gpu" in body
    assert "torch_installed" in body
    assert "cuda_available" in body
    assert "device_count" in body
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
    """POST /analyze returns both classifications and agreement=True when they match."""
    fake_llm = main.LLMResult(
        sentiment="positive", confidence=0.9,
        reasoning="clearly enthusiastic", model="mock-llm",
    )
    fake_local = main.LocalResult(
        sentiment="positive", confidence=0.95,
        model="mock-local", device="cpu",
    )

    with patch("main._classify_llm", return_value=fake_llm), \
         patch("main._classify_local", return_value=fake_local):
        r = client.post("/analyze", json={"text": "I loved it!"})

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "I loved it!"
    assert body["llm"]["sentiment"] == "positive"
    assert body["local"]["sentiment"] == "positive"
    assert body["agreement"] is True


def test_analyze_reports_disagreement(client):
    """When the two models disagree, /analyze returns agreement=False."""
    fake_llm = main.LLMResult(
        sentiment="positive", confidence=0.6,
        reasoning="mildly positive", model="mock-llm",
    )
    fake_local = main.LocalResult(
        sentiment="neutral", confidence=0.55,
        model="mock-local", device="cpu",
    )

    with patch("main._classify_llm", return_value=fake_llm), \
         patch("main._classify_local", return_value=fake_local):
        r = client.post("/analyze", json={"text": "it's fine i guess"})

    assert r.status_code == 200
    assert r.json()["agreement"] is False
