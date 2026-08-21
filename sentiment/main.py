"""FastAPI application — HTTP endpoints and startup lifecycle.

Business logic lives in dedicated modules; this file is thin on purpose:

  config.py            environment variables + APP_VERSION
  device.py            GPU / CPU detection + /gpu introspection
  schemas.py           Pydantic request/response shapes
  llm_client.py        sentiment via the classroom LiteLLM
  local_classifier.py  sentiment via a local HF pipeline on GPU

Endpoints:
  GET  /ready   cheap liveness probe (no dependencies exercised)
  GET  /gpu     GPU status introspection
  GET  /health  deep health check — exercises BOTH the LLM and local paths
  POST /analyze { "text": "..." } → both classifications + agreement flag
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import llm_client
import local_classifier
from config import APP_VERSION, LOCAL_MODEL_ID, SKIP_LOCAL_MODEL
from device import DEVICE, gpu_status
from schemas import AnalyzeRequest, AnalyzeResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the local HF pipeline at startup so first request is fast."""
    if SKIP_LOCAL_MODEL:
        print("[startup] SKIP_LOCAL_MODEL=1 — HF pipeline not loaded", flush=True)
    else:
        print(
            f"[startup] loading HF pipeline: {LOCAL_MODEL_ID} on {DEVICE}",
            flush=True,
        )
        local_classifier.load_pipeline()
        print("[startup] HF pipeline loaded", flush=True)
    yield


app = FastAPI(
    title="Sentiment via LiteLLM + local HF",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.get("/ready")
def ready():
    """Cheap liveness — proves the process is running. No dependencies exercised."""
    return {"ready": True, "version": APP_VERSION}


@app.get("/gpu")
def gpu():
    """GPU status introspection — see device.gpu_status for the response shape."""
    return gpu_status()


@app.get("/health")
def health():
    """Deep health check — verifies BOTH the LLM path AND the local pipeline.

    Returns 200 only if:
      * the process is up
      * the LiteLLM endpoint is reachable and returns a valid response
      * the local HF pipeline is loaded and can classify a sample input

    Coolify polls this after each deploy. A failure means the deploy is
    rolled back and previous version keeps serving.
    """
    checks: dict = {"version": APP_VERSION}
    try:
        checks["llm"] = llm_client.health_check()
    except Exception as e:
        raise HTTPException(503, f"LLM health check failed: {e}") from e
    try:
        checks["local"] = local_classifier.health_check()
    except Exception as e:
        raise HTTPException(
            503, f"local pipeline health check failed: {e}"
        ) from e
    checks["ok"] = True
    return checks


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Classify sentiment with BOTH models. Returns both + agreement flag."""
    try:
        llm_result = llm_client.classify_llm(req.text)
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    except Exception as e:  # extract_json parse errors, etc.
        raise HTTPException(500, f"LLM output could not be parsed: {e}") from e

    try:
        local_result = local_classifier.classify_local(req.text)
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    return AnalyzeResponse(
        text=req.text,
        llm=llm_result,
        local=local_result,
        agreement=(llm_result.sentiment == local_result.sentiment),
    )
