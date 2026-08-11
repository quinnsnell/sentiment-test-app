"""FastAPI sentiment-analysis app — dual-model reference for the classroom.

Demonstrates two ML paradigms side-by-side:
  * `llm` — prompt-engineered classification via the classroom LiteLLM
            (general-purpose model, structured output, ~500ms-1.5s per request).
  * `local` — a fine-tuned Hugging Face classifier running on the container's
             GPU (or CPU fallback), ~50ms per request.

Endpoints:
  GET  /ready   — cheap liveness probe (no model calls)
  GET  /health  — deep health check (exercises both models)
  POST /analyze — { "text": "..." } → both classifications + agreement flag

Config (all via env vars — never hardcode):
  LITELLM_URL        default http://ml-capstone.cs.byu.edu:4000/v1
  LITELLM_API_KEY    default sk-noauth
  MODEL              default classroom-chat
  LOCAL_MODEL_ID     default cardiffnlp/twitter-roberta-base-sentiment-latest
  DEVICE             default: auto — cuda:0 if available, else cpu
  SKIP_LOCAL_MODEL   default unset — set to "1" in tests to avoid loading the HF model
"""
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_VERSION = "0.4.2"

LITELLM_URL = os.environ.get("LITELLM_URL", "http://ml-capstone.cs.byu.edu:4000/v1").rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-noauth")
MODEL = os.environ.get("MODEL", "classroom-chat")
LOCAL_MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest")


def _detect_device() -> str:
    """Pick the best available device for the HF pipeline.

    Precedence:
      1. If DEVICE env var is set, use it verbatim (deterministic override).
      2. If CUDA is available and multiple GPUs are visible, pick the one with
         the most free VRAM ("least-loaded auto-pick"). Handles the case where
         several student containers share a machine and see all GPUs — spreads
         load without external coordination.
      3. If exactly one GPU is visible (e.g., because the container was started
         with CUDA_VISIBLE_DEVICES=N), use cuda:0 (which maps to that GPU).
      4. Otherwise, fall back to CPU.

    Admin-side alternative: at Coolify Application creation, set
    CUDA_VISIBLE_DEVICES=<group_num % 4> on each container to pin it to a
    specific GPU. Cheaper than runtime auto-pick and avoids race conditions
    when multiple containers restart simultaneously.
    """
    if forced := os.environ.get("DEVICE"):
        return forced
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        n = torch.cuda.device_count()
        if n == 1:
            return "cuda:0"
        # Pick the GPU with the most free memory
        best_idx = max(
            range(n),
            key=lambda i: torch.cuda.mem_get_info(i)[0],  # free bytes on device i
        )
        return f"cuda:{best_idx}"
    except ImportError:
        pass
    return "cpu"


DEVICE = _detect_device()

# Populated by the lifespan handler at startup (or by tests via mock).
_local_pipeline = None


def _load_local_pipeline():
    """Load the HF sentiment classifier onto DEVICE. Called once at startup."""
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model=LOCAL_MODEL_ID,
        device=DEVICE,
        top_k=None,       # return all class scores, not just the top one
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm the local HF pipeline so first request is fast."""
    global _local_pipeline
    if os.environ.get("SKIP_LOCAL_MODEL") == "1":
        # Test mode — bypass real load
        print(f"[startup] SKIP_LOCAL_MODEL=1 — not loading HF pipeline", flush=True)
    else:
        print(f"[startup] loading HF pipeline: {LOCAL_MODEL_ID} on {DEVICE}", flush=True)
        _local_pipeline = _load_local_pipeline()
        print(f"[startup] HF pipeline loaded", flush=True)
    yield


app = FastAPI(title="Sentiment via LiteLLM + local HF", version=APP_VERSION, lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str


class LLMResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str
    model: str


class LocalResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    model: str
    device: str


class AnalyzeResponse(BaseModel):
    text: str
    llm: LLMResult
    local: LocalResult
    agreement: bool


SYSTEM_PROMPT = (
    "You classify the sentiment of user-provided text. "
    'Respond with ONLY a compact JSON object matching this schema: '
    '{"sentiment": "positive"|"negative"|"neutral", '
    '"confidence": 0.0-1.0, "reasoning": "one short sentence"}. '
    "No prose outside the JSON. No code fences."
)


def _extract_json(content: str) -> dict:
    """Pull the first {...} block out of a possibly-messy LLM response."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object found in: {content[:200]}")
    return json.loads(m.group(0))


def _normalize_local_label(label: str) -> Literal["positive", "negative", "neutral"]:
    """The cardiffnlp model returns 'positive'/'negative'/'neutral' but other
    HF sentiment models use LABEL_0/LABEL_1 or POSITIVE/NEGATIVE. Normalize."""
    label = label.lower().strip()
    if label in ("positive", "label_2", "pos"):
        return "positive"
    if label in ("negative", "label_0", "neg"):
        return "negative"
    if label in ("neutral", "label_1"):
        return "neutral"
    # Fallback: return neutral for unknown labels
    return "neutral"


def _classify_local(text: str) -> LocalResult:
    """Run the HF sentiment pipeline. Returns the top-scoring class."""
    if _local_pipeline is None:
        raise RuntimeError("local pipeline not loaded")

    # top_k=None returns list of dicts sorted by score, wrapped in a list per input
    results = _local_pipeline(text)
    if isinstance(results[0], list):
        results = results[0]  # unwrap batched output
    top = max(results, key=lambda r: r["score"])

    return LocalResult(
        sentiment=_normalize_local_label(top["label"]),
        confidence=float(top["score"]),
        model=LOCAL_MODEL_ID,
        device=DEVICE,
    )


def _classify_llm(text: str) -> LLMResult:
    """Ask the classroom LLM to classify sentiment via structured-output prompt."""
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "max_tokens": 200,
                "temperature": 0.1,
            },
        )
    if r.status_code != 200:
        raise HTTPException(502, f"LiteLLM {r.status_code}: {r.text[:400]}")

    content = r.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    return LLMResult(
        sentiment=parsed["sentiment"],
        confidence=float(parsed.get("confidence", 0.0)),
        reasoning=parsed.get("reasoning", ""),
        model=MODEL,
    )


@app.get("/ready")
def ready():
    """Cheap liveness — proves the process is running. No dependencies exercised."""
    return {"ready": True, "version": APP_VERSION}


@app.get("/gpu")
def gpu_info():
    """GPU status introspection — hit this to confirm your Coolify Application
    actually has GPU access. Reports the configured device, whether PyTorch sees
    CUDA, and per-GPU stats (name, VRAM total + allocated).

    Educational endpoint — not required by Coolify or health checks. Handy for
    students setting up their own containers with GPU reservations.
    """
    info: dict = {
        "device_setting": DEVICE,
        "using_gpu": DEVICE.startswith("cuda"),
        "torch_installed": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch
        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            info["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3), 1
                    ),
                    "memory_allocated_gb": round(
                        torch.cuda.memory_allocated(i) / (1024**3), 2
                    ),
                    "memory_free_gb": round(
                        torch.cuda.mem_get_info(i)[0] / (1024**3), 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


@app.get("/health")
def health():
    """Deep health check — verifies BOTH the LLM path and the local pipeline.

    Returns 200 only if:
      - the process is up
      - the LiteLLM endpoint is reachable and returns a valid response
      - the local HF pipeline is loaded and can classify a sample input

    Coolify polls this after each deploy. Failure means the deploy is rolled back.
    """
    checks: dict = {"version": APP_VERSION}

    # LLM path
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                f"{LITELLM_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "healthcheck"}],
                    "max_tokens": 5,
                },
            )
        r.raise_for_status()
        r.json()["choices"][0]["message"]["content"]
        checks["llm"] = {"ok": True, "url": LITELLM_URL, "model": MODEL}
    except Exception as e:
        raise HTTPException(503, f"LLM health check failed: {e}") from e

    # Local pipeline path
    try:
        if _local_pipeline is None:
            raise RuntimeError("local pipeline not loaded")
        _classify_local("healthcheck")   # will raise if broken

        # Confirm GPU state matches what we asked for. If DEVICE=cuda:0 but
        # torch.cuda.is_available() is False, that means the container has no
        # GPU allocated — the pipeline silently fell back to CPU or is broken.
        gpu_ok = True
        gpu_note = None
        if DEVICE.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    gpu_ok = False
                    gpu_note = "DEVICE=cuda but torch.cuda.is_available() is False — no GPU allocated to this container"
            except ImportError:
                gpu_ok = False
                gpu_note = "DEVICE=cuda but torch not installed"

        checks["local"] = {
            "ok": True,
            "model": LOCAL_MODEL_ID,
            "device": DEVICE,
            "using_gpu": DEVICE.startswith("cuda") and gpu_ok,
        }
        if gpu_note:
            checks["local"]["gpu_note"] = gpu_note
    except Exception as e:
        raise HTTPException(503, f"local pipeline health check failed: {e}") from e

    checks["ok"] = True
    return checks


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Classify sentiment with BOTH models. Returns both results + agreement flag."""
    llm_result = _classify_llm(req.text)
    local_result = _classify_local(req.text)
    return AnalyzeResponse(
        text=req.text,
        llm=llm_result,
        local=local_result,
        agreement=(llm_result.sentiment == local_result.sentiment),
    )
