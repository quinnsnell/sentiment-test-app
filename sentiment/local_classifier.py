"""Local sentiment classification via a HuggingFace transformers pipeline.

Contrast with `llm_client.py` — that one calls a big general-purpose model
via HTTP; this one runs a small fine-tuned classifier directly on the
container's GPU. Different trade-offs, same end result (sentiment label +
confidence).

The pipeline is loaded once at startup (see `load_pipeline`) and reused
across requests. Module-level `_pipeline` holds the loaded instance.
"""
from typing import Literal

from config import LOCAL_MODEL_ID
from device import DEVICE
from schemas import LocalResult


# Module-private state. Populated by load_pipeline() at startup, patched by
# tests when SKIP_LOCAL_MODEL=1.
_pipeline = None


def load_pipeline():
    """Load the HF sentiment pipeline onto DEVICE. Called once at startup.

    Import `transformers` lazily so callers who set SKIP_LOCAL_MODEL=1
    don't pay the import cost.
    """
    global _pipeline
    from transformers import pipeline

    _pipeline = pipeline(
        "sentiment-analysis",
        model=LOCAL_MODEL_ID,
        device=DEVICE,
        top_k=None,  # return all class scores, not just the top one
    )
    return _pipeline


def normalize_label(label: str) -> Literal["positive", "negative", "neutral"]:
    """Map an HF model's label output to our standard three-class vocabulary.

    Different HF sentiment models use different label conventions:
      * cardiffnlp roberta returns 'positive'/'negative'/'neutral' directly
      * distilbert-sst returns 'POSITIVE'/'NEGATIVE' (no neutral)
      * some models return 'LABEL_0', 'LABEL_1', 'LABEL_2'

    We map everything to lowercase positive/negative/neutral. Unknown
    labels are mapped to neutral as a safe default.
    """
    label = label.lower().strip()
    if label in ("positive", "label_2", "pos"):
        return "positive"
    if label in ("negative", "label_0", "neg"):
        return "negative"
    if label in ("neutral", "label_1"):
        return "neutral"
    return "neutral"


def classify_local(text: str) -> LocalResult:
    """Run the HF pipeline on `text` and return the top-scoring class.

    Raises RuntimeError if the pipeline hasn't been loaded (either
    startup failed, or the caller is running with SKIP_LOCAL_MODEL=1 and
    never patched _pipeline).
    """
    if _pipeline is None:
        raise RuntimeError("local pipeline not loaded")

    # top_k=None returns list of dicts sorted by score, wrapped in a list
    # per input.
    results = _pipeline(text)
    if isinstance(results[0], list):
        results = results[0]  # unwrap batched output
    top = max(results, key=lambda r: r["score"])

    return LocalResult(
        sentiment=normalize_label(top["label"]),
        confidence=float(top["score"]),
        model=LOCAL_MODEL_ID,
        device=DEVICE,
    )


def health_check() -> dict:
    """Verify the local pipeline is loaded and can classify a sample.

    Also confirms the GPU-vs-CPU state matches what we asked for: if
    DEVICE='cuda:0' but torch.cuda.is_available() is False, that's a
    silent CPU fallback — flag it in the response.

    Returns a dict describing what was checked. Raises on failure.
    """
    if _pipeline is None:
        raise RuntimeError("local pipeline not loaded")

    classify_local("healthcheck")  # raises if pipeline is broken

    gpu_ok = True
    gpu_note = None
    if DEVICE.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                gpu_ok = False
                gpu_note = (
                    "DEVICE=cuda but torch.cuda.is_available() is False "
                    "— no GPU allocated to this container"
                )
        except ImportError:
            gpu_ok = False
            gpu_note = "DEVICE=cuda but torch not installed"

    result = {
        "ok": True,
        "model": LOCAL_MODEL_ID,
        "device": DEVICE,
        "using_gpu": DEVICE.startswith("cuda") and gpu_ok,
    }
    if gpu_note:
        result["gpu_note"] = gpu_note
    return result
