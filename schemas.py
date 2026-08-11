"""Pydantic request/response schemas for the sentiment API.

Called `schemas.py` (not `models.py`) to avoid overloading the word "model" —
this file is about *data shapes*; the ML models live in `local_classifier.py`
and are reached through LiteLLM in `llm_client.py`.

Pydantic gives us:
  * automatic FastAPI request validation (400/422 on malformed input)
  * automatic response serialization
  * type hints that IDEs and mypy understand
"""
from typing import Literal

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """POST /analyze body."""

    text: str


class LLMResult(BaseModel):
    """Classification returned by the classroom LiteLLM (Qwen coder model)."""

    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str
    model: str


class LocalResult(BaseModel):
    """Classification returned by the local HF pipeline (roberta-base)."""

    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    model: str
    device: str


class AnalyzeResponse(BaseModel):
    """POST /analyze response — includes both classifications side-by-side."""

    text: str
    llm: LLMResult
    local: LocalResult
    agreement: bool
