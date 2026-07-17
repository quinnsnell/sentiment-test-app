"""FastAPI sentiment-analysis app for smoke-testing Coolify + LiteLLM.

POST /analyze  { "text": "..." }
  -> { "text", "sentiment": positive|negative|neutral, "confidence", "reasoning" }

GET  /health   -> { "ok": true, "litellm": "<upstream url>" }

Configure with env:
  LITELLM_URL       default http://rigel.cs.byu.edu:4000/v1
  LITELLM_API_KEY   default sk-noauth
  MODEL             default classroom-chat
"""
import json
import os
import re
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LITELLM_URL = os.environ.get("LITELLM_URL", "http://rigel.cs.byu.edu:4000/v1").rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-noauth")
MODEL = os.environ.get("MODEL", "classroom-chat")

app = FastAPI(title="Sentiment via LiteLLM", version="0.1.0")


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str


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


@app.get("/health")
def health():
    return {"ok": True, "litellm": LITELLM_URL, "model": MODEL}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.text},
                ],
                "max_tokens": 200,
                "temperature": 0.1,
            },
        )
    if r.status_code != 200:
        raise HTTPException(502, f"LiteLLM {r.status_code}: {r.text[:400]}")

    content = r.json()["choices"][0]["message"]["content"]
    try:
        parsed = _extract_json(content)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"could not parse LLM output: {e}") from e

    return AnalyzeResponse(
        text=req.text,
        sentiment=parsed["sentiment"],
        confidence=float(parsed.get("confidence", 0.0)),
        reasoning=parsed.get("reasoning", ""),
    )
