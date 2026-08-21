"""Sentiment classification via the classroom LiteLLM.

Uses prompt-engineered structured output — we ask the general-purpose Qwen
coder model for a strict JSON object, then parse it. The model can be a
little chatty (wrap JSON in code fences, add prose), so `extract_json`
handles the common messiness.
"""
import json
import re

import httpx

from config import LITELLM_API_KEY, LITELLM_URL, MODEL
from schemas import LLMResult


SYSTEM_PROMPT = (
    "You classify the sentiment of user-provided text. "
    'Respond with ONLY a compact JSON object matching this schema: '
    '{"sentiment": "positive"|"negative"|"neutral", '
    '"confidence": 0.0-1.0, "reasoning": "one short sentence"}. '
    "No prose outside the JSON. No code fences."
)


def extract_json(content: str) -> dict:
    """Pull the first {...} block out of a possibly-messy LLM response.

    Handles three common patterns from LLMs:
      * plain JSON object (best case)
      * JSON wrapped in ```json ... ``` code fences
      * JSON preceded/followed by prose ("Here is my analysis: {...}")

    Raises ValueError if no `{...}` block is found, or JSONDecodeError if
    the block isn't valid JSON.
    """
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object found in: {content[:200]}")
    return json.loads(m.group(0))


def classify_llm(text: str) -> LLMResult:
    """Ask the classroom LLM to classify sentiment.

    Raises RuntimeError on any failure (network, non-200 response, malformed
    output). Caller decides how to surface that to the client.
    """
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
        raise RuntimeError(f"LiteLLM {r.status_code}: {r.text[:400]}")

    content = r.json()["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    return LLMResult(
        sentiment=parsed["sentiment"],
        confidence=float(parsed.get("confidence", 0.0)),
        reasoning=parsed.get("reasoning", ""),
        model=MODEL,
    )


def health_check() -> dict:
    """Verify the LLM path is reachable + returns valid structure.

    Sends a trivial "healthcheck" prompt and checks the response shape. Uses
    a smaller max_tokens and shorter timeout than /analyze because we're
    only proving the pipeline works, not doing real classification.

    Returns a dict describing what was checked. Raises on failure.
    """
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
    r.json()["choices"][0]["message"]["content"]  # KeyError = malformed
    return {"ok": True, "url": LITELLM_URL, "model": MODEL}
