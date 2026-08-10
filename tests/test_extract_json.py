"""Tests for the _extract_json helper — a pure function that pulls JSON out of
possibly-messy LLM responses. No LLM needed for these tests.
"""
import pytest

from main import _extract_json


def test_plain_json_passes_through():
    result = _extract_json('{"sentiment": "positive", "confidence": 0.9}')
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.9


def test_json_inside_code_fence():
    """Some models wrap JSON in ```json ... ``` — the extractor should handle it."""
    result = _extract_json('```json\n{"sentiment": "positive"}\n```')
    assert result["sentiment"] == "positive"


def test_json_inside_bare_code_fence():
    """Some models use ``` without the 'json' language marker."""
    result = _extract_json('```\n{"sentiment": "negative"}\n```')
    assert result["sentiment"] == "negative"


def test_json_embedded_in_prose():
    """Some models add prose before/after the JSON — extractor should find the object."""
    result = _extract_json('Here is my analysis: {"sentiment": "neutral"} thanks!')
    assert result["sentiment"] == "neutral"


def test_multiline_json():
    """JSON often spans multiple lines with indentation."""
    result = _extract_json('''{
        "sentiment": "positive",
        "confidence": 0.87,
        "reasoning": "clearly enthusiastic"
    }''')
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.87


def test_raises_when_no_json():
    """If the response contains no JSON object, we should raise a ValueError."""
    with pytest.raises(ValueError, match="no JSON"):
        _extract_json("just some prose no braces here")


def test_raises_when_json_is_invalid():
    """Malformed JSON inside recognizable braces should raise JSONDecodeError."""
    import json
    with pytest.raises(json.JSONDecodeError):
        # Has braces so the regex matches, but the contents aren't valid JSON.
        _extract_json('{"sentiment": positive}')
