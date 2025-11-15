from webcrawlagent.llm.gemini_client import (
    GeminiContentError,
    _parse_summary_text,
)


def test_parse_summary_text_handles_code_fence():
    text = """```json
    {
        "overview": "ok"
    }
    ```"""
    parsed = _parse_summary_text(text)
    assert parsed["overview"] == "ok"


def test_parse_summary_text_raises_for_invalid_json():
    text = "```json not valid ```"
    try:
        _parse_summary_text(text)
        assert False, "Expected GeminiContentError"
    except GeminiContentError as exc:
        assert "invalid JSON" in str(exc)

