import pytest

from webcrawlagent.config import Settings
from webcrawlagent.llm.factory import create_llm_client
from webcrawlagent.llm.gemini_client import GeminiClient
from webcrawlagent.llm.grok_client import GrokClient


def test_factory_returns_gemini_client():
    settings = Settings(GEMINI_API_KEY="foo", LLM_PROVIDER="gemini")
    client = create_llm_client(settings)
    assert isinstance(client, GeminiClient)


def test_factory_returns_grok_client():
    settings = Settings(
        LLM_PROVIDER="grok",
        GROK_API_KEY="bar",
        GEMINI_API_KEY="unused",
    )
    client = create_llm_client(settings)
    assert isinstance(client, GrokClient)


def test_factory_requires_keys():
    settings = Settings(LLM_PROVIDER="grok", GEMINI_API_KEY="foo")
    with pytest.raises(RuntimeError):
        create_llm_client(settings)

