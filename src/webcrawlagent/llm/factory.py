from __future__ import annotations

from webcrawlagent.config import Settings
from webcrawlagent.llm.gemini_client import GeminiClient
from webcrawlagent.llm.grok_client import GrokClient


def create_llm_client(settings: Settings):
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        if not settings.gemini_api_key or not settings.gemini_api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini. Please set a valid API key in your environment variables or .env file.")
        return GeminiClient(settings)
    if provider == "grok":
        if not settings.grok_api_key or not settings.grok_api_key.strip():
            raise RuntimeError("GROK_API_KEY is required when LLM_PROVIDER=grok. Please set a valid API key in your environment variables or .env file.")
        return GrokClient(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

