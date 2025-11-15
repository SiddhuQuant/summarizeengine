from __future__ import annotations

from webcrawlagent.config import Settings
from webcrawlagent.llm.gemini_client import GeminiClient
from webcrawlagent.llm.grok_client import GrokClient


def create_llm_client(settings: Settings):
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        return GeminiClient(settings)
    if provider == "grok":
        if not settings.grok_api_key:
            raise RuntimeError("GROK_API_KEY is required when LLM_PROVIDER=grok")
        return GrokClient(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

