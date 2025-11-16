from __future__ import annotations

import json
from typing import Any

import httpx

from webcrawlagent.config import Settings
from webcrawlagent.crawler.analyzer import AnalysisSummary
from webcrawlagent.crawler.extractor import CrawlResult
from webcrawlagent.llm.exceptions import LLMContentError
from webcrawlagent.llm.summary import SUMMARY_SCHEMA, build_summary_prompt, build_text_summary_prompt
from webcrawlagent.report.models import SiteSummary

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiContentError(LLMContentError):
    """Raised when the Gemini API responds without usable text."""


class GeminiClient:
    """Lightweight wrapper around the Gemini REST API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=50)

    async def summarize_site(self, crawl: CrawlResult, analysis: AnalysisSummary) -> SiteSummary:
        if not self.settings.gemini_api_key or not self.settings.gemini_api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is not configured. Please set a valid API key in your environment variables or .env file.")
        prompt = build_summary_prompt(crawl, analysis, self.settings.crawl_max_tokens)
        url = f"{GEMINI_BASE_URL}/models/{self.settings.gemini_model}:generateContent"
        response = await self._client.post(
            url,
            params={"key": self.settings.gemini_api_key},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                    "responseMimeType": "application/json",
                    "responseSchema": SUMMARY_SCHEMA,
                },
            },
        )
        if not response.is_success:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("error", {}).get("message", str(error_data))
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"
            raise GeminiContentError(
                f"Gemini API error: {error_detail}", payload={"status": response.status_code}
            )
        
        payload = response.json()
        text = _extract_text(payload)
        parsed = _parse_summary_text(text)
        return SiteSummary.from_llm_payload(parsed)

    async def summarize_text(self, text: str, document_type: str = "document") -> SiteSummary:
        """Summarize raw text content using Gemini."""
        if not self.settings.gemini_api_key or not self.settings.gemini_api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is not configured. Please set a valid API key in your environment variables or .env file.")
        prompt = build_text_summary_prompt(text, document_type)
        url = f"{GEMINI_BASE_URL}/models/{self.settings.gemini_model}:generateContent"
        # Note: We don't use responseSchema here because Gemini doesn't support
        # additionalProperties for dynamic sections. Instead, we request JSON format
        # and parse it manually.
        response = await self._client.post(
            url,
            params={"key": self.settings.gemini_api_key},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,  # Increased for more detailed summaries
                    "responseMimeType": "application/json",
                },
            },
        )
        if not response.is_success:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("error", {}).get("message", str(error_data))
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"
            raise GeminiContentError(
                f"Gemini API error: {error_detail}", payload={"status": response.status_code}
            )
        
        payload = response.json()
        text_response = _extract_text(payload)
        parsed = _parse_summary_text(text_response)
        return SiteSummary.from_llm_payload(parsed)

    async def aclose(self) -> None:
        await self._client.aclose()

def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise GeminiContentError("Gemini returned no candidates", payload=payload)

    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts") or []
        for part in parts:
            text = part.get("text")
            if text:
                return text

    finish_reason = candidates[0].get("finishReason")
    block_reason = payload.get("promptFeedback", {}).get("blockReason")
    details: list[str] = []
    if finish_reason:
        details.append(f"finishReason={finish_reason}")
    if block_reason:
        details.append(f"blockReason={block_reason}")

    message = "Gemini response had no text part"
    if details:
        message = f"{message} ({', '.join(details)})"
    raise GeminiContentError(message, payload=payload)


def _parse_summary_text(text: str) -> dict[str, Any]:
    """Parse Gemini free-form output into JSON, trimming Markdown fences if needed."""
    cleaned = _strip_code_block(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on remote output
        preview = cleaned.strip().replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:237] + "..."
        raise GeminiContentError(
            f"Gemini returned invalid JSON: {preview}", payload={"text": text}
        ) from exc


def _strip_code_block(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    body: list[str] = []
    started = False
    for line in lines:
        fence = line.strip().startswith("```")
        if not started:
            if fence:
                started = True
            continue
        if fence:
            break
        body.append(line)
    if not body:
        return stripped
    return "\n".join(body).strip()
