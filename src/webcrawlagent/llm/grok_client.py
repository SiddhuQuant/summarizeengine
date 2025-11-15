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

GROK_BASE_URL = "https://api.x.ai/v1"


class GrokContentError(LLMContentError):
    """Raised when Grok responds without usable JSON."""


class GrokClient:
    """Lightweight wrapper for xAI's Grok chat completions API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.grok_api_key:
            raise RuntimeError("GROK_API_KEY is not configured")
        self._client = httpx.AsyncClient(
            timeout=60,
            headers={"Authorization": f"Bearer {settings.grok_api_key}"},
        )

    async def summarize_site(self, crawl: CrawlResult, analysis: AnalysisSummary) -> SiteSummary:
        prompt = build_summary_prompt(crawl, analysis, self.settings.crawl_max_tokens)
        response = await self._client.post(
            f"{GROK_BASE_URL}/chat/completions",
            json={
                "model": self.settings.grok_model,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Grok, an investigator who converts crawl data into concise, "
                            "actionable website summaries. Respond strictly with JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "website_report",
                        "schema": SUMMARY_SCHEMA,
                    },
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        text = _extract_text(payload)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - depends on remote output
            raise GrokContentError(f"Grok returned invalid JSON: {text}") from exc
        return SiteSummary.from_llm_payload(parsed)

    async def summarize_text(self, text: str, document_type: str = "document") -> SiteSummary:
        """Summarize raw text content using Grok."""
        prompt = build_text_summary_prompt(text, document_type)
        response = await self._client.post(
            f"{GROK_BASE_URL}/chat/completions",
            json={
                "model": self.settings.grok_model,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Grok, an analyst who converts text content into concise, "
                            "actionable summaries. Respond strictly with JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "text_summary",
                        "schema": SUMMARY_SCHEMA,
                    },
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        text_response = _extract_text(payload)
        try:
            parsed = json.loads(text_response)
        except json.JSONDecodeError as exc:  # pragma: no cover - depends on remote output
            raise GrokContentError(f"Grok returned invalid JSON: {text_response}") from exc
        return SiteSummary.from_llm_payload(parsed)

    async def aclose(self) -> None:
        await self._client.aclose()


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise GrokContentError("Grok returned no choices", payload=payload)

    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(content, str) and content.strip():
            return content

    finish_reason = choices[0].get("finish_reason")
    details = f"finish_reason={finish_reason}" if finish_reason else "no finish_reason provided"
    raise GrokContentError(f"Grok response had no text part ({details})", payload=payload)

