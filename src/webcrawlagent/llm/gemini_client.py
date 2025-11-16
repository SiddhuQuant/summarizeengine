from __future__ import annotations

import json
import re
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
                    "maxOutputTokens": 4096,  # Increased for complex documents and detailed summaries
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
        
        # Check if response was truncated
        finish_reason = None
        if payload.get("candidates"):
            finish_reason = payload["candidates"][0].get("finishReason")
            if finish_reason == "MAX_TOKENS":
                # Response was truncated, try to extract partial JSON
                pass
        
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
        # Try to extract partial JSON if the response was truncated
        partial_data = _try_extract_partial_json(cleaned)
        if partial_data:
            return partial_data
        
        preview = cleaned.strip().replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:237] + "..."
        raise GeminiContentError(
            f"Gemini returned invalid JSON: {preview}", payload={"text": text}
        ) from exc


def _try_extract_partial_json(text: str) -> dict[str, Any] | None:
    """Attempt to extract partial JSON from truncated responses."""
    # Try to find the last complete JSON structure
    # Look for closing braces to find where JSON might be complete
    text = text.strip()
    
    # Try to find and extract a valid JSON object even if incomplete
    # Look for the start of a JSON object
    start_idx = text.find("{")
    if start_idx == -1:
        return None
    
    # Try progressively shorter suffixes to find valid JSON
    for end_offset in range(0, len(text) - start_idx):
        try:
            candidate = text[start_idx:len(text) - end_offset]
            # Try to close any open structures
            open_braces = candidate.count("{") - candidate.count("}")
            open_brackets = candidate.count("[") - candidate.count("]")
            
            # Close open structures
            if open_braces > 0:
                candidate += "}" * open_braces
            if open_brackets > 0:
                candidate += "]" * open_brackets
            
            # Try to parse
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    
    # If that didn't work, try to extract just the overview field
    # Look for "overview" field even in incomplete JSON
    overview_match = None
    overview_pattern = r'"overview"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
    match = re.search(overview_pattern, text)
    if match:
        overview_match = match.group(1).replace('\\"', '"').replace('\\n', '\n')
    
    if overview_match:
        # Return a minimal valid structure with at least the overview
        return {
            "overview": overview_match,
            "content_type": "document",
            "sections": {
                "note": ["JSON response was incomplete. Only partial data extracted."]
            }
        }
    
    return None


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
