from __future__ import annotations

import json
import logging
from typing import Any

from webcrawlagent.crawler.analyzer import AnalysisSummary
from webcrawlagent.crawler.extractor import CrawlResult
from webcrawlagent.report.models import SiteSummary

logger = logging.getLogger(__name__)

# Gemini-compatible schema - using a more flexible structure
# Note: Gemini API requires additionalProperties to be a schema object, not just True
SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "content_type": {"type": "string"},
        "sections": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"}
            },
        },
    },
    "required": ["overview", "content_type", "sections"],
}


def build_summary_prompt(
    crawl: CrawlResult, analysis: AnalysisSummary, max_tokens: int
) -> str:
    content_chunks = crawl.aggregate_text(max_tokens)
    context = "\n\n".join(content_chunks)
    summary_metadata = json.dumps(
        {
            "root_url": analysis.root_url,
            "pages": analysis.page_summaries,
            "keywords": analysis.keywords,
            "cta_links": analysis.ctas,
        },
        ensure_ascii=False,
    )
    instructions = (
        "You are an analyst generating a concise website briefing. "
        "Blend the structured metadata with the raw text to produce actionable insight.\n\n"
        "Return **only** JSON with this structure:\n"
        "{\n"
        '  "overview": <2-4 sentence comprehensive synopsis>,\n'
        '  "content_type": "website",\n'
        '  "sections": {\n'
        '    "key_sections": [list of key sections and their purpose],\n'
        '    "highlights": [bullet-level product/features/metrics insights],\n'
        '    "recommendations": [next actions or opportunities]\n'
        '  }\n'
        "}\n"
    )
    return (
        f"{instructions}\n"
        f"Metadata: {summary_metadata}\n"
        "Content: \n"
        f"{context}"
    )


def build_text_summary_prompt(text: str, document_type: str = "document") -> str:
    """Build a context-aware prompt for summarizing raw text content."""
    instructions = (
        "You are an intelligent analyst that adapts summaries based on content type. "
        "First, analyze the content to determine its type (conversation, article, meeting notes, "
        "document, email thread, etc.), then generate an appropriate summary structure.\n\n"
        "For conversations/dialogues: Identify speakers, main topics discussed, key points made by each speaker, "
        "decisions reached, and action items.\n\n"
        "For articles/documents: Identify main themes, key arguments, important facts, and conclusions.\n\n"
        "For meeting notes: Identify participants, agenda items, decisions made, and action items.\n\n"
        "Return **only** JSON with this structure:\n"
        "{\n"
        '  "overview": <2-4 sentence comprehensive synopsis that captures the essence>,\n'
        '  "content_type": <detected type: "conversation", "article", "meeting", "document", etc.>,\n'
        '  "sections": {\n'
        '    "<section_name>": [<relevant items>],\n'
        '    ...\n'
        '  }\n'
        "}\n\n"
        "Use appropriate section names based on content type:\n"
        "- Conversations: 'speakers', 'topics_discussed', 'key_points', 'decisions', 'action_items'\n"
        "- Articles: 'main_themes', 'key_arguments', 'important_facts', 'conclusions'\n"
        "- Meetings: 'participants', 'agenda_items', 'decisions_made', 'action_items', 'next_steps'\n"
        "- Documents: 'main_sections', 'key_findings', 'important_details', 'summary_points'\n"
        "- General: 'main_points', 'key_insights', 'important_information', 'takeaways'\n\n"
        "Be intelligent and create sections that make sense for the content. Include 3-6 relevant sections.\n\n"
        "Content to analyze:\n"
        f"{text}"
    )
    return instructions


def build_fallback_summary(
    crawl: CrawlResult, analysis: AnalysisSummary, *, reason: str
) -> SiteSummary:
    """Construct a deterministic summary when an LLM output is unavailable."""
    sections_list: list[str] = []
    for page in analysis.page_summaries[:3]:
        snippet = (
            page.get("description")
            or " / ".join(page.get("headings", []))
            or f"{page.get('word_count', 0)} words (status {page.get('status')})"
        )
        sections_list.append(f"{page.get('title')}: {snippet}")

    if not sections_list and crawl.pages:
        first_page = crawl.pages[0]
        sections_list.append(f"{first_page.title or first_page.url}: {first_page.description}")

    highlights: list[str] = []
    if analysis.keywords:
        highlights.append("Top keywords: " + ", ".join(analysis.keywords[:6]))
    highlights.append(
        f"Internal links: {analysis.internal_links} · External links: {analysis.external_links}"
    )
    if analysis.ctas:
        highlights.append("Detected CTAs: " + ", ".join(analysis.ctas[:5]))

    recommendations = [
        "Review the crawler output manually because the LLM blocked the content.",
        "Retry with sanitized text or a different site if you need an AI-authored summary.",
    ]

    overview = (
        f"Crawled {analysis.total_pages} page(s) from {analysis.root_url}. "
        f"LLM output unavailable ({reason}); showing crawler-derived summary."
    )

    logger.warning("Using fallback summary because %s", reason)
    return SiteSummary(
        overview=overview,
        content_type="website",
        sections={
            "key_sections": sections_list or [analysis.root_url],
            "highlights": highlights,
            "recommendations": recommendations,
        },
        highlights=highlights,
        recommendations=recommendations,
    )

