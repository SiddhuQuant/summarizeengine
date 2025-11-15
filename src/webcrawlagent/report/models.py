from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from webcrawlagent.crawler.analyzer import AnalysisSummary


@dataclass(slots=True)
class SiteSummary:
    overview: str
    content_type: str
    sections: dict[str, list[str]]  # Dynamic sections based on content type
    # Legacy fields for backward compatibility with website summaries
    highlights: list[str] | None = None
    recommendations: list[str] | None = None

    @classmethod
    def from_llm_payload(cls, payload: dict) -> SiteSummary:
        # Handle new flexible format
        if "sections" in payload and isinstance(payload.get("sections"), dict):
            return cls(
                overview=payload.get("overview", ""),
                content_type=payload.get("content_type", "document"),
                sections=payload.get("sections", {}),
                highlights=payload.get("highlights"),
                recommendations=payload.get("recommendations"),
            )
        # Handle legacy format (for website summaries)
        else:
            sections_dict: dict[str, list[str]] = {}
            if payload.get("sections"):
                sections_dict["key_sections"] = payload.get("sections", [])
            if payload.get("highlights"):
                sections_dict["highlights"] = payload.get("highlights", [])
            if payload.get("recommendations"):
                sections_dict["recommendations"] = payload.get("recommendations", [])
            
            return cls(
                overview=payload.get("overview", ""),
                content_type=payload.get("content_type", "website"),
                sections=sections_dict,
                highlights=payload.get("highlights"),
                recommendations=payload.get("recommendations"),
            )


@dataclass(slots=True)
class ReportPayload:
    url: str
    summary: SiteSummary
    metrics: AnalysisSummary
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pdf_path: str | None = None
