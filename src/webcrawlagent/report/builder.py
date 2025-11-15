from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from webcrawlagent.config import Settings
from webcrawlagent.report.models import ReportPayload

TITLE = "Content Summary Report"
LEFT_MARGIN = 20
RIGHT_MARGIN = 20
TOP_MARGIN = 20
SECTION_SPACING = 3
LINE_HEIGHT = 6


class PdfReportBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.output_dir = self.settings.ensure_report_dir()

    def build(self, payload: ReportPayload) -> Path:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=TOP_MARGIN)
        pdf.set_margins(LEFT_MARGIN, TOP_MARGIN, RIGHT_MARGIN)
        pdf.add_page()

        self._write_header(pdf, payload)
        self._section(pdf, "Overview", [payload.summary.overview], emphasize=True)
        
        # Use dynamic sections if available, otherwise fall back to legacy format
        if payload.summary.sections:
            # Render dynamic sections with human-readable titles
            section_titles = {
                "speakers": "Speakers/Participants",
                "topics_discussed": "Topics Discussed",
                "key_points": "Key Points",
                "main_points": "Main Points",
                "main_themes": "Main Themes",
                "key_arguments": "Key Arguments",
                "important_facts": "Important Facts",
                "conclusions": "Conclusions",
                "decisions": "Decisions Made",
                "decisions_made": "Decisions Made",
                "action_items": "Action Items",
                "next_steps": "Next Steps",
                "participants": "Participants",
                "agenda_items": "Agenda Items",
                "main_sections": "Main Sections",
                "key_findings": "Key Findings",
                "important_details": "Important Details",
                "summary_points": "Summary Points",
                "key_insights": "Key Insights",
                "important_information": "Important Information",
                "takeaways": "Takeaways",
                "key_sections": "Key Sections",
                "highlights": "Highlights",
                "recommendations": "Recommendations",
                "error_info": "Error Information",
                "content_stats": "Content Statistics",
                "suggestions": "Suggestions",
            }
            
            for section_key, section_items in payload.summary.sections.items():
                if section_items:  # Only render non-empty sections
                    title = section_titles.get(section_key, section_key.replace("_", " ").title())
                    self._section(pdf, title, section_items)
        else:
            # Legacy format fallback
            if payload.summary.highlights:
                self._section(pdf, "Highlights", payload.summary.highlights)
            if payload.summary.recommendations:
                self._section(pdf, "Recommendations", payload.summary.recommendations)

        # Only show crawl metrics if this was a web crawl (not text/document)
        if payload.metrics and payload.metrics.total_pages > 0 and payload.url != "text-input":
            metrics_lines = [
                f"Pages crawled: {payload.metrics.total_pages}",
                f"Internal links: {payload.metrics.internal_links}",
                f"External links: {payload.metrics.external_links}",
                f"Top keywords: {', '.join(payload.metrics.keywords[:8]) or 'n/a'}",
                f"CTA links detected: {len(payload.metrics.ctas)}",
            ]
            self._section(pdf, "Crawl Metrics", metrics_lines)

        file_name = (payload.summary.overview[:30] or payload.url or "summary").strip()
        safe_name = "".join(ch if ch.isalnum() else "-" for ch in file_name).strip("-") or "summary"
        output_path = self.output_dir / f"{safe_name}-{uuid4().hex[:8]}.pdf"
        pdf.output(str(output_path))
        payload.pdf_path = str(output_path)
        return output_path

    def _write_header(self, pdf: FPDF, payload: ReportPayload) -> None:
        pdf.set_fill_color(32, 44, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(
            0,
            14,
            TITLE,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="C",
            fill=True,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

        pdf.set_font("helvetica", size=11)
        pdf.cell(0, LINE_HEIGHT + 1, f"Source: {payload.url}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        generated = payload.generated_at.strftime("%Y-%m-%d %H:%M UTC")
        pdf.cell(0, LINE_HEIGHT + 1, f"Generated: {generated}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    def _section(
        self, pdf: FPDF, title: str, lines: Iterable[str], *, emphasize: bool = False
    ) -> None:
        pdf.set_fill_color(237, 240, 245)
        pdf.set_font("helvetica", "B", 13)
        pdf.cell(
            0,
            9,
            title,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            fill=True,
        )
        pdf.set_font("helvetica", "B" if emphasize else "", 11)
        self._write_lines(pdf, lines, emphasize=emphasize)
        pdf.ln(SECTION_SPACING)

    def _write_lines(self, pdf: FPDF, lines: Iterable[str], *, emphasize: bool = False) -> None:
        pdf.set_font("helvetica", "B" if emphasize else "", 11)
        sanitized = [self._clean(line) for line in lines if self._clean(line)]
        if not sanitized:
            pdf.multi_cell(0, LINE_HEIGHT, "— (no data available)")
            return

        for line in sanitized:
            prefix = "- " if not emphasize else ""
            pdf.set_x(pdf.l_margin + (0 if emphasize else 2))
            pdf.multi_cell(0, LINE_HEIGHT, f"{prefix}{line}", align="L")

    @staticmethod
    def _clean(text: str | None) -> str:
        if not text:
            return ""
        return " ".join(text.split())
