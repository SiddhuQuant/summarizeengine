from __future__ import annotations

from webcrawlagent.config import Settings
from webcrawlagent.crawler.analyzer import AnalysisSummary, build_analysis
from webcrawlagent.crawler.extractor import CrawlResult, PageSnapshot
from webcrawlagent.report.builder import PdfReportBuilder
from webcrawlagent.report.models import ReportPayload, SiteSummary


def make_page(url: str, text: str) -> PageSnapshot:
    return PageSnapshot(
        url=url,
        title="Title",
        description="Description",
        headings=["Heading"],
        links=[url + "/contact", "https://external.com"],
        text=text,
        word_count=len(text.split()),
        token_estimate=len(text.split()),
        status="200",
    )


def test_build_analysis_counts_links():
    result = CrawlResult(
        root_url="https://example.com",
        pages=[
            make_page("https://example.com", "This is sample text for insights"),
            make_page(
                "https://example.com/about",
                "Another section talking about contact actions",
            ),
        ],
    )
    analysis = build_analysis(result)
    assert analysis.total_pages == 2
    assert analysis.internal_links >= 2
    assert "contact" in " ".join(analysis.ctas).lower()


def test_pdf_report_builder_creates_file(tmp_path):
    settings = Settings(GEMINI_API_KEY="dummy", REPORT_OUTPUT_DIR=tmp_path)
    builder = PdfReportBuilder(settings)
    analysis = AnalysisSummary(
        root_url="https://example.com",
        total_pages=1,
        internal_links=1,
        external_links=0,
        top_headings=["Heading"],
        keywords=["example"],
        ctas=["https://example.com/contact"],
        page_summaries=[],
    )
    summary = SiteSummary(
        overview="Example overview",
        sections=["Section"],
        highlights=["Highlight"],
        recommendations=["Recommendation"],
    )
    payload = ReportPayload(url="https://example.com", summary=summary, metrics=analysis)
    pdf_path = builder.build(payload)
    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
