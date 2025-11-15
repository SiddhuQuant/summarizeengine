from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from webcrawlagent.config import Settings
from webcrawlagent.crawler import BrowserSession, crawl_site
from webcrawlagent.crawler.analyzer import AnalysisSummary, build_analysis
from webcrawlagent.crawler.extractor import CrawlResult
from webcrawlagent.llm.exceptions import LLMContentError
from webcrawlagent.llm.factory import create_llm_client
from webcrawlagent.llm.summary import build_fallback_summary
from webcrawlagent.report.builder import PdfReportBuilder
from webcrawlagent.report.models import ReportPayload, SiteSummary

ProgressHook = Callable[[str], Coroutine[None, None, None]]


@dataclass(slots=True)
class ServiceResult:
    url: str
    crawl: CrawlResult | None
    analysis: AnalysisSummary | None
    summary: SiteSummary
    pdf_path: str


class CrawlAgentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = create_llm_client(settings)
        self.report_builder = PdfReportBuilder(settings)

    async def run(self, url: str, progress: ProgressHook | None = None) -> ServiceResult:
        async def emit(message: str):
            if progress:
                await progress(message)

        await emit("Launching headless browser")
        async with BrowserSession(self.settings) as session:
            crawl = await crawl_site(url, session, self.settings, progress)
        await emit("Crawl complete; building metadata")
        analysis = build_analysis(crawl)
        await emit("Calling Gemini for summary")
        try:
            summary = await self.llm.summarize_site(crawl, analysis)
        except LLMContentError as exc:
            await emit("LLM blocked the content; using crawler-only summary")
            summary = build_fallback_summary(crawl, analysis, reason=str(exc))
        await emit("Generating PDF report")
        payload = ReportPayload(url=url, summary=summary, metrics=analysis)
        pdf_path = str(self.report_builder.build(payload))
        await emit("Report saved")
        return ServiceResult(
            url=url,
            crawl=crawl,
            analysis=analysis,
            summary=summary,
            pdf_path=pdf_path,
        )

    async def run_from_text(
        self, text: str, document_type: str = "document", progress: ProgressHook | None = None
    ) -> ServiceResult:
        """Summarize raw text content without crawling."""
        async def emit(message: str):
            if progress:
                await progress(message)

        await emit("Analyzing text content")
        try:
            summary = await self.llm.summarize_text(text, document_type)
        except LLMContentError as exc:
            await emit("LLM blocked the content; generating basic summary")
            # Create a simple fallback summary for content blocking
            summary = SiteSummary(
                overview=f"Text analysis unavailable ({exc}). Content length: {len(text)} characters.",
                content_type="document",
                sections={
                    "error_info": ["Content could not be analyzed by LLM"],
                    "content_stats": [f"Text length: {len(text)} characters"],
                    "suggestions": ["Try with different content or check LLM configuration"],
                },
            )
        # Let RuntimeError and other exceptions propagate so API can return proper error messages
        
        await emit("Generating PDF report")
        # Create a minimal analysis for text-only summaries
        from webcrawlagent.crawler.analyzer import AnalysisSummary
        dummy_analysis = AnalysisSummary(
            root_url="text-input",
            total_pages=1,
            internal_links=0,
            external_links=0,
            top_headings=[],
            keywords=[],
            ctas=[],
            page_summaries=[],
        )
        payload = ReportPayload(url="text-input", summary=summary, metrics=dummy_analysis)
        pdf_path = str(self.report_builder.build(payload))
        await emit("Report saved")
        return ServiceResult(
            url="text-input",
            crawl=None,
            analysis=dummy_analysis,
            summary=summary,
            pdf_path=pdf_path,
        )

    async def shutdown(self) -> None:
        await self.llm.aclose()
