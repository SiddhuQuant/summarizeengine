from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from webcrawlagent.crawler.extractor import CrawlResult, PageSnapshot

CTA_KEYWORDS = {"contact", "buy", "get", "demo", "signup", "book", "start", "quote"}


@dataclass(slots=True)
class AnalysisSummary:
    root_url: str
    total_pages: int
    internal_links: int
    external_links: int
    top_headings: list[str]
    keywords: list[str]
    ctas: list[str]
    page_summaries: list[dict[str, Any]]


def build_analysis(result: CrawlResult) -> AnalysisSummary:
    page_summaries: list[dict[str, Any]] = []
    all_headings: list[str] = []
    internal_links = 0
    external_links = 0
    keyword_counter: Counter[str] = Counter()
    ctas: set[str] = set()

    root_netloc = urlparse(result.root_url).netloc

    for page in result.pages:
        all_headings.extend(page.headings)
        page_internal, page_external = _link_split(page.links, root_netloc)
        internal_links += len(page_internal)
        external_links += len(page_external)

        keyword_counter.update(_keywords(page.text))
        ctas.update(_cta_candidates(page))

        summary = {
            "url": page.url,
            "title": page.title or "Untitled page",
            "description": page.description,
            "headings": page.headings[:5],
            "word_count": page.word_count,
            "status": page.status,
        }
        page_summaries.append(summary)

    top_headings = all_headings[:10]
    keywords = [word for word, _ in keyword_counter.most_common(12)]

    return AnalysisSummary(
        root_url=result.root_url,
        total_pages=len(result.pages),
        internal_links=internal_links,
        external_links=external_links,
        top_headings=top_headings,
        keywords=keywords,
        ctas=sorted(ctas),
        page_summaries=page_summaries,
    )


def _link_split(links: Iterable[str], root_netloc: str) -> tuple[list[str], list[str]]:
    internal: list[str] = []
    external: list[str] = []
    for link in links:
        netloc = urlparse(link).netloc
        if netloc == root_netloc:
            internal.append(link)
        else:
            external.append(link)
    return internal, external


def _keywords(text: str) -> Iterable[str]:
    tokens = [token.lower() for token in text.split()]
    return [token for token in tokens if token.isalpha() and len(token) > 3]


def _cta_candidates(page: PageSnapshot) -> Iterable[str]:
    matches: set[str] = set()
    for link in page.links:
        lower = link.lower()
        if any(keyword in lower for keyword in CTA_KEYWORDS):
            matches.add(link)
    return matches
