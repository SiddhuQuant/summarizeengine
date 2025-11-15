from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from webcrawlagent.config import Settings
from webcrawlagent.crawler.session import BrowserSession

ProgressHook = Callable[[str], Coroutine[None, None, None]]


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.2))


def _clean_text(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed.strip()


@dataclass(slots=True)
class PageSnapshot:
    url: str
    title: str
    description: str
    headings: list[str]
    links: list[str]
    text: str
    word_count: int
    token_estimate: int
    status: str = "ok"

    def trimmed_text(self, max_tokens: int) -> str:
        if self.token_estimate <= max_tokens:
            return self.text
        words = self.text.split()
        allowed = int(max_tokens / max(self.token_estimate, 1) * len(words))
        return " ".join(words[:max(allowed, 1)]) + "..."


@dataclass(slots=True)
class CrawlResult:
    root_url: str
    pages: list[PageSnapshot] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(page.token_estimate for page in self.pages)

    def aggregate_text(self, max_tokens: int) -> list[str]:
        remaining = max_tokens
        chunks: list[str] = []
        for page in self.pages:
            if remaining <= 0:
                break
            allowance = min(page.token_estimate, remaining)
            chunks.append(f"URL: {page.url}\n{page.trimmed_text(allowance)}")
            remaining -= allowance
        return chunks


async def crawl_site(
    url: str,
    session: BrowserSession,
    settings: Settings,
    progress: ProgressHook | None = None,
) -> CrawlResult:
    root = url.rstrip("/")
    parsed_root = urlparse(root)
    queue: deque[str] = deque([root])
    seen: set[str] = set()
    pages: list[PageSnapshot] = []

    async def emit(message: str) -> None:
        if progress:
            await progress(message)

    while queue and len(pages) < settings.crawl_max_pages:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        await emit(f"Visiting {current}")
        page = await session.new_page()
        try:
            response = await page.goto(current, wait_until="networkidle")
            status = str(response.status) if response else "unknown"
            html = await page.content()
            text = await page.inner_text("body")
        except Exception as exc:  # pragma: no cover - network instability
            await emit(f"Failed to load {current}: {exc}")
            await page.close()
            continue
        finally:
            await page.close()

        soup = BeautifulSoup(html, "html.parser")
        title = _clean_text(soup.title.string) if soup.title and soup.title.string else ""
        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag and description_tag.get("content"):
            description = _clean_text(description_tag["content"])
        else:
            description = ""
        headings = [
            _clean_text(h.get_text(" ", strip=True)) for h in soup.find_all(["h1", "h2", "h3"])
        ]
        headings = [h for h in headings if h]
        links = [_normalize_link(a.get("href"), current) for a in soup.find_all("a", href=True)]
        links = [link for link in links if link]
        cleaned_text = _clean_text(text)
        token_estimate = _approx_tokens(cleaned_text)
        page_snapshot = PageSnapshot(
            url=current,
            title=title,
            description=description,
            headings=headings[:30],
            links=links,
            text=cleaned_text,
            word_count=len(cleaned_text.split()),
            token_estimate=token_estimate,
            status=status,
        )
        pages.append(page_snapshot)

        internal_links = _internal_links(links, parsed_root.netloc)
        for link in internal_links:
            if link not in seen and len(queue) < settings.crawl_max_pages * 3:
                queue.append(link)

        if settings.crawl_delay:
            await asyncio.sleep(settings.crawl_delay)

    return CrawlResult(root_url=root, pages=pages)


def _normalize_link(href: str | None, base_url: str) -> str | None:
    if not href:
        return None
    if href.startswith("javascript:"):
        return None
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    return absolute.split("#")[0]


def _internal_links(links: Iterable[str], netloc: str) -> list[str]:
    internal: list[str] = []
    for link in links:
        parsed = urlparse(link)
        if parsed.netloc == netloc:
            internal.append(link)
    return internal
