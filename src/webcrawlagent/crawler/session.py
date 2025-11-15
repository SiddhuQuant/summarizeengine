from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Callable, TypeVar

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from webcrawlagent.config import Settings

_T = TypeVar("_T")


class BrowserSession:
    """Manages a Playwright browser/context lifecycle."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> BrowserSession:
        self._loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
        try:
            await self._run(self._start)
        except Exception:
            self._shutdown_executor()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - cleanup safety
        if self._executor:
            try:
                await self._run(self._stop)
            finally:
                self._shutdown_executor()
        else:
            self._cleanup_sync()

    async def new_page(self) -> "AsyncPage":
        if not self._context:
            raise RuntimeError("BrowserSession is not running")
        page = await self._run(self._context.new_page)
        return AsyncPage(self, page)

    async def _run(self, func: Callable[..., _T], /, *args, **kwargs) -> _T:
        if not self._executor or not self._loop:
            raise RuntimeError("BrowserSession is not running")
        call = partial(func, *args, **kwargs)
        return await self._loop.run_in_executor(self._executor, call)

    def _start(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.settings.playwright_headless
        )
        self._context = self._browser.new_context()
        timeout_ms = self.settings.crawl_timeout * 1000
        self._context.set_default_navigation_timeout(timeout_ms)
        self._context.set_default_timeout(timeout_ms)

    def _stop(self) -> None:
        self._cleanup_sync()

    def _cleanup_sync(self) -> None:
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def _shutdown_executor(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._loop = None


class AsyncPage:
    """Async wrapper over a sync Playwright page bound to a single thread."""

    def __init__(self, session: BrowserSession, page: Page):
        self._session = session
        self._page = page

    def __getattr__(self, name):
        attr = getattr(self._page, name)
        if callable(attr):
            async def _method(*args, **kwargs):
                return await self._session._run(attr, *args, **kwargs)

            return _method
        return attr


@asynccontextmanager
async def browser_session(settings: Settings):
    session = BrowserSession(settings)
    await session.__aenter__()
    try:
        yield session
    finally:
        await session.__aexit__(None, None, None)
