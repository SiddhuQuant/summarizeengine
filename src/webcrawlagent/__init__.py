from __future__ import annotations

import asyncio
import sys

if sys.platform.startswith("win"):
    # Playwright requires subprocess support, so enforce the Proactor loop on Windows.
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        # Fallback for very old Python builds.
        pass

__all__ = []
