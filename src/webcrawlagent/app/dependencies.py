from __future__ import annotations

from functools import lru_cache

from webcrawlagent.app.service import CrawlAgentService
from webcrawlagent.config import get_settings


@lru_cache(maxsize=1)
def get_service() -> CrawlAgentService:
    settings = get_settings()
    return CrawlAgentService(settings)


async def shutdown_service() -> None:
    service = get_service()
    await service.shutdown()
