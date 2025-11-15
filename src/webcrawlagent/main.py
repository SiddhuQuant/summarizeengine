from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from webcrawlagent.app.api import router as agent_router
from webcrawlagent.app.dependencies import shutdown_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await shutdown_service()


def create_app() -> FastAPI:
    app = FastAPI(title="Web Crawl Agent", version="0.1.0", lifespan=lifespan)
    app.include_router(agent_router)

    static_dir = Path(__file__).parent / "app" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))

    return app


app = create_app()
