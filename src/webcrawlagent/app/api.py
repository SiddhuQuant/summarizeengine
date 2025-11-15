from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from sse_starlette.sse import EventSourceResponse

from webcrawlagent.app.dependencies import get_service
from webcrawlagent.app.service import CrawlAgentService, ServiceResult
from webcrawlagent.config import get_settings
from webcrawlagent.utils.document import extract_text_from_file

router = APIRouter(prefix="/api", tags=["agent"])


class AnalyzeRequest(BaseModel):
    url: HttpUrl


class AnalyzeTextRequest(BaseModel):
    text: str
    document_type: str = "document"


class AnalyzeResponse(BaseModel):
    url: HttpUrl | str
    summary: dict
    metrics: dict
    pdf_path: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest, service: CrawlAgentService = Depends(get_service)  # noqa: B008
):
    try:
        result = await service.run(str(request.url))
    except Exception as exc:  # pragma: no cover - network/LLM errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _serialize_result(result)


@router.post("/analyze-text", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeTextRequest, service: CrawlAgentService = Depends(get_service)  # noqa: B008
):
    """Analyze raw text input with an optional document type."""
    try:
        result = await service.run_from_text(request.text, request.document_type)
    except Exception as exc:  # pragma: no cover - network/LLM errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _serialize_result(result)


@router.post("/analyze-document", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(...),  # noqa: B008
    document_type: str = Form("document"),  # noqa: B008
    service: CrawlAgentService = Depends(get_service),  # noqa: B008
):
    """Analyze an uploaded document file."""
    try:
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or could not be extracted")
        result = await service.run_from_text(text, document_type)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network/LLM errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _serialize_result(result)


@router.get("/stream")
async def stream(
    url: HttpUrl = Query(..., description="Website to analyze"),  # noqa: B008
    service: CrawlAgentService = Depends(get_service),  # noqa: B008
):
    async def event_generator():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def progress(message: str):
            await queue.put({"type": "status", "message": message})

        task = asyncio.create_task(service.run(str(url), progress))

        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.2)
                    yield {"event": "message", "data": json.dumps(event)}
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue
                except asyncio.CancelledError:
                    task.cancel()
                    break

            try:
                result = await task
            except asyncio.CancelledError:
                return
            except Exception as exc:  # pragma: no cover
                yield {"event": "message", "data": json.dumps({"type": "error", "message": str(exc)})}
                return
            summary_payload = {"type": "summary", **_serialize_result(result)}
            yield {"event": "message", "data": json.dumps(summary_payload)}
        except asyncio.CancelledError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise

    return EventSourceResponse(event_generator())


def _serialize_result(result: ServiceResult) -> dict:
    file_name = Path(result.pdf_path).name
    metrics_dict = asdict(result.analysis) if result.analysis else {
        "root_url": result.url,
        "total_pages": 1,
        "internal_links": 0,
        "external_links": 0,
        "top_headings": [],
        "keywords": [],
        "ctas": [],
        "page_summaries": [],
    }
    return {
        "url": result.url,
        "summary": {
            "overview": result.summary.overview,
            "content_type": result.summary.content_type,
            "sections": result.summary.sections,
            "highlights": result.summary.highlights,
            "recommendations": result.summary.recommendations,
        },
        "metrics": metrics_dict,
        "pdf_path": f"/api/reports/{file_name}",
    }


@router.get("/reports/{file_name}")  # pragma: no cover - exercised via UI/manual tests
async def download_report(file_name: str):
    settings = get_settings()
    report_dir = settings.ensure_report_dir()
    safe_path = (report_dir / file_name).resolve()
    if not str(safe_path).startswith(str(report_dir.resolve())) or not safe_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(safe_path, media_type="application/pdf", filename=file_name)
