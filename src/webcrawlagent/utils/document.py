from __future__ import annotations

import io
from pathlib import Path

from fastapi import UploadFile


async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text content from an uploaded file."""
    content = await file.read()
    file_extension = Path(file.filename or "").suffix.lower()

    # Handle text files
    if file_extension in [".txt", ".md", ".markdown"]:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("latin-1")
            except UnicodeDecodeError:
                return content.decode("utf-8", errors="replace")

    # Handle PDF files
    if file_extension == ".pdf":
        try:
            import PyPDF2  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "PDF extraction requires PyPDF2. Install it with: pip install PyPDF2"
            )
        
        pdf_file = io.BytesIO(content)
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n\n".join(text_parts)
        except Exception as exc:
            raise RuntimeError(f"Failed to extract text from PDF: {exc}") from exc

    # For other file types, try to decode as text
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return content.decode("latin-1")
        except UnicodeDecodeError:
            raise RuntimeError(
                f"Unsupported file type: {file_extension}. "
                "Supported types: .txt, .md, .pdf"
            )

