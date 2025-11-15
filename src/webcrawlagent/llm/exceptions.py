from __future__ import annotations


class LLMContentError(RuntimeError):
    """Raised when an LLM response cannot be converted into a usable summary."""

    def __init__(self, message: str, *, payload: dict | None = None):
        super().__init__(message)
        self.payload = payload or {}

