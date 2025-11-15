from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    llm_provider: Literal["gemini", "grok"] = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    grok_api_key: str | None = Field(default=None, alias="GROK_API_KEY")
    grok_model: str = Field(default="grok-2-latest", alias="GROK_MODEL")
    crawl_max_pages: int = Field(default=3, ge=1, alias="CRAWL_MAX_PAGES")
    crawl_max_tokens: int = Field(default=4000, ge=1000, alias="CRAWL_MAX_TOKENS")
    crawl_timeout: int = Field(default=45, ge=10, alias="CRAWL_TIMEOUT")
    crawl_delay: float = Field(default=1.0, ge=0.0, alias="CRAWL_DELAY_SECONDS")
    playwright_headless: bool = Field(default=True, alias="PLAYWRIGHT_HEADLESS")
    report_output_dir: Path = Field(default=Path("reports"), alias="REPORT_OUTPUT_DIR")
    log_level: Literal["info", "debug"] = Field(default="info", alias="LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    def ensure_report_dir(self) -> Path:
        self.report_output_dir.mkdir(parents=True, exist_ok=True)
        return self.report_output_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
