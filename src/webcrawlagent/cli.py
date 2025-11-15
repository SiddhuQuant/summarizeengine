from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from rich.console import Console

from webcrawlagent.app.service import CrawlAgentService
from webcrawlagent.config import get_settings

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the web crawl agent once from the CLI")
    parser.add_argument("--url", required=True, help="Website to crawl")
    parser.add_argument("--out", help="Optional PDF destination path")
    return parser


async def _async_main(url: str, out: str | None) -> int:
    settings = get_settings()
    service = CrawlAgentService(settings)
    try:
        result = await service.run(url)
    finally:
        await service.shutdown()

    console.print(f"[bold green]Overview:[/bold green] {result.summary.overview}")
    for section in result.summary.sections:
        console.print(f"  [cyan]-[/cyan] {section}")

    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(result.pdf_path, target)
        console.print(f"Report copied to {target}")
    else:
        console.print(f"Report saved at {result.pdf_path}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args.url, args.out))


if __name__ == "__main__":
    main()
