"""Playwright scraping helpers.

Inside Jupyter/VS Code notebooks, always use the *async* function with
top-level ``await`` — the notebook event loop is already running, so the
sync Playwright API (and ``scrape_article_text_sync``) would raise there.
"""

import asyncio

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


async def scrape_article_text(url: str) -> str:
    """Uses Playwright in headless mode to retrieve a page, stripping scripts/nav elements."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=15000)
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            for bad_tag in soup(
                ["script", "style", "nav", "footer", "header", "aside"]
            ):
                bad_tag.decompose()
            text = soup.get_text(separator=" ")
            cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(cleaned_lines)
        except Exception as e:
            return f"Failed scraping URL: {str(e)}"
        finally:
            await browser.close()


def scrape_article_text_sync(url: str) -> str:
    """Blocking wrapper for plain Python scripts.

    Do NOT call from a notebook — Jupyter already runs an event loop, so
    ``asyncio.run`` raises. In notebooks use ``await scrape_article_text(url)``.
    """
    return asyncio.run(scrape_article_text(url))
