"""Content Reader agent — extracts and summarizes web page content.

Reads ``search_results`` from state, extracts full page text via the
Tavily Extract API (with httpx + BeautifulSoup fallback), then uses
Gemini Flash to produce a concise summary of each page. Results are
cached to minimise redundant API calls.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import AsyncTavilyClient

from src.config import settings
from src.state import PageContent, ResearchState, SearchResult
from src.utils.cache import get_cache
from src.utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# Maximum number of URLs to process per invocation (saves API credits).
_MAX_URLS = 5

_SUMMARISE_PROMPT = """\
You are a research assistant. Summarise the following web page content in
3-5 concise paragraphs. Preserve all important facts, statistics, and
quotes. Include the source context.

Title: {title}
URL: {url}

Content:
{content}

Provide a clear, information-dense summary:"""


# ── Helpers ──────────────────────────────────────────────────


def _build_tavily_client() -> AsyncTavilyClient:
    """Create a Tavily async client from settings."""
    if not settings.tavily.api_key:
        raise ValueError(
            "TAVILY_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return AsyncTavilyClient(api_key=settings.tavily.api_key)


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini Flash LLM instance from settings."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        temperature=settings.gemini.temperature,
        max_output_tokens=settings.gemini.max_output_tokens,
        google_api_key=settings.google_api_key,
    )


async def _extract_with_tavily(
    client: AsyncTavilyClient, urls: list[str]
) -> dict[str, dict]:
    """Attempt to extract page content via Tavily Extract API.

    Args:
        client: The Tavily async client.
        urls: List of URLs to extract.

    Returns:
        Mapping of URL → {"title": ..., "raw_content": ...} for
        successfully extracted pages.
    """
    extracted: dict[str, dict] = {}
    try:
        response = await client.extract(urls=urls)
        for item in response.get("results", []):
            url = item.get("url", "")
            if url and item.get("raw_content"):
                extracted[url] = {
                    "title": item.get("title", ""),
                    "raw_content": item["raw_content"],
                }
    except Exception as exc:
        logger.warning(f"Tavily extract failed: {exc}")
    return extracted


async def _extract_with_httpx(url: str) -> dict | None:
    """Fallback extraction using httpx + BeautifulSoup.

    Returns:
        Dict with ``title`` and ``raw_content``, or None on failure.
    """
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "ResearchAssistant/1.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator="\n", strip=True)

        # Basic length guard — too short means extraction likely failed
        if len(text) < 100:
            return None

        # Truncate very long pages to ~15 000 chars to stay within LLM context
        return {"title": title, "raw_content": text[:15_000]}

    except Exception as exc:
        logger.warning(f"httpx extraction failed for {url}: {exc}")
        return None


async def _summarise_content(
    llm: ChatGoogleGenerativeAI,
    url: str,
    title: str,
    raw_content: str,
) -> str:
    """Use Gemini Flash to summarise extracted page content.

    Rate-limited to stay within free-tier RPM.
    """
    limiter = get_rate_limiter(rpm=settings.gemini.rpm_limit)
    prompt = _SUMMARISE_PROMPT.format(
        title=title,
        url=url,
        content=raw_content[:12_000],  # trim for token budget
    )

    async def _invoke():
        response = await llm.ainvoke(prompt)
        return response.content

    return await limiter.execute_with_retry(_invoke)


# ── Main Node Function ───────────────────────────────────────


async def content_reader(state: ResearchState) -> dict[str, Any]:
    """Extract and summarise content for the top search results.

    Workflow:
        1. Select the top ``_MAX_URLS`` unique URLs from search results.
        2. Check cache — skip extraction for cached pages.
        3. Attempt Tavily Extract for all uncached URLs in one batch.
        4. Fall back to httpx + BeautifulSoup for any URLs Tavily missed.
        5. Summarise each page with Gemini Flash (rate-limited).
        6. Cache and return serialized PageContent dicts.

    Args:
        state: The shared research state.

    Returns:
        Partial state update with ``scraped_content``, ``status``,
        and ``errors``.
    """
    raw_results = state.get("search_results", [])
    if not raw_results:
        logger.warning("content_reader called with no search_results")
        return {
            "scraped_content": [],
            "errors": ["content_reader: no search results to read"],
            "status": "content_read_complete",
        }

    # ── Select top URLs ──────────────────────────────────────
    search_results = [SearchResult.from_dict(r) for r in raw_results]
    # Sort by score descending, pick top N unique URLs
    search_results.sort(key=lambda r: r.score, reverse=True)

    selected_urls: list[tuple[str, str]] = []  # (url, title)
    seen: set[str] = set()
    # Also exclude URLs we've already scraped in prior iterations
    for existing in state.get("scraped_content", []):
        seen.add(existing["url"])

    for sr in search_results:
        if sr.url not in seen:
            seen.add(sr.url)
            selected_urls.append((sr.url, sr.title))
        if len(selected_urls) >= _MAX_URLS:
            break

    if not selected_urls:
        logger.info("content_reader: all URLs already scraped")
        return {
            "scraped_content": [],
            "status": "content_read_complete",
            "errors": [],
        }

    logger.info(f"content_reader: processing {len(selected_urls)} URLs")

    cache = get_cache(
        db_path=settings.cache.db_path,
        search_ttl_hours=settings.cache.search_ttl_hours,
        page_ttl_hours=settings.cache.page_ttl_hours,
        llm_ttl_hours=settings.cache.llm_ttl_hours,
    )
    llm = _build_llm()
    client = _build_tavily_client()

    pages: list[dict] = []
    errors: list[str] = []
    urls_needing_extraction: list[str] = []
    url_title_map: dict[str, str] = {}

    # ── Cache check ──────────────────────────────────────────
    for url, title in selected_urls:
        url_title_map[url] = title
        cached = await cache.get("page", url)
        if cached is not None:
            logger.info(f"content_reader: cache hit for {url[:80]}")
            pages.append(cached)
        else:
            urls_needing_extraction.append(url)

    if not urls_needing_extraction:
        logger.info("content_reader: all pages served from cache")
        return {
            "scraped_content": pages,
            "status": "content_read_complete",
            "errors": [],
        }

    # ── Tavily batch extraction ──────────────────────────────
    tavily_results = await _extract_with_tavily(client, urls_needing_extraction)
    remaining_urls = [u for u in urls_needing_extraction if u not in tavily_results]

    # ── Fallback extraction for misses ───────────────────────
    httpx_results: dict[str, dict] = {}
    for url in remaining_urls:
        result = await _extract_with_httpx(url)
        if result is not None:
            httpx_results[url] = result

    # ── Merge and summarise ──────────────────────────────────
    all_extracted = {**tavily_results, **httpx_results}

    for url in urls_needing_extraction:
        if url not in all_extracted:
            error_msg = f"content_reader: failed to extract content from {url[:80]}"
            logger.warning(error_msg)
            errors.append(error_msg)
            continue

        data = all_extracted[url]
        title = data.get("title") or url_title_map.get(url, "")
        raw_content = data["raw_content"]
        extraction_method = "tavily" if url in tavily_results else "httpx"

        try:
            summary = await _summarise_content(llm, url, title, raw_content)
        except Exception as exc:
            error_msg = (
                f"content_reader: summarisation failed for {url[:80]}: {exc}"
            )
            logger.error(error_msg)
            errors.append(error_msg)
            # Use a truncated raw content snippet as a fallback summary
            summary = raw_content[:1000] + "..."

        page = PageContent(
            url=url,
            title=title,
            raw_content=raw_content[:10_000],  # trim for state size
            summary=summary,
            word_count=len(raw_content.split()),
            extraction_method=extraction_method,
        )
        page_dict = page.to_dict()
        await cache.set("page", url, page_dict)
        pages.append(page_dict)

    logger.info(
        f"content_reader: returning {len(pages)} pages ({len(errors)} errors)"
    )

    return {
        "scraped_content": pages,
        "status": "content_read_complete",
        "errors": errors,
    }
