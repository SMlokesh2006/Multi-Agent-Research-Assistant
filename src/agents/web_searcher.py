"""Web Searcher agent — executes search queries via the Tavily API.

Reads ``search_queries`` from state, searches each query through Tavily,
caches results to avoid redundant API calls, deduplicates URLs across
queries, and returns serialized SearchResult dicts.
"""

from __future__ import annotations

import logging
from typing import Any

from tavily import AsyncTavilyClient

from src.config import settings
from src.state import ResearchState, SearchResult
from src.utils.cache import get_cache

logger = logging.getLogger(__name__)


def _build_tavily_client() -> AsyncTavilyClient:
    """Create a Tavily async client from settings."""
    if not settings.tavily.api_key:
        raise ValueError(
            "TAVILY_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return AsyncTavilyClient(api_key=settings.tavily.api_key)


async def web_searcher(state: ResearchState) -> dict[str, Any]:
    """Search the web for each query in ``state['search_queries']``.

    Workflow:
        1. Check cache for each query — skip API call on hit.
        2. Call Tavily search API for cache misses.
        3. Deduplicate results by URL across all queries.
        4. Return partial state update with ``search_results`` and ``status``.

    Args:
        state: The shared research state.

    Returns:
        Partial state update dict with ``search_results``, ``status``,
        and ``errors`` (if any failures occurred).
    """
    queries: list[str] = state.get("search_queries", [])
    if not queries:
        logger.warning("web_searcher called with no search_queries")
        return {
            "search_results": [],
            "errors": ["web_searcher: no search queries provided"],
            "status": "search_complete",
        }

    logger.info(f"web_searcher: searching {len(queries)} queries")

    cache = get_cache(
        db_path=settings.cache.db_path,
        search_ttl_hours=settings.cache.search_ttl_hours,
        page_ttl_hours=settings.cache.page_ttl_hours,
        llm_ttl_hours=settings.cache.llm_ttl_hours,
    )
    client = _build_tavily_client()

    all_results: list[dict] = []
    seen_urls: set[str] = set()
    errors: list[str] = []

    # Collect already-seen URLs from prior results in state
    for existing in state.get("search_results", []):
        seen_urls.add(existing["url"])

    for query in queries:
        try:
            # ── Cache check ──────────────────────────────────
            cached = await cache.get("search", query)
            if cached is not None:
                logger.info(f"web_searcher: cache hit for '{query[:60]}'")
                results_data = cached
            else:
                # ── Tavily API call ──────────────────────────
                logger.info(f"web_searcher: searching Tavily for '{query[:60]}'")
                response = await client.search(
                    query=query,
                    max_results=settings.tavily.max_results,
                    search_depth=settings.tavily.search_depth,
                    include_raw_content=settings.tavily.include_raw_content,
                )
                results_data = response.get("results", [])
                await cache.set("search", query, results_data)

            # ── Parse and deduplicate ────────────────────────
            for item in results_data:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    result = SearchResult(
                        url=url,
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        score=float(item.get("score", 0.0)),
                    )
                    all_results.append(result.to_dict())

        except Exception as exc:
            error_msg = f"web_searcher: error searching '{query[:60]}': {exc}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info(
        f"web_searcher: returning {len(all_results)} unique results "
        f"({len(errors)} errors)"
    )

    return {
        "search_results": all_results,
        "status": "search_complete",
        "errors": errors,
    }
