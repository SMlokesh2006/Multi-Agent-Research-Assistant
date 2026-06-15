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
        raise ValueError("TAVILY_API_KEY not set. Copy .env.example to .env and add your key.")
    return AsyncTavilyClient(api_key=settings.tavily.api_key)


async def web_searcher(state: ResearchState | str | dict) -> dict[str, Any]:
    """Search the web for one or more queries.

    This agent supports both sequential and parallel (fan-out) execution.
    If called with a string, it searches that single query.
    If called with ResearchState, it searches all ``state['search_queries']``.

    Args:
        state: ResearchState dict, or a single query string, or a dict
               containing 'query'.

    Returns:
        Partial state update dict with ``search_results``.
    """
    # ── Parse input ──────────────────────────────────────────
    if isinstance(state, str):
        queries = [state]
    elif isinstance(state, dict) and "query" in state and "search_queries" not in state:
        queries = [state["query"]]
    else:
        queries = state.get("search_queries", [])

    if not queries:
        logger.warning("web_searcher called with no search_queries")
        return {"search_results": []}

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

    # If we have the full state, we can avoid duplicates from prior search batches
    if isinstance(state, dict):
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

    logger.info(f"web_searcher: returning {len(all_results)} unique results")

    return {
        "search_results": all_results,
        "status": "search_complete",
        "errors": errors,
    }
