"""Supervisor (Planner) agent — orchestrates the initial research queries.

This agent replaces the old dynamic LLM router. It now acts as a pure
deterministic planner that runs once at the start of the graph to
decompose the user's research question into specific search queries.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.state import ResearchState
from src.utils import extract_text_content
from src.utils.cache import get_cache
from src.utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini Flash LLM instance."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        temperature=0.1,
        max_output_tokens=1024,
        google_api_key=settings.google_api_key,
    )


async def plan_research(query: str) -> dict[str, Any]:
    """Decompose a research question or detect ambiguity."""
    llm = _build_llm()
    limiter = get_rate_limiter(rpm=settings.gemini.rpm_limit)
    cache = get_cache(
        db_path=settings.cache.db_path,
        search_ttl_hours=settings.cache.search_ttl_hours,
        page_ttl_hours=settings.cache.page_ttl_hours,
        llm_ttl_hours=settings.cache.llm_ttl_hours,
    )

    cache_key = f"plan:{query}"
    cached = await cache.get("llm", cache_key)
    if cached is not None:
        logger.info("plan_research: returning cached queries")
        return cached  # type: ignore[return-value]

    prompt = (
        f"You are a research planner. Evaluate the following user research query.\n"
        f"If the query is highly ambiguous (e.g., 'jaguar' could mean the animal, the car, or the OS) "
        f"and you cannot confidently guess the user's intent, you must ask for clarification.\n"
        f"Otherwise, generate 2-4 diverse, specific search queries that together would comprehensively investigate the topic.\n\n"
        f"Return ONLY a JSON object with EXACTLY this structure, with no markdown fences:\n"
        f"{{\n"
        f"  \"needs_clarification\": true or false,\n"
        f"  \"clarification_question\": \"Ask the user what they meant, if needs_clarification is true. Otherwise empty.\",\n"
        f"  \"queries\": [\"query 1\", \"query 2\"] // if needs_clarification is false\n"
        f"}}\n\n"
        f"Research question: {query}\n\n"
        f"JSON output:"
    )

    async def _invoke():
        response = await llm.ainvoke(prompt)
        return extract_text_content(response.content)

    try:
        raw = await limiter.execute_with_retry(_invoke)
        text = raw.strip()
        
        # Robustly extract JSON
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            parsed = {"needs_clarification": False, "clarification_question": "", "queries": [query]}
    except Exception as exc:
        logger.error(f"plan_research: failed to parse or generate queries: {exc}")
        parsed = {"needs_clarification": False, "clarification_question": "", "queries": [query]}

    needs_clarification = parsed.get("needs_clarification", False)
    clarification_question = parsed.get("clarification_question", "")
    queries = parsed.get("queries", [])
    
    if not needs_clarification:
        # Ensure we have 1-4 valid strings
        queries = [str(q).strip() for q in queries if q][:4]
        if not queries:
            queries = [query]
            
    result = {
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "queries": queries,
    }

    await cache.set("llm", cache_key, result)
    logger.info(f"plan_research: result: {result}")
    return result


async def supervisor(state: ResearchState) -> dict[str, Any]:
    """Act as the initial planner for the research graph."""
    query = state.get("query", "")
    iteration = state.get("iteration", 0)

    # Initial planning phase
    if iteration == 0:
        logger.info("supervisor: planning initial research queries")
        plan_result = await plan_research(query)
        
        if plan_result.get("needs_clarification"):
            return {
                "needs_clarification": True,
                "clarification_question": plan_result["clarification_question"],
                "status": "awaiting_human_review",
                "next": "human_review",
            }
            
        return {
            "search_queries": plan_result["queries"],
            "needs_clarification": False,
            "clarification_question": "",
            "status": "searching",
            "iteration": 1,
            "next": "web_searcher",
        }
    
    # Fallback if somehow called again
    logger.warning("supervisor: called after initialization, routing to writer")
    return {"next": "writer"}
