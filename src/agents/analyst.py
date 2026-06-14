"""Analyst agent — synthesises scraped content into structured findings.

Takes all ``scraped_content`` from state, feeds it to Gemini Flash with
a structured JSON output schema, and extracts key findings (with source
attribution), conflicting information, knowledge gaps, and suggested
follow-up queries. Flags the analysis for human review when significant
conflicts or ambiguity are detected.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.state import AnalysisResult, Finding, PageContent, ResearchState
from src.utils.cache import get_cache
from src.utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """\
You are a senior research analyst. Analyse the following scraped web pages
and produce a structured JSON analysis of the research topic.

Research query: {query}

--- SCRAPED PAGES ---
{pages_text}
--- END PAGES ---

Return a JSON object with EXACTLY this structure (no markdown fences):
{{
  "key_findings": [
    {{
      "claim": "A clear factual claim or insight",
      "evidence": "Supporting evidence or data from the source",
      "source_url": "URL where this was found",
      "confidence": 0.85
    }}
  ],
  "conflicts": [
    "Description of conflicting information found across sources"
  ],
  "knowledge_gaps": [
    "Topics or questions not adequately covered by the sources"
  ],
  "follow_up_queries": [
    "Suggested search queries to fill knowledge gaps"
  ]
}}

Guidelines:
- Include 3-8 key findings, each with a specific source URL.
- Confidence is 0.0-1.0 based on source reliability and corroboration.
- Identify ALL contradictions between sources.
- Note what information is missing or insufficiently covered.
- Suggest 1-3 follow-up queries only if there are real gaps.
- Be precise and factual — do not speculate.

JSON output:"""


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini Flash LLM instance from settings."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        temperature=settings.gemini.temperature,
        max_output_tokens=settings.gemini.max_output_tokens,
        google_api_key=settings.google_api_key,
    )


def _format_pages(pages: list[PageContent]) -> str:
    """Format scraped pages into a text block for the LLM prompt."""
    parts: list[str] = []
    for i, page in enumerate(pages, 1):
        parts.append(
            f"[Source {i}]\n"
            f"Title: {page.title}\n"
            f"URL: {page.url}\n"
            f"Summary:\n{page.summary}\n"
        )
    return "\n---\n".join(parts)


def _parse_analysis_json(raw_text: str) -> dict:
    """Robustly parse the LLM's JSON output.

    Handles common issues like markdown fences wrapping the JSON.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[: -len("```")]

    text = text.strip()
    return json.loads(text)


def _determine_needs_review(analysis: AnalysisResult) -> tuple[bool, str]:
    """Decide whether the analysis needs human review.

    Triggers review when:
    - ≥ 2 conflicting claims are detected.
    - Any finding has confidence below 0.5.
    - ≥ 3 knowledge gaps are identified.

    Returns:
        (needs_review, reason) tuple.
    """
    reasons: list[str] = []

    if len(analysis.conflicts) >= 2:
        reasons.append(
            f"{len(analysis.conflicts)} conflicting claims detected across sources"
        )

    low_confidence = [f for f in analysis.key_findings if f.confidence < 0.5]
    if low_confidence:
        reasons.append(
            f"{len(low_confidence)} findings have low confidence (< 0.5)"
        )

    if len(analysis.knowledge_gaps) >= 3:
        reasons.append(
            f"{len(analysis.knowledge_gaps)} significant knowledge gaps identified"
        )

    if reasons:
        return True, "; ".join(reasons)
    return False, ""


async def analyst(state: ResearchState) -> dict[str, Any]:
    """Analyse all scraped content and produce structured findings.

    Workflow:
        1. Check LLM cache for a prior analysis of the same content set.
        2. Build a prompt with all scraped page summaries.
        3. Call Gemini Flash with rate limiting.
        4. Parse the structured JSON response.
        5. Determine if human review is warranted.
        6. Return serialized AnalysisResult.

    Args:
        state: The shared research state.

    Returns:
        Partial state update with ``analysis``, ``status``,
        ``needs_human_review``, and ``errors``.
    """
    raw_pages = state.get("scraped_content", [])
    query = state.get("query", "")

    if not raw_pages:
        logger.warning("analyst called with no scraped_content")
        return {
            "analysis": AnalysisResult().to_dict(),
            "errors": ["analyst: no scraped content to analyse"],
            "status": "analysis_complete",
        }

    pages = [PageContent.from_dict(p) for p in raw_pages]
    logger.info(f"analyst: analysing {len(pages)} pages for '{query[:60]}'")

    cache = get_cache(
        db_path=settings.cache.db_path,
        search_ttl_hours=settings.cache.search_ttl_hours,
        page_ttl_hours=settings.cache.page_ttl_hours,
        llm_ttl_hours=settings.cache.llm_ttl_hours,
    )

    # ── Cache key from query + page URLs (order-independent) ──
    page_urls_key = "|".join(sorted(p.url for p in pages))
    cache_key = f"analysis:{query}:{page_urls_key}"

    cached = await cache.get("llm", cache_key)
    if cached is not None:
        logger.info("analyst: returning cached analysis")
        return {
            "analysis": cached,
            "status": "analysis_complete",
            "errors": [],
        }

    # ── Build prompt and invoke LLM ──────────────────────────
    pages_text = _format_pages(pages)
    prompt = _ANALYSIS_PROMPT.format(query=query, pages_text=pages_text)

    llm = _build_llm()
    limiter = get_rate_limiter(rpm=settings.gemini.rpm_limit)
    errors: list[str] = []

    try:

        async def _invoke():
            response = await llm.ainvoke(prompt)
            return response.content

        raw_response = await limiter.execute_with_retry(_invoke)
        parsed = _parse_analysis_json(raw_response)

    except json.JSONDecodeError as exc:
        error_msg = f"analyst: failed to parse LLM JSON: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        # Return an empty analysis rather than crashing the pipeline
        return {
            "analysis": AnalysisResult().to_dict(),
            "status": "analysis_complete",
            "errors": errors,
        }
    except Exception as exc:
        error_msg = f"analyst: LLM invocation failed: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "analysis": AnalysisResult().to_dict(),
            "status": "analysis_complete",
            "errors": errors,
        }

    # ── Build AnalysisResult ─────────────────────────────────
    findings = [
        Finding(
            claim=f.get("claim", ""),
            evidence=f.get("evidence", ""),
            source_url=f.get("source_url", ""),
            confidence=float(f.get("confidence", 0.0)),
        )
        for f in parsed.get("key_findings", [])
    ]

    analysis = AnalysisResult(
        key_findings=findings,
        conflicts=parsed.get("conflicts", []),
        knowledge_gaps=parsed.get("knowledge_gaps", []),
        follow_up_queries=parsed.get("follow_up_queries", []),
    )

    needs_review, review_reason = _determine_needs_review(analysis)
    analysis.needs_human_review = needs_review
    analysis.review_reason = review_reason

    analysis_dict = analysis.to_dict()
    await cache.set("llm", cache_key, analysis_dict)

    logger.info(
        f"analyst: {len(findings)} findings, "
        f"{len(analysis.conflicts)} conflicts, "
        f"needs_review={needs_review}"
    )

    return {
        "analysis": analysis_dict,
        "status": "analysis_complete",
        "errors": errors,
    }
