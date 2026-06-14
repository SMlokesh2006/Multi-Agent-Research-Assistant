"""Writer agent — generates a structured markdown research report.

Consumes the ``analysis`` and ``scraped_content`` from state and uses
Gemini Flash to produce a polished report with inline citations,
an executive summary, and clearly labelled sections.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.state import AnalysisResult, PageContent, ResearchState
from src.utils.cache import get_cache
from src.utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

_REPORT_PROMPT = """\
You are a professional research writer. Using the structured analysis and
source material below, write a comprehensive markdown research report.

Research query: {query}

--- ANALYSIS ---
Key Findings:
{findings_text}

Conflicting Views:
{conflicts_text}

Knowledge Gaps:
{gaps_text}
--- END ANALYSIS ---

--- SOURCES ---
{sources_text}
--- END SOURCES ---

Write the report with EXACTLY these sections:

# Research Report: {query}

## Executive Summary
A 2-3 paragraph overview of the key findings and conclusions.

## Key Findings
Detailed discussion of each finding. Use inline citations like [1], [2]
referring to the numbered sources list. Each finding should be a
subsection (### heading) with supporting evidence.

## Detailed Analysis
Deeper exploration of the topic, synthesising information across sources.

## Conflicting Views
Discussion of any contradictions or disagreements found across sources.
If none, note that sources were largely consistent.

## Limitations & Gaps
What the research could not fully answer, and suggestions for further
investigation.

## Sources
Numbered list of all sources used:
[1] Title — URL
[2] Title — URL
...

Guidelines:
- Be factual, precise, and well-structured.
- Use inline citations [N] throughout the text.
- Highlight data, statistics, and direct quotes where available.
- Keep the tone professional but accessible.
- The report should be 800-1500 words.

Write the full report now:"""


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini Flash LLM instance from settings."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        temperature=settings.gemini.temperature,
        max_output_tokens=settings.gemini.max_output_tokens,
        google_api_key=settings.google_api_key,
    )


def _format_findings(analysis: AnalysisResult, pages: list[PageContent]) -> str:
    """Format key findings with source references for the prompt."""
    url_to_index = {p.url: i + 1 for i, p in enumerate(pages)}
    parts: list[str] = []
    for f in analysis.key_findings:
        source_idx = url_to_index.get(f.source_url, "?")
        parts.append(
            f"- [{source_idx}] {f.claim}\n"
            f"  Evidence: {f.evidence}\n"
            f"  Confidence: {f.confidence:.0%}"
        )
    return "\n".join(parts) if parts else "No key findings available."


def _format_sources(pages: list[PageContent]) -> str:
    """Format the numbered source list for the prompt."""
    return "\n".join(
        f"[{i}] {p.title} — {p.url}\n    Summary: {p.summary[:300]}"
        for i, p in enumerate(pages, 1)
    )


async def writer(state: ResearchState) -> dict[str, Any]:
    """Generate a structured markdown report from analysis results.

    Workflow:
        1. Check LLM cache for a prior report on the same analysis.
        2. Build a detailed prompt with findings, conflicts, gaps, sources.
        3. Call Gemini Flash with rate limiting.
        4. Return the report string.

    Args:
        state: The shared research state.

    Returns:
        Partial state update with ``report``, ``status``, and ``errors``.
    """
    analysis_dict = state.get("analysis")
    raw_pages = state.get("scraped_content", [])
    query = state.get("query", "")

    if not analysis_dict:
        logger.warning("writer called with no analysis")
        return {
            "report": "",
            "errors": ["writer: no analysis available to write from"],
            "status": "report_complete",
        }

    analysis = AnalysisResult.from_dict(analysis_dict)
    pages = [PageContent.from_dict(p) for p in raw_pages]

    logger.info(f"writer: generating report for '{query[:60]}'")

    cache = get_cache(
        db_path=settings.cache.db_path,
        search_ttl_hours=settings.cache.search_ttl_hours,
        page_ttl_hours=settings.cache.page_ttl_hours,
        llm_ttl_hours=settings.cache.llm_ttl_hours,
    )

    # ── Cache key from query + analysis hash ─────────────────
    page_urls_key = "|".join(sorted(p.url for p in pages))
    cache_key = f"report:{query}:{page_urls_key}"

    cached = await cache.get("llm", cache_key)
    if cached is not None:
        logger.info("writer: returning cached report")
        # cached value is stored as {"report": "..."} dict
        return {
            "report": cached.get("report", ""),
            "status": "report_complete",
            "errors": [],
        }

    # ── Build prompt ─────────────────────────────────────────
    findings_text = _format_findings(analysis, pages)
    conflicts_text = (
        "\n".join(f"- {c}" for c in analysis.conflicts)
        if analysis.conflicts
        else "No significant conflicts detected."
    )
    gaps_text = (
        "\n".join(f"- {g}" for g in analysis.knowledge_gaps)
        if analysis.knowledge_gaps
        else "No major knowledge gaps identified."
    )
    sources_text = _format_sources(pages)

    prompt = _REPORT_PROMPT.format(
        query=query,
        findings_text=findings_text,
        conflicts_text=conflicts_text,
        gaps_text=gaps_text,
        sources_text=sources_text,
    )

    llm = _build_llm()
    limiter = get_rate_limiter(rpm=settings.gemini.rpm_limit)
    errors: list[str] = []

    try:

        async def _invoke():
            response = await llm.ainvoke(prompt)
            return response.content

        report = await limiter.execute_with_retry(_invoke)

    except Exception as exc:
        error_msg = f"writer: LLM invocation failed: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)

        # Generate a minimal fallback report from the analysis data
        report = _build_fallback_report(query, analysis, pages)

    await cache.set("llm", cache_key, {"report": report})

    logger.info(f"writer: report generated ({len(report)} chars)")

    return {
        "report": report,
        "status": "report_complete",
        "errors": errors,
    }


def _build_fallback_report(
    query: str,
    analysis: AnalysisResult,
    pages: list[PageContent],
) -> str:
    """Build a minimal report when the LLM fails.

    Ensures the user always gets *something* readable even if report
    generation errors out.
    """
    sections: list[str] = [f"# Research Report: {query}\n"]
    sections.append("## Executive Summary\n")
    sections.append(
        "*Report generation encountered an error. "
        "Below is a summary of the raw analysis.*\n"
    )

    sections.append("## Key Findings\n")
    for f in analysis.key_findings:
        sections.append(f"- **{f.claim}** (confidence: {f.confidence:.0%})")
        sections.append(f"  - {f.evidence}")
        sections.append(f"  - Source: {f.source_url}\n")

    if analysis.conflicts:
        sections.append("## Conflicting Views\n")
        for c in analysis.conflicts:
            sections.append(f"- {c}")

    if analysis.knowledge_gaps:
        sections.append("\n## Limitations & Gaps\n")
        for g in analysis.knowledge_gaps:
            sections.append(f"- {g}")

    sections.append("\n## Sources\n")
    for i, p in enumerate(pages, 1):
        sections.append(f"[{i}] {p.title} — {p.url}")

    return "\n".join(sections)
