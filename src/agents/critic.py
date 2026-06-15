"""Critic agent — evaluates analysis for gaps and missing perspectives.

Runs after the Analyst. Identifies claims without sources, missing
opposing views, or specific data points that should exist.
Outputs either follow-up search queries to resolve the gaps, or
"SUFFICIENT" to proceed to the writer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.state import AnalysisResult, ResearchState
from src.utils import extract_text_content
from src.utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """\
You are a senior Research Critic. Evaluate the following research analysis.
Your job is to identify gaps in the research, such as:
1. Claims made without sufficient evidence or clear source attribution.
2. Missing perspectives (e.g., opposing views, controversies, alternative approaches).
3. Specific data points or details that should exist for a comprehensive report but weren't found.

Research query: {query}

--- CURRENT ANALYSIS ---
Findings:
{findings}

Conflicts:
{conflicts}

Knowledge Gaps (identified by analyst):
{gaps}
--- END ANALYSIS ---

--- PAST SEARCH QUERIES ---
The following search queries were attempted in previous iterations:
{past_queries}
--- END PAST SEARCH QUERIES ---

Evaluate the analysis. If the analysis comprehensively covers the query with no major gaps, set "status" to "SUFFICIENT".
If there are gaps, provide 1-3 highly specific follow-up search queries to fill those gaps.

For the limitations section of the final report, categorize each identified gap into one of two categories:
- "searched_but_not_found": If a query related to this gap was already attempted (check the PAST SEARCH QUERIES) but the information is still missing.
- "not_attempted": If it's a gap that requires primary research or has not been searched for yet.

Return a JSON object with EXACTLY this structure (no markdown fences):
{{
  "critique": "A brief explanation of what is missing or weak in the current analysis.",
  "status": "GAPS_FOUND or SUFFICIENT",
  "follow_up_queries": ["query 1", "query 2"],
  "limitations": [
    {{"gap": "Description of the missing information", "category": "searched_but_not_found or not_attempted"}}
  ]
}}

JSON output:"""


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        temperature=0.2,
        max_output_tokens=1024,
        google_api_key=settings.google_api_key,
    )


def _parse_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[: -len("```")]
    text = text.strip()
    return json.loads(text)


async def critic(state: ResearchState) -> dict[str, Any]:
    """Critique the current analysis and optionally generate follow-up searches.

    Args:
        state: The shared research state.

    Returns:
        Partial state update.
    """
    query = state.get("query", "")
    iteration = state.get("iteration", 1)
    analysis_dict = state.get("analysis")
    
    if not analysis_dict:
        logger.warning("critic: no analysis to critique")
        return {"status": "critic_complete", "next": "writer"}

    analysis = AnalysisResult.from_dict(analysis_dict)
    
    findings_text = "\n".join(f"- {f.claim} (Confidence: {f.confidence})" for f in analysis.key_findings)
    conflicts_text = "\n".join(f"- {c}" for c in analysis.conflicts)
    gaps_text = "\n".join(f"- {g}" for g in analysis.knowledge_gaps)

    past_queries_text = "\n".join(f"- {q}" for q in state.get("search_queries", []))

    prompt = _CRITIC_PROMPT.format(
        query=query,
        findings=findings_text or "None",
        conflicts=conflicts_text or "None",
        gaps=gaps_text or "None",
        past_queries=past_queries_text or "None",
    )

    llm = _build_llm()
    limiter = get_rate_limiter(rpm=settings.gemini.rpm_limit)
    errors: list[str] = []

    try:
        async def _invoke():
            response = await llm.ainvoke(prompt)
            return extract_text_content(response.content)

        raw_response = await limiter.execute_with_retry(_invoke)
        parsed = _parse_json(raw_response)
    except Exception as exc:
        error_msg = f"critic: LLM invocation failed: {exc}"
        logger.error(error_msg)
        return {
            "errors": [error_msg],
            "status": "critic_complete",
            "next": "writer",
            "iteration": iteration + 1,
        }

    critique_text = parsed.get("critique", "")
    status = parsed.get("status", "SUFFICIENT")
    queries = parsed.get("follow_up_queries", [])
    limitations = parsed.get("limitations", [])

    logger.info(f"critic: status={status}, critique={critique_text[:60]}")

    if status == "SUFFICIENT" or not queries:
        return {
            "critique": critique_text,
            "status": "critic_complete",
            "next": "writer",
            "iteration": iteration + 1,
            "limitations_log": limitations,
        }
    else:
        return {
            "critique": critique_text,
            "search_queries": queries[:3],  # limit to 3 queries
            "status": "critic_complete",
            "next": "web_searcher",
            "iteration": iteration + 1,
            "limitations_log": limitations,
        }
