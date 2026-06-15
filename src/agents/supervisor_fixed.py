"""Supervisor agent — orchestrates the multi-agent research pipeline.

This is the central routing node. It uses Gemini Flash with tool-calling
to decide which sub-agent to invoke next, decomposes research questions
into search sub-queries, tracks iteration counts, and triggers
human-in-the-loop reviews when the analyst flags ambiguity.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Send

from src.config import settings
from src.state import AnalysisResult, ResearchState
from src.utils import extract_text_content
from src.utils.cache import get_cache
from src.utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# The tools the supervisor can "call" — modelled as plain dicts so we
# can use Gemini's function-calling interface.
_SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web for information. Use this when you need to "
                "gather information on a topic. Provide a list of search queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 focused search queries to investigate the research question.",
                    }
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_content",
            "description": (
                "Extract and read the full content of web pages found by search. "
                "Use after search results are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": (
                "Analyse scraped content to extract findings, conflicts, and gaps. "
                "Use after content has been read."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": (
                "Generate a final research report from the analysis. "
                "Use after analysis is complete and no further research is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_human_input",
            "description": (
                "Request human review and feedback before proceeding. "
                "Use when there are significant conflicts or ambiguity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why human input is needed.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "End the research pipeline. Use when the report is complete "
                "or max iterations have been reached."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

_SUPERVISOR_SYSTEM = """
You are the Supervisor of a multi-agent research team. Your role is to
coordinate the research pipeline by deciding which action to take next.

Available actions:
- search: Search the web (provide 2-4 focused queries)
- read_content: Extract full content from search result URLs
- analyze: Analyse all scraped content for findings and conflicts
- write_report: Generate the final research report
- request_human_input: Ask the human for guidance (use when conflicts/ambiguity exist)
- finish: End the pipeline (only after report is complete)

Decision rules:
1. If no search has been done yet → search.
2. If search results exist but content hasn't been read → read_content.
3. If content has been read but not analysed → analyze.
4. If analysis shows major gaps AND iterations remain → search with follow-up queries.
5. If analysis is complete and human review is flagged → request_human_input.
6. If analysis is complete (or human has responded) → write_report.
7. If report is complete → finish.
8. NEVER exceed max_iterations — if at limit, go to write_report or finish.

Always call exactly ONE tool per response.
"""


# ── Helpers ──────────────────────────────────────────────────


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini Flash LLM instance configured for tool calling."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        temperature=0.1,  # Low temperature for deterministic routing
        max_output_tokens=1024,  # Routing decisions are short
        google_api_key=settings.google_api_key,
    )


def _build_state_summary(state: ResearchState) -> str:
    """Build a concise text summary of current pipeline state for the LLM."""
    parts: list[str] = []
    parts.append(f"Research query: {state.get('query', '')}")
    parts.append(f"Current status: {state.get('status', 'initialized')}")
    parts.append(f"Iteration: {state.get('iteration', 0)} / {state.get('max_iterations', 3)}")

    search_results = state.get("search_results", [])
    parts.append(f"Search results collected: {len(search_results)}")

    scraped = state.get("scraped_content", [])
    parts.append(f"Pages scraped: {len(scraped)}")

    analysis = state.get("analysis")
    if analysis:
        a = AnalysisResult.from_dict(analysis)
        parts.append(f"Analysis: {len(a.key_findings)} findings, {len(a.conflicts)} conflicts")
        parts.append(f"Knowledge gaps: {len(a.knowledge_gaps)}")
        parts.append(f"Needs human review: {a.needs_human_review}")
        if a.follow_up_queries:
            parts.append(f"Suggested follow-ups: {a.follow_up_queries}")
    else:
        parts.append("Analysis: not yet performed")

    report = state.get("report", "")
    parts.append(f"Report written: {'yes' if report else 'no'}")

    human_feedback = state.get("human_feedback")
    if human_feedback:
        parts.append(f"Human feedback received: {human_feedback}")

    errors = state.get("errors", [])
    if errors:
        parts.append(f"Errors ({len(errors)}): {errors[-3:]}")  # last 3

    return "\n".join(parts)


async def plan_research(query: str) -> list[str]:
    """Decompose a research question into 2-4 focused search queries."""
    llm = _build_llm()
    limiter = get_rate_limiter(rpm=settings.gemini.rpm_limit)
    cache = get_cache(
        db_path=settings.cache.db_path,
        search_ttl_hours=settings.cache.search_ttl_hours,
        page_ttl_hours=settings.cache.page_ttl_hours,
        llm_ttl_hours=settings.cache.llm_ttl_hours,
    )

    # ── Cache check ──────────────────────────────────────────
    cache_key = f"plan:{query}"
    cached = await cache.get("llm", cache_key)
    if cached is not None:
        logger.info("plan_research: returning cached queries")
        return cached  # type: ignore[return-value]

    prompt = (
        f"You are a research planner. Given the following research question, "
        f"generate 2-4 diverse, specific search queries that together would "
        f"comprehensively investigate the topic. Return ONLY a JSON array of "
        f"strings, no other text.\n\n"  # Added \n\n here
        f"Research question: {query}\n\n"  # Added \n\n here
        f"JSON array of search queries:"
    )

    async def _invoke():
        response = await llm.ainvoke(prompt)
        return extract_text_content(response.content)

    try:
        raw = await limiter.execute_with_retry(_invoke)
        text = raw.strip()
        
        # Robustly extract JSON from potential markdown fences
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        text = text.strip()

        logger.info(f"plan_research: LLM response: {text[:100]}...")
        queries = json.loads(text)
        
        if not isinstance(queries, list) or not queries:
            logger.warning("plan_research: LLM returned empty or non-list, falling back")
            queries = [query]
    except Exception as exc:
        logger.error(f"plan_research: failed to parse or generate queries: {exc}")
        queries = [query]

    # Ensure we have 1-4 valid strings
    queries = [str(q).strip() for q in queries if q][:4]
    if not queries:
        queries = [query]

    await cache.set("llm", cache_key, queries)
    logger.info(f"plan_research: generated {len(queries)} queries: {queries}")
    return queries


# ── Route Decision Types ─────────────────────────────────────

# Possible next-node names that the graph can route to.
NextNode = Literal[
    "web_searcher",
    "content_reader",
    "analyst",
    "writer",
    "human_review",
    "__end__",
]


async def supervisor(state: ResearchState) -> dict[str, Any]:
    """Decide and route to the next step in the research pipeline."""
    logger.debug(f"DEBUG: supervisor called with state type: {type(state)}")
    query = state.get("query", "")
    status = state.get("status", "initialized")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    logger.info(f"supervisor: status={status}, iteration={iteration}/{max_iterations}")

    # ── Fast-path: first invocation → plan and search ────────
    if status == "initialized" and not state.get("search_results"):
        search_queries = await plan_research(query)
        return {
            "search_queries": search_queries,
            "status": "searching",
            "iteration": iteration + 1,
            "next": "web_searcher",
        }

    # ── Fast-path: max iterations reached ────────────────────
    if iteration >= max_iterations:
        logger.warning("supervisor: max iterations reached")
        if state.get("report"):
            return {"status": "complete", "next": "__end__"}
        if state.get("analysis"):
            return {"status": "writing", "next": "writer"}
        return {
            "status": "complete",
            "next": "__end__",
            "errors": ["supervisor: max iterations reached without completing research"],
        }

    # ── LLM-based routing for all other states ───────────────
    llm = _build_llm()
    limiter = get_rate_limiter(rpm=settings.gemini.rpm_limit)

    state_summary = _build_state_summary(state)
    messages = [
        SystemMessage(content=_SUPERVISOR_SYSTEM),
        HumanMessage(
            content=f"""Current pipeline state:
{state_summary}

Decide the next action."""
        ),
    ]

    try:
        async def _invoke():
            return await llm.ainvoke(messages, tools=_SUPERVISOR_TOOLS)
        response = await limiter.execute_with_retry(_invoke)
    except Exception as exc:
        error_msg = f"supervisor: LLM routing failed: {exc}"
        logger.error(error_msg)
        return _fallback_route(state)

    # ── Parse tool call from response ────────────────────────
    return _parse_tool_call(response, state)


def _parse_tool_call(response: Any, state: ResearchState) -> dict[str, Any]:
    """Extract the supervisor's routing decision from the LLM response."""
    logger.debug(f"DEBUG: _parse_tool_call called with state type: {type(state)}")
    iteration = state.get("iteration", 0)

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        logger.info(f"supervisor: LLM chose tool '{tool_name}'")

        if tool_name == "search":
            queries = tool_args.get("queries", [])
            if not queries:
                logger.warning("supervisor: LLM returned empty search queries, using fallback")
                queries = [state.get("query", "")]
            return {
                "search_queries": queries,
                "status": "searching",
                "iteration": iteration + 1,
                "next": "web_searcher",
            }
        elif tool_name == "read_content":
            return {"status": "reading", "next": "content_reader"}
        elif tool_name == "analyze":
            return {"status": "analysing", "next": "analyst"}
        elif tool_name == "write_report":
            return {"status": "writing", "next": "writer"}
        elif tool_name == "request_human_input":
            reason = tool_args.get("reason", "Human review requested")
            return {
                "status": "awaiting_human",
                "human_feedback": None,
                "next": "human_review",
                "messages": [AIMessage(content=f"🔍 Human review requested: {reason}")],
            }
        elif tool_name == "finish":
            return {"status": "complete", "next": "__end__"}

    logger.warning("supervisor: no valid tool call in LLM response, using fallback")
    return _fallback_route(state)


def _fallback_route(state: ResearchState) -> dict[str, Any]:
    """Deterministic fallback routing when LLM doesn't provide a tool call."""
    logger.debug(f"DEBUG: _fallback_route called with state type: {type(state)}")
    iteration = state.get("iteration", 0)

    has_results = bool(state.get("search_results"))
    has_content = bool(state.get("scraped_content"))
    has_analysis = state.get("analysis") is not None
    has_report = bool(state.get("report"))

    if has_report:
        return {"status": "complete", "next": "__end__"}

    if has_analysis:
        analysis = AnalysisResult.from_dict(state["analysis"])
        if analysis.needs_human_review and state.get("human_feedback") is None:
            return {
                "status": "awaiting_human",
                "next": "human_review",
                "messages": [AIMessage(content=f"🔍 Human review requested: {analysis.review_reason}")],
            }
        return {"status": "writing", "next": "writer"}

    if has_content:
        return {"status": "analysing", "next": "analyst"}

    if has_results:
        return {"status": "reading", "next": "content_reader"}

    return {
        "search_queries": [state.get("query", "")],
        "status": "searching",
        "iteration": iteration + 1,
        "next": "web_searcher",
    }


def route_supervisor(state: ResearchState) -> NextNode | list[Send]:
    """Conditional edge function: route from supervisor to the next node."""
    next_node: str = state.get("next", "__end__")

    if next_node == "web_searcher":
        queries = state.get("search_queries", [])
        if len(queries) > 1:
            logger.info(f"supervisor: fanning out {len(queries)} search workers")
            return [Send("web_searcher", {"query": q}) for q in queries]
        return "web_searcher"

    return next_node
