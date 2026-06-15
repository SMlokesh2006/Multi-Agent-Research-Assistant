"""LangGraph StateGraph construction for the Multi-Agent Research Assistant.

Builds the research pipeline graph with a deterministic loop:
supervisor (planner) -> web_searcher -> content_reader -> analyst -> critic
The critic conditionally routes back to web_searcher or to writer based on iterations.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.agents.analyst import analyst
from src.agents.content_reader import content_reader
from src.agents.critic import critic
from src.agents.supervisor_fixed import supervisor
from src.agents.web_searcher import web_searcher
from src.agents.writer import writer
from src.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)

# ── Node names ───────────────────────────────────────────────

SUPERVISOR = "supervisor"
WEB_SEARCHER = "web_searcher"
CONTENT_READER = "content_reader"
ANALYST = "analyst"
CRITIC = "critic"
WRITER = "writer"


# ── Routing logic ────────────────────────────────────────────


def route_after_supervisor(state: ResearchState) -> str | list[Send]:
    """Route from the initial planner to the web searchers."""
    queries = state.get("search_queries", [])
    if len(queries) > 1:
        return [Send(WEB_SEARCHER, {"query": q}) for q in queries]
    return WEB_SEARCHER


def route_after_analyst(state: ResearchState) -> str:
    """Route after analyst. If max_iterations is 1, skip critic."""
    max_iterations = state.get("max_iterations", 1)
    if max_iterations <= 1:
        return WRITER
    return CRITIC


def route_after_critic(state: ResearchState) -> str | list[Send]:
    """Route after critic. Loop to search if gaps found and within iteration cap."""
    next_node = state.get("next", WRITER)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 1)
    
    if next_node == WEB_SEARCHER and iteration <= max_iterations:
        queries = state.get("search_queries", [])
        if len(queries) > 1:
            return [Send(WEB_SEARCHER, {"query": q}) for q in queries]
        return WEB_SEARCHER
        
    return WRITER


# ── Graph construction ───────────────────────────────────────


def create_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the research pipeline StateGraph."""
    builder = StateGraph(ResearchState)

    # ── Add nodes ────────────────────────────────────────────
    builder.add_node(SUPERVISOR, supervisor)
    builder.add_node(WEB_SEARCHER, web_searcher)
    builder.add_node(CONTENT_READER, content_reader)
    builder.add_node(ANALYST, analyst)
    builder.add_node(CRITIC, critic)
    builder.add_node(WRITER, writer)

    # ── Entry point ──────────────────────────────────────────
    builder.add_edge(START, SUPERVISOR)

    # ── Deterministic edges & routing ────────────────────────
    builder.add_conditional_edges(
        SUPERVISOR,
        route_after_supervisor,
        {WEB_SEARCHER: WEB_SEARCHER}
    )
    
    # After search fan-out finishes, read content
    builder.add_edge(WEB_SEARCHER, CONTENT_READER)
    
    # After reading, analyze
    builder.add_edge(CONTENT_READER, ANALYST)
    
    # After analyst, conditionally go to critic or writer
    builder.add_conditional_edges(
        ANALYST,
        route_after_analyst,
        {CRITIC: CRITIC, WRITER: WRITER}
    )
    
    # After critic, loop to search or go to writer
    builder.add_conditional_edges(
        CRITIC,
        route_after_critic,
        {WEB_SEARCHER: WEB_SEARCHER, WRITER: WRITER}
    )

    # Writer produces the final output
    builder.add_edge(WRITER, END)

    # ── Compile ──────────────────────────────────────────────
    # We removed HUMAN_REVIEW, so no interrupt_before is needed for now.
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Research graph compiled (checkpointer=%s)", type(checkpointer).__name__)
    return graph


# ── Quick-run helper ─────────────────────────────────────────


async def run_research(
    query: str,
    max_iterations: int = 3,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str = "default",
) -> dict[str, Any]:
    """Run a full research pipeline for a query."""
    graph = create_graph(checkpointer=checkpointer)
    initial_state = create_initial_state(query, max_iterations=max_iterations)

    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Starting research for: %s", query[:100])

    final_state = await graph.ainvoke(initial_state, config=config)

    report = final_state.get("report", "")
    logger.info(
        "Research complete — report length: %d chars, iterations: %d",
        len(report),
        final_state.get("iteration", 0),
    )

    return final_state
