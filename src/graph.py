"""LangGraph StateGraph construction for the Multi-Agent Research Assistant.

Builds the research pipeline graph with:
- Supervisor-driven routing (manual pattern, not auto)
- Send API for parallel fan-out to sub-agents
- interrupt_before for Human-in-the-Loop checkpointing
- SqliteSaver for cross-session persistence

The supervisor node sets a ``next`` field in the state update, which
the conditional edge function reads to route to the appropriate node.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt

from src.agents.analyst import analyst
from src.agents.content_reader import content_reader
from src.agents.supervisor_fixed import supervisor
from src.agents.supervisor_fixed import route_supervisor as _supervisor_route
from src.agents.web_searcher import web_searcher
from src.agents.writer import writer
from src.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)

# ── Node names ───────────────────────────────────────────────

SUPERVISOR = "supervisor"
WEB_SEARCHER = "web_searcher"
CONTENT_READER = "content_reader"
ANALYST = "analyst"
WRITER = "writer"
HUMAN_REVIEW = "human_review"


# ── Human review node ────────────────────────────────────────


async def human_review(state: ResearchState) -> dict:
    """Passthrough node for Human-in-the-Loop review.

    This node is gated by ``interrupt_before`` — LangGraph pauses
    execution *before* entering this node, allowing external systems
    (CLI, API) to inject ``human_feedback`` into the state.

    When resumed via ``Command(resume=feedback)``, it reads the
    feedback and passes it along to the supervisor for incorporation.
    """
    # The interrupt() call will pause the graph. When resumed, the value
    # passed to resume will be returned by this call.
    feedback_command = interrupt("Awaiting human review feedback")

    # If feedback is a string (from resume), update the state
    if isinstance(feedback_command, str):
        feedback = feedback_command
    else:
        feedback = None  # Or handle other types as needed

    logger.info("Human review received feedback: %s", feedback[:100] if feedback else "None")

    return {
        "human_feedback": feedback,
        "status": "human_feedback_received",
    }


# ── Routing logic ────────────────────────────────────────────


def route_after_supervisor(state: ResearchState) -> str | list[Send]:
    """Conditional edge: route from supervisor to the next node.

    The supervisor node sets a ``next`` field in its state update.
    This function delegates to the supervisor module's own
    ``route_supervisor`` function which reads that field and
    optionally returns ``Send`` objects for parallel fan-out.

    Returns:
        A node name string or a list of ``Send`` objects.
    """
    result = _supervisor_route(state)
    logger.info("Routing from supervisor → %s", result)
    return result


# ── Graph construction ───────────────────────────────────────


def create_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the research pipeline StateGraph.

    Node topology::

        START → supervisor ─┬─→ web_searcher   → supervisor
                            ├─→ content_reader → supervisor
                            ├─→ analyst        → supervisor
                            ├─→ writer         → END
                            ├─→ human_review   → supervisor
                            └─→ END

    Args:
        checkpointer: Optional checkpoint saver for persistence.
            Pass a ``SqliteSaver`` / ``AsyncSqliteSaver`` for
            cross-session persistence, or ``None`` for ephemeral
            in-memory execution.

    Returns:
        A compiled LangGraph ready for ``.ainvoke()`` or ``.astream()``.
    """
    builder = StateGraph(ResearchState)

    # ── Add nodes ────────────────────────────────────────────
    builder.add_node(SUPERVISOR, supervisor)
    builder.add_node(WEB_SEARCHER, web_searcher)
    builder.add_node(CONTENT_READER, content_reader)
    builder.add_node(ANALYST, analyst)
    builder.add_node(WRITER, writer)
    builder.add_node(HUMAN_REVIEW, human_review)

    # ── Entry point ──────────────────────────────────────────
    builder.add_edge(START, SUPERVISOR)

    # ── Supervisor conditional routing ───────────────────────
    # The supervisor returns a ``next`` field; the route function
    # maps it to one of the node names or END.
    builder.add_conditional_edges(
        SUPERVISOR,
        route_after_supervisor,
        {
            WEB_SEARCHER: WEB_SEARCHER,
            CONTENT_READER: CONTENT_READER,
            ANALYST: ANALYST,
            WRITER: WRITER,
            HUMAN_REVIEW: HUMAN_REVIEW,
            END: END,
        },
    )

    # ── Sub-agents always return to supervisor ───────────────
    builder.add_edge(WEB_SEARCHER, SUPERVISOR)
    builder.add_edge(CONTENT_READER, SUPERVISOR)
    builder.add_edge(ANALYST, SUPERVISOR)

    # ── Writer produces the final output ─────────────────────
    builder.add_edge(WRITER, END)

    # ── Human review feeds back into supervisor ──────────────
    builder.add_edge(HUMAN_REVIEW, SUPERVISOR)

    # ── Compile with HITL interrupt ──────────────────────────
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=[HUMAN_REVIEW],
    )

    logger.info("Research graph compiled (checkpointer=%s)", type(checkpointer).__name__)
    return graph


# ── Quick-run helper ─────────────────────────────────────────


async def run_research(
    query: str,
    max_iterations: int = 3,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str = "default",
) -> dict[str, Any]:
    """Run a full research pipeline for a query (no HITL).

    Convenience function for scripts and notebooks. For production
    use with HITL, streaming, and persistence, use ``create_graph()``
    directly with a proper checkpointer.

    Args:
        query: The research question to investigate.
        max_iterations: Maximum supervisor loop iterations.
        checkpointer: Optional checkpoint saver.
        thread_id: Thread identifier for checkpointing.

    Returns:
        The final state dict with the completed report.
    """
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
