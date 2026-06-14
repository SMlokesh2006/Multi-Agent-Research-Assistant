"""FastAPI server for the Multi-Agent Research Assistant.

Endpoints:
- POST   /research               — start a new research session
- GET    /research/{id}/stream    — SSE stream of graph updates
- POST   /research/{id}/feedback  — submit HITL feedback
- GET    /research/{id}/status    — check session status
- GET    /sessions                — list past research sessions
- GET    /health                  — health check
- /                               — serve frontend static files
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import settings
from src.graph import create_graph
from src.persistence import get_checkpointer, get_session_manager
from src.state import create_initial_state

logger = logging.getLogger(__name__)


# ── Request / Response models ────────────────────────────────


class ResearchRequest(BaseModel):
    """Request body for starting a new research session."""

    query: str = Field(..., min_length=3, max_length=2000, description="The research question")
    max_iterations: int = Field(
        default=3, ge=1, le=10, description="Maximum research iterations"
    )


class FeedbackRequest(BaseModel):
    """Request body for submitting HITL feedback."""

    feedback: str = Field(..., min_length=1, max_length=5000, description="Human feedback text")


class ResearchResponse(BaseModel):
    """Response for a newly created research session."""

    thread_id: str
    query: str
    status: str = "started"


class SessionStatusResponse(BaseModel):
    """Response for session status queries."""

    thread_id: str
    query: str
    status: str
    report: str = ""
    created_at: float
    updated_at: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"


# ── In-memory tracking of active research tasks ──────────────

_active_tasks: dict[str, asyncio.Task] = {}
_active_graphs: dict[str, Any] = {}


# ── Lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize persistence on startup, cleanup on shutdown."""
    logger.info("Starting Multi-Agent Research Assistant server")
    # Initialize checkpointer and session manager eagerly
    await get_checkpointer()
    get_session_manager()
    yield
    # Cancel any running research tasks
    for task in _active_tasks.values():
        task.cancel()
    logger.info("Server shutdown complete")


# ── App setup ────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="AI-powered research pipeline with LangGraph, Gemini Flash, and Tavily",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Background research runner ───────────────────────────────


async def _run_research_background(thread_id: str, query: str, max_iterations: int) -> None:
    """Execute the research graph in the background.

    Updates the session record as the research progresses.
    """
    session_mgr = get_session_manager()
    try:
        checkpointer = await get_checkpointer()
        graph = create_graph(checkpointer=checkpointer)
        _active_graphs[thread_id] = graph

        initial_state = create_initial_state(query, max_iterations=max_iterations)
        config = {"configurable": {"thread_id": thread_id}}

        final_state = await graph.ainvoke(initial_state, config=config)

        # Update session with final results
        report = final_state.get("report", "")
        status = final_state.get("status", "complete")
        await session_mgr.save_session(
            thread_id=thread_id,
            query=query,
            status=status,
            report=report,
            metadata={"iteration": final_state.get("iteration", 0)},
        )

    except asyncio.CancelledError:
        logger.info("Research task cancelled: %s", thread_id)
        await session_mgr.save_session(
            thread_id=thread_id, query=query, status="cancelled"
        )
    except Exception as e:
        logger.exception("Research task failed: %s", thread_id)
        await session_mgr.save_session(
            thread_id=thread_id, query=query, status="error",
            metadata={"error": str(e)},
        )
    finally:
        _active_tasks.pop(thread_id, None)
        _active_graphs.pop(thread_id, None)


# ── Endpoints ────────────────────────────────────────────────


@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest) -> ResearchResponse:
    """Start a new research session.

    Creates a background task running the research graph and returns
    the thread ID for streaming and status queries.
    """
    thread_id = str(uuid.uuid4())

    # Save initial session record
    session_mgr = get_session_manager()
    await session_mgr.save_session(
        thread_id=thread_id,
        query=request.query,
        status="started",
    )

    # Launch background research task
    task = asyncio.create_task(
        _run_research_background(thread_id, request.query, request.max_iterations)
    )
    _active_tasks[thread_id] = task

    logger.info("Research started: %s — %s", thread_id, request.query[:80])
    return ResearchResponse(thread_id=thread_id, query=request.query)


@app.get("/research/{thread_id}/stream")
async def stream_research(thread_id: str):
    """SSE endpoint streaming real-time graph updates.

    Event types:
    - ``status``        — pipeline stage change
    - ``search_result`` — new search results found
    - ``content``       — page content extracted
    - ``analysis``      — analysis complete
    - ``report``        — final report generated
    - ``human_review``  — awaiting human input
    - ``error``         — error occurred
    - ``complete``      — research finished
    """

    async def event_generator():
        """Yield SSE events from the graph stream."""
        try:
            checkpointer = await get_checkpointer()
            graph = create_graph(checkpointer=checkpointer)

            # Check if session exists
            session_mgr = get_session_manager()
            session = await session_mgr.get_session(thread_id)
            if not session:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Session not found"}),
                }
                return

            config = {"configurable": {"thread_id": thread_id}}

            # Stream graph updates
            async for event in graph.astream(
                None,  # Resume from checkpoint (state already initialized)
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    sse_event = _map_node_to_sse_event(node_name, node_output)
                    yield {
                        "event": sse_event["type"],
                        "data": json.dumps(sse_event["data"]),
                    }

            # Signal completion
            yield {
                "event": "complete",
                "data": json.dumps({"message": "Research complete"}),
            }

        except Exception as e:
            logger.exception("Stream error for thread %s", thread_id)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


def _map_node_to_sse_event(node_name: str, output: dict) -> dict[str, Any]:
    """Map a graph node update to an SSE event type and payload.

    Args:
        node_name: The name of the graph node that produced the update.
        output: The state update dict from the node.

    Returns:
        A dict with ``type`` and ``data`` keys for SSE serialization.
    """
    event_map: dict[str, str] = {
        "supervisor": "status",
        "web_searcher": "search_result",
        "content_reader": "content",
        "analyst": "analysis",
        "writer": "report",
        "human_review": "human_review",
    }

    event_type = event_map.get(node_name, "status")

    # Build a serializable payload from the output
    data: dict[str, Any] = {"node": node_name}

    if "status" in output:
        data["status"] = output["status"]
    if "search_results" in output:
        data["search_results"] = output["search_results"]
    if "scraped_content" in output:
        data["scraped_content"] = output["scraped_content"]
    if "analysis" in output:
        data["analysis"] = output["analysis"]
    if "report" in output:
        data["report"] = output["report"]
    if "errors" in output:
        data["errors"] = output["errors"]
    if "human_feedback" in output:
        data["human_feedback"] = output["human_feedback"]

    return {"type": event_type, "data": data}


@app.post("/research/{thread_id}/feedback")
async def submit_feedback(thread_id: str, request: FeedbackRequest) -> dict:
    """Submit HITL feedback to resume a paused research session.

    The graph must be in a ``human_review`` interrupt state for this
    endpoint to work.
    """
    try:
        checkpointer = await get_checkpointer()
        graph = create_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        from langgraph.types import Command

        # Resume the graph with human feedback
        result = await graph.ainvoke(
            Command(resume=request.feedback),
            config=config,
        )

        # Update session
        session_mgr = get_session_manager()
        status = result.get("status", "in_progress") if isinstance(result, dict) else "in_progress"
        report = result.get("report", "") if isinstance(result, dict) else ""
        await session_mgr.save_session(
            thread_id=thread_id,
            query=(await session_mgr.get_session(thread_id)).query
            if await session_mgr.get_session(thread_id)
            else "",
            status=status,
            report=report,
        )

        return {"status": "feedback_submitted", "thread_id": thread_id}

    except Exception as e:
        logger.exception("Failed to submit feedback for thread %s", thread_id)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/research/{thread_id}/status", response_model=SessionStatusResponse)
async def get_status(thread_id: str) -> SessionStatusResponse:
    """Check the current status of a research session."""
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(thread_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionStatusResponse(
        thread_id=session.thread_id,
        query=session.query,
        status=session.status,
        report=session.report,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@app.get("/sessions")
async def list_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """List past research sessions, ordered by most recent first."""
    session_mgr = get_session_manager()
    sessions = await session_mgr.get_sessions(limit=limit, offset=offset)
    return [s.to_dict() for s in sessions]


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


# ── Static file serving (frontend) ───────────────────────────

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ── Entrypoint ───────────────────────────────────────────────


def main() -> None:
    """Run the server with uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    uvicorn.run(
        "src.server:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
