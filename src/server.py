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
from collections import defaultdict
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
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum research iterations")


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
_event_queues: dict[str, set[asyncio.Queue]] = defaultdict(set)


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
    from src.persistence import close_checkpointer

    await close_checkpointer()
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


async def _run_research_background(
    thread_id: str,
    query: str,
    max_iterations: int,
    initial_input: Any = None,
) -> None:
    """Execute the research graph in the background.

    Updates the session record and broadcasts updates to connected SSE clients.
    """
    session_mgr = get_session_manager()
    try:
        checkpointer = await get_checkpointer()
        graph = create_graph(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": thread_id}}

        # If no initial_input, this is a fresh start
        if initial_input is None:
            initial_input = create_initial_state(query, max_iterations=max_iterations)

        last_state = initial_input

        # Stream graph updates and broadcast to queues
        async for event in graph.astream(initial_input, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                # Skip internal LangGraph events like __interrupt__
                if node_name.startswith("__"):
                    continue
                last_state = node_output
                sse_event = _map_node_to_sse_event(node_name, node_output)
                
                # Broadcast to all listening queues
                if thread_id in _event_queues:
                    payload = {"type": sse_event["type"], "data": sse_event["data"]}
                    for queue in _event_queues[thread_id]:
                        await queue.put(payload)

        # ── Check if graph paused for HITL interrupt ─────────
        graph_state = await graph.aget_state(config)
        is_interrupted = bool(graph_state.next)

        if is_interrupted:
            # Graph is paused at interrupt_before (human_review node).
            # Send a human_review event so the frontend shows the HITL modal,
            # then exit WITHOUT sending "complete".
            logger.info("Research paused for human review: %s", thread_id)

            # Build the HITL event payload from the current graph state
            state_values = graph_state.values or {}
            analysis = state_values.get("analysis") or {}
            hitl_payload = {
                "type": "human_review",
                "data": {
                    "node": "human_review",
                    "agent": "Reviewer",
                    "status": "awaiting_human_review",
                    "message": "Awaiting Human Review",
                    "reason": analysis.get("review_reason", "The research agents need your input to proceed."),
                    "conflicts": analysis.get("conflicts", []),
                    "findings": analysis.get("key_findings", []),
                    "gaps": analysis.get("knowledge_gaps", []),
                },
            }
            if thread_id in _event_queues:
                for queue in _event_queues[thread_id]:
                    await queue.put(hitl_payload)

            await session_mgr.save_session(
                thread_id=thread_id,
                query=query,
                status="awaiting_human_review",
                metadata={"iteration": state_values.get("iteration", 0)},
            )
            # Do NOT send "complete" — the graph is paused, not done
            return

        # ── Graph completed normally ─────────────────────────
        final_state = last_state if isinstance(last_state, dict) else {}
        report = final_state.get("report", "")
        status = final_state.get("status", "complete")
        
        await session_mgr.save_session(
            thread_id=thread_id,
            query=query,
            status=status,
            report=report,
            metadata={"iteration": final_state.get("iteration", 0)},
        )

        # Signal completion to queues
        if thread_id in _event_queues:
            for queue in _event_queues[thread_id]:
                await queue.put({"type": "complete", "data": {"message": "Research complete"}})

    except asyncio.CancelledError:
        logger.info("Research task cancelled: %s", thread_id)
        await session_mgr.save_session(thread_id=thread_id, query=query, status="cancelled")
    except Exception as e:
        logger.exception("Research task failed: %s", thread_id)
        await session_mgr.save_session(
            thread_id=thread_id,
            query=query,
            status="error",
            metadata={"error": str(e)},
        )
        # Signal error to queues
        if thread_id in _event_queues:
            for queue in _event_queues[thread_id]:
                await queue.put({"type": "error", "data": {"message": str(e)}})
    finally:
        _active_tasks.pop(thread_id, None)


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
    """SSE endpoint streaming real-time graph updates via broadcast queues."""

    async def event_generator():
        queue = asyncio.Queue()
        _event_queues[thread_id].add(queue)
        
        try:
            # If the task is already finished, tell the client
            session_mgr = get_session_manager()
            session = await session_mgr.get_session(thread_id)
            if not session:
                yield {"event": "message", "data": json.dumps({"type": "error", "data": {"message": "Session not found"}})}
                return

            if session.status in ["complete", "error", "cancelled"] and thread_id not in _active_tasks:
                yield {"event": "message", "data": json.dumps({"type": "complete", "data": {"message": "Research already finished"}})}
                return

            # Stream from the queue
            while True:
                try:
                    # Wait for an event with a timeout to keep the connection alive
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"event": "message", "data": json.dumps(event)}
                    
                    if event["type"] in ["complete", "error"]:
                        break
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield {"event": "ping", "data": ""}
                
        finally:
            _event_queues[thread_id].remove(queue)
            if not _event_queues[thread_id]:
                del _event_queues[thread_id]

    return EventSourceResponse(event_generator())


def _map_node_to_sse_event(node_name: str, output: dict) -> dict[str, Any]:
    """Map a graph node update to an SSE event type and payload.

    Restructures the raw graph output into the format the frontend expects:
    - ``search_results`` → ``results`` (with url/title/snippet per item)
    - ``scraped_content`` → ``pages`` (with url/title per item)
    - ``analysis`` dict → flat ``findings``, ``conflicts``, ``gaps``
    - ``report`` → ``content``
    - node name → ``agent``, status → ``message``

    Args:
        node_name: The name of the graph node that produced the update.
        output: The state update dict from the node.

    Returns:
        A dict with ``type`` and ``data`` keys for SSE serialization.
    """
    _AGENT_DISPLAY_NAMES: dict[str, str] = {
        "supervisor": "Planner",
        "web_searcher": "Searcher",
        "content_reader": "Extractor",
        "analyst": "Analyzer",
        "writer": "Writer",
        "human_review": "Reviewer",
    }

    event_map: dict[str, str] = {
        "supervisor": "status",
        "web_searcher": "search_result",
        "content_reader": "content",
        "analyst": "analysis",
        "writer": "report",
        "human_review": "human_review",
    }

    event_type = event_map.get(node_name, "status")

    # Build a serializable payload matching the frontend's expected schema
    data: dict[str, Any] = {
        "node": node_name,
        "agent": _AGENT_DISPLAY_NAMES.get(node_name, node_name),
    }

    if "status" in output:
        data["status"] = output["status"]
        data["message"] = output["status"].replace("_", " ").title()

    # Search results: frontend reads `data.results[]`
    if "search_results" in output:
        data["results"] = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("snippet", r.get("content", "")),
            }
            for r in output["search_results"]
        ]

    # Scraped content: frontend reads `data.pages[]`
    if "scraped_content" in output:
        data["pages"] = [
            {
                "url": p.get("url", ""),
                "title": p.get("title", ""),
            }
            for p in output["scraped_content"]
        ]

    # Analysis: frontend reads `data.findings`, `data.conflicts`, `data.gaps`
    if "analysis" in output and isinstance(output["analysis"], dict):
        analysis = output["analysis"]
        data["findings"] = analysis.get("key_findings", [])
        data["conflicts"] = analysis.get("conflicts", [])
        data["gaps"] = analysis.get("knowledge_gaps", [])

    # Report: frontend reads `data.content`
    if "report" in output:
        data["content"] = output["report"]

    if "errors" in output:
        data["errors"] = output["errors"]

    # Human review: frontend reads `data.reason` and `data.conflicts`
    if "human_feedback" in output:
        data["human_feedback"] = output["human_feedback"]
    if node_name == "human_review" and "analysis" in output and isinstance(output["analysis"], dict):
        data["reason"] = output["analysis"].get("review_reason", "Human review requested")

    return {"type": event_type, "data": data}


@app.post("/research/{thread_id}/feedback")
async def submit_feedback(thread_id: str, request: FeedbackRequest) -> dict:
    """Submit HITL feedback to resume a paused research session.

    Resumes by launching a new background task with the feedback command.
    """
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if thread_id in _active_tasks:
        raise HTTPException(status_code=400, detail="Research task is already running")

    from langgraph.types import Command

    # Launch background task with resume command
    task = asyncio.create_task(
        _run_research_background(
            thread_id, 
            session.query, 
            3, # default iterations
            initial_input=Command(resume=request.feedback)
        )
    )
    _active_tasks[thread_id] = task

    return {"status": "feedback_submitted", "thread_id": thread_id}


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
