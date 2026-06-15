"""Human-in-the-loop review node.

This node acts as an anchor for the `interrupt_before` functionality.
When execution pauses before this node, the frontend prompts the user for feedback.
When execution resumes, this node receives the user's feedback via LangGraph's `Command(resume=...)` 
and appends it to the query before routing back to the supervisor to re-plan.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import Command

from src.state import ResearchState

logger = logging.getLogger(__name__)


def human_review(state: ResearchState, command: Command) -> dict[str, Any]:
    """Receive human feedback and append it to the research query."""
    feedback = command.resume if command and hasattr(command, "resume") else None
    
    if not feedback:
        logger.warning("human_review: resumed without feedback")
        return {"next": "supervisor"}

    logger.info("human_review: received feedback: %s", str(feedback)[:50])
    
    original_query = state.get("query", "")
    new_query = f"{original_query}\n\n[Clarification from user: {feedback}]"
    
    return {
        "query": new_query,
        "needs_clarification": False,
        "clarification_question": "",
        "human_feedback": str(feedback),
        "status": "planning",
        "next": "supervisor",
    }
