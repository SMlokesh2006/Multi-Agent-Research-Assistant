"""Agent sub-package for the Multi-Agent Research Assistant.

Each agent is an async node function with the signature::

    async def agent_name(state: ResearchState) -> dict[str, Any]

Agents return partial state updates that LangGraph merges into the
shared :class:`~src.state.ResearchState`.
"""

from src.agents.analyst import analyst
from src.agents.content_reader import content_reader
from src.agents.supervisor import route_supervisor, supervisor
from src.agents.web_searcher import web_searcher
from src.agents.writer import writer

__all__ = [
    "analyst",
    "content_reader",
    "route_supervisor",
    "supervisor",
    "web_searcher",
    "writer",
]
