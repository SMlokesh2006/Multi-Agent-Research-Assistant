"""Cross-session persistence for the Multi-Agent Research Assistant.

Provides:
- ``get_checkpointer()`` — LangGraph SqliteSaver for graph checkpoints
- ``SessionManager`` — research session metadata storage (separate DB)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import settings

logger = logging.getLogger(__name__)


# ── Checkpoint saver ─────────────────────────────────────────

_checkpointer_context: Any = None
_checkpointer: AsyncSqliteSaver | None = None


async def get_checkpointer() -> AsyncSqliteSaver:
    """Get or create the global LangGraph checkpoint saver.

    Uses ``AsyncSqliteSaver`` backed by the checkpoint DB configured
    in ``settings.persistence.checkpoint_db``.

    Returns:
        An initialized ``AsyncSqliteSaver`` instance.
    """
    global _checkpointer, _checkpointer_context
    if _checkpointer is None:
        db_path = settings.persistence.checkpoint_db
        _checkpointer_context = AsyncSqliteSaver.from_conn_string(db_path)
        _checkpointer = await _checkpointer_context.__aenter__()
        await _checkpointer.setup()
        logger.info("Checkpoint saver initialized: %s", db_path)
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the global checkpoint saver if it is open."""
    global _checkpointer, _checkpointer_context
    if _checkpointer_context is not None:
        await _checkpointer_context.__aexit__(None, None, None)
        _checkpointer_context = None
        _checkpointer = None
        logger.info("Checkpoint saver closed")


# ── Session metadata ─────────────────────────────────────────


@dataclass
class SessionRecord:
    """Metadata for a completed (or in-progress) research session."""

    thread_id: str
    query: str
    status: str = "in_progress"
    report: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> SessionRecord:
        """Construct from a SQLite row tuple."""
        return cls(
            thread_id=row[0],
            query=row[1],
            status=row[2],
            report=row[3],
            created_at=row[4],
            updated_at=row[5],
            metadata=json.loads(row[6]) if row[6] else {},
        )


class SessionManager:
    """Manages research session metadata in a dedicated SQLite database.

    Separate from the LangGraph checkpoint DB — this stores
    human-readable session metadata for the API and CLI.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.persistence.memory_db
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create the sessions table if it doesn't exist."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'in_progress',
                    report TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created
                ON sessions (created_at DESC)
            """)
            await db.commit()

        self._initialized = True
        logger.info("Session manager initialized: %s", self.db_path)

    async def save_session(
        self,
        thread_id: str,
        query: str,
        status: str = "in_progress",
        report: str = "",
        metadata: dict | None = None,
    ) -> SessionRecord:
        """Create or update a research session record.

        Args:
            thread_id: Unique thread identifier.
            query: The original research question.
            status: Current session status.
            report: The generated report (empty if in-progress).
            metadata: Optional additional metadata.

        Returns:
            The saved ``SessionRecord``.
        """
        await self._ensure_initialized()
        now = time.time()
        meta_json = json.dumps(metadata or {})

        async with aiosqlite.connect(self.db_path) as db:
            # Check if session exists to preserve created_at
            cursor = await db.execute(
                "SELECT created_at FROM sessions WHERE thread_id = ?",
                (thread_id,),
            )
            existing = await cursor.fetchone()
            created_at = existing[0] if existing else now

            await db.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (thread_id, query, status, report, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (thread_id, query, status, report, created_at, now, meta_json),
            )
            await db.commit()

        record = SessionRecord(
            thread_id=thread_id,
            query=query,
            status=status,
            report=report,
            created_at=created_at,
            updated_at=now,
            metadata=metadata or {},
        )
        logger.debug("Session saved: %s (status=%s)", thread_id, status)
        return record

    async def get_session(self, thread_id: str) -> SessionRecord | None:
        """Retrieve a single session by thread ID.

        Args:
            thread_id: The thread identifier to look up.

        Returns:
            The ``SessionRecord`` if found, or ``None``.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()

        if row:
            return SessionRecord.from_row(row)
        return None

    async def get_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        """List past research sessions, ordered by most recent first.

        Args:
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip (for pagination).

        Returns:
            A list of ``SessionRecord`` objects.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT * FROM sessions
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()

        return [SessionRecord.from_row(row) for row in rows]

    async def delete_session(self, thread_id: str) -> bool:
        """Delete a session record.

        Args:
            thread_id: The thread identifier to delete.

        Returns:
            ``True`` if the session was deleted, ``False`` if not found.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE thread_id = ?",
                (thread_id,),
            )
            await db.commit()
            return cursor.rowcount > 0


# ── Singleton ────────────────────────────────────────────────

_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
