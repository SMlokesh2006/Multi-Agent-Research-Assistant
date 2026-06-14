"""SQLite-based response cache to avoid redundant API calls.

Caches search results, page content, and LLM responses with TTL-based expiration.
Critical for staying within free tier limits during development and demos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


class ResponseCache:
    """Async SQLite cache with TTL expiration.

    Three namespaces:
    - search: Tavily search results keyed by query
    - page: Scraped page content keyed by URL
    - llm: LLM responses keyed by prompt hash
    """

    def __init__(
        self,
        db_path: str = "cache.db",
        search_ttl_hours: int = 24,
        page_ttl_hours: int = 168,
        llm_ttl_hours: int = 72,
    ):
        self.db_path = db_path
        self.ttl_map = {
            "search": search_ttl_hours * 3600,
            "page": page_ttl_hours * 3600,
            "llm": llm_ttl_hours * 3600,
        }
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create the cache table if it doesn't exist."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expiry
                ON cache (namespace, created_at)
            """)
            await db.commit()
        self._initialized = True

    @staticmethod
    def _hash_key(key: str) -> str:
        """Create a consistent hash for cache keys."""
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    async def get(self, namespace: str, key: str) -> dict | list | None:
        """Retrieve a cached value if it exists and hasn't expired.

        Args:
            namespace: One of "search", "page", "llm".
            key: The cache key (query, URL, or prompt).

        Returns:
            The cached value, or None if not found / expired.
        """
        await self._ensure_initialized()
        hashed_key = self._hash_key(key)
        ttl = self.ttl_map.get(namespace, 3600)
        cutoff = time.time() - ttl

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT value FROM cache
                WHERE namespace = ? AND key = ? AND created_at > ?
                """,
                (namespace, hashed_key, cutoff),
            )
            row = await cursor.fetchone()

        if row:
            logger.debug(f"Cache HIT: {namespace}/{key[:50]}")
            return json.loads(row[0])

        logger.debug(f"Cache MISS: {namespace}/{key[:50]}")
        return None

    async def set(self, namespace: str, key: str, value: dict | list) -> None:
        """Store a value in the cache.

        Args:
            namespace: One of "search", "page", "llm".
            key: The cache key.
            value: The value to cache (must be JSON-serializable).
        """
        await self._ensure_initialized()
        hashed_key = self._hash_key(key)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO cache (namespace, key, value, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (namespace, hashed_key, json.dumps(value), time.time()),
            )
            await db.commit()
        logger.debug(f"Cache SET: {namespace}/{key[:50]}")

    async def clear(self, namespace: str | None = None) -> int:
        """Clear cached entries.

        Args:
            namespace: If provided, only clear entries in this namespace.
                      If None, clear all entries.

        Returns:
            Number of entries deleted.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            if namespace:
                cursor = await db.execute(
                    "DELETE FROM cache WHERE namespace = ?", (namespace,)
                )
            else:
                cursor = await db.execute("DELETE FROM cache")
            await db.commit()
            return cursor.rowcount

    async def cleanup_expired(self) -> int:
        """Remove expired entries from all namespaces.

        Returns:
            Number of entries deleted.
        """
        await self._ensure_initialized()
        total_deleted = 0

        async with aiosqlite.connect(self.db_path) as db:
            for namespace, ttl in self.ttl_map.items():
                cutoff = time.time() - ttl
                cursor = await db.execute(
                    "DELETE FROM cache WHERE namespace = ? AND created_at < ?",
                    (namespace, cutoff),
                )
                total_deleted += cursor.rowcount
            await db.commit()

        if total_deleted > 0:
            logger.info(f"Cache cleanup: removed {total_deleted} expired entries")
        return total_deleted

    async def stats(self) -> dict:
        """Return cache statistics."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT namespace, COUNT(*) as count
                FROM cache
                GROUP BY namespace
                """
            )
            rows = await cursor.fetchall()

        return {row[0]: row[1] for row in rows}


# ── Singleton ────────────────────────────────────────────────

_global_cache: ResponseCache | None = None


def get_cache(
    db_path: str = "cache.db",
    search_ttl_hours: int = 24,
    page_ttl_hours: int = 168,
    llm_ttl_hours: int = 72,
) -> ResponseCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache(
            db_path=db_path,
            search_ttl_hours=search_ttl_hours,
            page_ttl_hours=page_ttl_hours,
            llm_ttl_hours=llm_ttl_hours,
        )
    return _global_cache
