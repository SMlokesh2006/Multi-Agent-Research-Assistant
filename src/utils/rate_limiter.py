"""Token-bucket rate limiter for Gemini API calls.

Ensures we stay within the free tier RPM limits by throttling requests
with exponential backoff and jitter.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter with async support.

    Args:
        rpm: Maximum requests per minute.
        max_retries: Maximum number of retries on rate-limit errors.
        base_delay: Base delay in seconds for exponential backoff.
    """

    def __init__(self, rpm: int = 15, max_retries: int = 5, base_delay: float = 2.0):
        self.rpm = rpm
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Token bucket state
        self._tokens = float(rpm)
        self._max_tokens = float(rpm)
        self._refill_rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

        # Stats
        self.total_requests = 0
        self.total_waits = 0
        self.total_wait_time = 0.0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary.

        Blocks until a token is available, ensuring we don't exceed RPM.
        """
        async with self._lock:
            self._refill()

            if self._tokens < 1.0:
                # Calculate wait time until a token is available
                wait_time = (1.0 - self._tokens) / self._refill_rate
                # Add small jitter to prevent thundering herd
                jitter = random.uniform(0, 0.5)
                total_wait = wait_time + jitter

                logger.info(f"Rate limiter: waiting {total_wait:.1f}s (tokens={self._tokens:.2f})")
                self.total_waits += 1
                self.total_wait_time += total_wait

                await asyncio.sleep(total_wait)
                self._refill()

            self._tokens -= 1.0
            self.total_requests += 1

    async def execute_with_retry(self, coro_factory, *args, **kwargs):
        """Execute an async function with rate limiting and retry logic.

        Args:
            coro_factory: A callable that returns a coroutine when called.
            *args, **kwargs: Arguments passed to the callable.

        Returns:
            The result of the coroutine.

        Raises:
            Exception: After max_retries is exhausted.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            await self.acquire()

            try:
                return await coro_factory(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if it's a rate limit error (retry) vs other error (raise)
                is_rate_limit = any(
                    phrase in error_msg
                    for phrase in ["rate limit", "429", "quota", "resource exhausted"]
                )

                if not is_rate_limit or attempt >= self.max_retries:
                    raise

                # Exponential backoff with jitter
                delay = self.base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Rate limited (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

        raise last_exception  # type: ignore[misc]

    @property
    def stats(self) -> dict:
        """Return usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_waits": self.total_waits,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "tokens_remaining": round(self._tokens, 2),
        }


# ── Singleton for global use ─────────────────────────────────

_global_limiter: RateLimiter | None = None


def get_rate_limiter(rpm: int = 15) -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter(rpm=rpm)
    return _global_limiter
