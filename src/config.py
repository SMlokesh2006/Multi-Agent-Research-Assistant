"""Central configuration for the Multi-Agent Research Assistant."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for the Gemini Flash LLM."""

    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    temperature: float = 0.3
    max_output_tokens: int = 8192
    rpm_limit: int = field(default_factory=lambda: int(os.getenv("GEMINI_RPM_LIMIT", "15")))


@dataclass(frozen=True)
class TavilyConfig:
    """Configuration for the Tavily search API."""

    api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    max_results: int = 5
    search_depth: str = "advanced"  # "basic" (1 credit) or "advanced" (2 credits)
    include_raw_content: bool = False


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the response cache."""

    db_path: str = "cache.db"
    search_ttl_hours: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_HOURS", "24")))
    page_ttl_hours: int = 168  # 7 days
    llm_ttl_hours: int = 72  # 3 days


@dataclass(frozen=True)
class PersistenceConfig:
    """Configuration for session persistence."""

    checkpoint_db: str = "checkpoints.db"
    memory_db: str = "research_memory.db"


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the FastAPI server."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass(frozen=True)
class Settings:
    """Root settings object aggregating all config sections."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def google_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set. Copy .env.example to .env and add your key.")
        return key

    @property
    def langsmith_enabled(self) -> bool:
        return bool(os.getenv("LANGSMITH_API_KEY"))


# ── Singleton ────────────────────────────────────────────────
settings = Settings()
