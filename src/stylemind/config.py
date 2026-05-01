from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Literal


def get_required_variable(name: str) -> str:
    """Raises ValueError with clear message if env var is missing."""
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"Required environment variable '{name}' is not set")
    return value


def get_optional_variable(name: str, default: str) -> str:
    """Returns env var or default."""
    return os.environ.get(name, default)


def get_float_variable(name: str, default: float) -> float:
    """Returns env var parsed as float, or default on missing/invalid value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable '{name}' must be a float, got {raw!r}") from None


def get_int_variable(name: str, default: int) -> int:
    """Returns env var parsed as int, or default on missing/invalid value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable '{name}' must be an integer, got {raw!r}") from None


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j database connection settings."""

    uri: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> Neo4jConfig:
        return cls(
            uri=get_optional_variable("NEO4J_URI", "bolt://localhost:7687"),
            user=get_optional_variable("NEO4J_USER", "neo4j"),
            password=get_required_variable("NEO4J_PASSWORD"),
        )


@dataclass(frozen=True)
class ChatLLMConfig:
    """Chat LLM provider settings (Groq, OpenAI-compatible)."""

    base_url: str
    api_key: str
    model: str
    temperature: float
    include_usage_in_stream: bool = True

    @classmethod
    def from_env(cls) -> ChatLLMConfig:
        raw = get_optional_variable("CHAT_INCLUDE_USAGE_IN_STREAM", "true")
        return cls(
            base_url=get_optional_variable("CHAT_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=get_required_variable("CHAT_API_KEY"),
            model=get_optional_variable("CHAT_MODEL", "llama-3.3-70b-versatile"),
            temperature=get_float_variable("CHAT_TEMPERATURE", 0.7),
            include_usage_in_stream=raw.lower() not in ("false", "0", "no"),
        )


@dataclass(frozen=True)
class ExtractionLLMConfig:
    """Extraction LLM provider settings for structured persona signal extraction."""

    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> ExtractionLLMConfig:
        return cls(
            base_url=get_optional_variable("EXTRACTION_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=get_required_variable("EXTRACTION_API_KEY"),
            model=get_optional_variable("EXTRACTION_MODEL", "llama-3.3-70b-versatile"),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model settings (local sentence-transformers or OpenAI)."""

    provider: Literal["local", "openai"]
    model_name: str
    dimensions: int

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        return cls(
            provider=get_optional_variable("EMBEDDING_PROVIDER", "local"),  # type: ignore[arg-type]
            model_name=get_optional_variable("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            dimensions=get_int_variable("EMBEDDING_DIMENSIONS", 384),
        )


@dataclass(frozen=True)
class LangfuseConfig:
    public_key: str
    secret_key: str
    host: str

    @classmethod
    def from_env(cls) -> LangfuseConfig:
        return cls(
            public_key=get_optional_variable("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=get_optional_variable("LANGFUSE_SECRET_KEY", ""),
            host=get_optional_variable("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
        )


@dataclass(frozen=True)
class AppSettings:
    """Application-level tuning parameters with range validation."""

    log_level: str
    vector_top_k: int
    persona_decay_rate: float
    expected_signals_per_turn: float
    min_similarity_threshold: float

    def __post_init__(self) -> None:
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {self.log_level!r}")
        if self.vector_top_k <= 0:
            raise ValueError(f"vector_top_k must be > 0, got {self.vector_top_k}")
        if not (0.0 < self.persona_decay_rate < 1.0):
            raise ValueError(f"persona_decay_rate must be in (0, 1), got {self.persona_decay_rate}")
        if self.expected_signals_per_turn <= 0.0:
            raise ValueError(f"expected_signals_per_turn must be > 0, got {self.expected_signals_per_turn}")
        if not (0.0 <= self.min_similarity_threshold <= 1.0):
            raise ValueError(f"min_similarity_threshold must be in [0, 1], got {self.min_similarity_threshold}")

    @classmethod
    def from_env(cls) -> AppSettings:
        return cls(
            log_level=get_optional_variable("LOG_LEVEL", "INFO"),
            vector_top_k=get_int_variable("VECTOR_TOP_K", 10),
            persona_decay_rate=get_float_variable("PERSONA_DECAY_RATE", 0.15),
            expected_signals_per_turn=get_float_variable("EXPECTED_SIGNALS_PER_TURN", 3.0),
            min_similarity_threshold=get_float_variable("MIN_SIMILARITY_THRESHOLD", 0.3),
        )


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration aggregating all sub-configs."""

    neo4j: Neo4jConfig
    chat_llm: ChatLLMConfig
    extraction_llm: ExtractionLLMConfig
    embedding: EmbeddingConfig
    langfuse: LangfuseConfig
    settings: AppSettings

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            neo4j=Neo4jConfig.from_env(),
            chat_llm=ChatLLMConfig.from_env(),
            extraction_llm=ExtractionLLMConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            langfuse=LangfuseConfig.from_env(),
            settings=AppSettings.from_env(),
        )


# Thread-safe singleton
_config: AppConfig | None = None
_lock = threading.Lock()


def get_config() -> AppConfig:
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                _config = AppConfig.from_env()
    return _config


def _reset_config() -> None:
    global _config
    with _lock:
        _config = None
