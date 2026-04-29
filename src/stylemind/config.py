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


@dataclass(frozen=True)
class Neo4jConfig:
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
    base_url: str
    api_key: str
    model: str
    temperature: float

    @classmethod
    def from_env(cls) -> ChatLLMConfig:
        return cls(
            base_url=get_optional_variable("CHAT_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=get_required_variable("CHAT_API_KEY"),
            model=get_optional_variable("CHAT_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(get_optional_variable("CHAT_TEMPERATURE", "0.7")),
        )


@dataclass(frozen=True)
class ExtractionLLMConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> ExtractionLLMConfig:
        return cls(
            base_url=get_optional_variable("EXTRACTION_BASE_URL", "https://api.openai.com/v1"),
            api_key=get_required_variable("EXTRACTION_API_KEY"),
            model=get_optional_variable("EXTRACTION_MODEL", "gpt-4.1-nano"),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Literal["local", "openai"]
    model_name: str
    dimensions: int

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        return cls(
            provider=get_optional_variable("EMBEDDING_PROVIDER", "local"),
            model_name=get_optional_variable("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            dimensions=int(get_optional_variable("EMBEDDING_DIMENSIONS", "384")),
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
            host=get_optional_variable("LANGFUSE_HOST", "http://localhost:3000"),
        )


@dataclass(frozen=True)
class AppSettings:
    log_level: str
    vector_top_k: int
    persona_decay_rate: float
    expected_signals_per_turn: float
    min_similarity_threshold: float

    def __post_init__(self) -> None:
        assert self.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), f"Invalid log_level: {self.log_level}"
        assert self.vector_top_k > 0, f"vector_top_k must be > 0, got {self.vector_top_k}"
        assert 0.0 < self.persona_decay_rate < 1.0, f"persona_decay_rate must be in (0, 1), got {self.persona_decay_rate}"
        assert self.expected_signals_per_turn > 0.0, f"expected_signals_per_turn must be > 0, got {self.expected_signals_per_turn}"
        assert 0.0 <= self.min_similarity_threshold <= 1.0, f"min_similarity_threshold must be in [0, 1], got {self.min_similarity_threshold}"

    @classmethod
    def from_env(cls) -> AppSettings:
        return cls(
            log_level=get_optional_variable("LOG_LEVEL", "INFO"),
            vector_top_k=int(get_optional_variable("VECTOR_TOP_K", "10")),
            persona_decay_rate=float(get_optional_variable("PERSONA_DECAY_RATE", "0.15")),
            expected_signals_per_turn=float(get_optional_variable("EXPECTED_SIGNALS_PER_TURN", "3.0")),
            min_similarity_threshold=float(get_optional_variable("MIN_SIMILARITY_THRESHOLD", "0.3")),
        )


@dataclass(frozen=True)
class AppConfig:
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
