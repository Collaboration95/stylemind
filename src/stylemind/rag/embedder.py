from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol defining the embedding interface."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        ...

    @property
    def dimensions(self) -> int:
        """Embedding vector dimensionality."""
        ...


class LocalEmbedder:
    """sentence-transformers/all-MiniLM-L6-v2 (384 dims, no API key required)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", *, lazy: bool = False) -> None:
        self._model_name = model_name
        self._model = None
        self._dimensions: int | None = None
        if not lazy:
            self._get_model()

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model=%s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            dim_fn = (
                getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            )
            self._dimensions = dim_fn()
        return self._model

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._get_model()
        return self._dimensions  # type: ignore[return-value]


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small via OpenAI SDK (configurable dimensions)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dimensions,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self._dimensions


def get_embedder(config=None) -> LocalEmbedder | OpenAIEmbedder:
    """Factory: returns the configured embedder based on EMBEDDING_PROVIDER env."""
    if config is None:
        from stylemind.config import get_config

        config = get_config().embedding

    if config.provider == "openai":
        from stylemind.config import get_config

        app_config = get_config()
        return OpenAIEmbedder(
            api_key=app_config.extraction_llm.api_key,
            model=config.model_name,
            dimensions=config.dimensions,
        )
    elif config.provider == "local":
        return LocalEmbedder(model_name=config.model_name)
    else:
        raise ValueError(f"Invalid EMBEDDING_PROVIDER: {config.provider!r}. Valid options: 'local', 'openai'")
