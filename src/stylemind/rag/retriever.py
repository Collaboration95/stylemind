from __future__ import annotations

import logging
from typing import Any

from stylemind.graph.client import Neo4jClient
from stylemind.graph.queries import VECTOR_SEARCH_PRODUCTS
from stylemind.models.domain import RetrievedProduct
from stylemind.observability import observe
from stylemind.rag.embedder import Embedder

logger = logging.getLogger(__name__)


def _row_to_retrieved_product(row: dict[str, Any]) -> RetrievedProduct:
    """Map a Cypher result row to a RetrievedProduct dataclass."""
    return RetrievedProduct(
        product_id=row["product_id"],
        name=row["name"],
        description=row["description"] or "",
        price=row["price"] or 0,
        category=row["category"] or "",
        brand=row["brand"] or "",
        budget_tier=row["budget_tier"] or "",
        aesthetics=row["aesthetics"] or [],
        occasions=row["occasions"] or [],
        colors=row["colors"] or [],
        seasons=row["seasons"] or [],
        pairs_with=row["pairs_with"] or [],
        similarity_score=row["score"],
    )


class ProductRetriever:
    """Retrieves products via vector similarity search + graph expansion."""

    def __init__(
        self,
        client: Neo4jClient,
        embedder: Embedder,
        top_k: int = 10,
        min_threshold: float = 0.3,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._top_k = top_k
        self._min_threshold = min_threshold

    @observe(name="retrieve")
    def retrieve(self, query: str) -> list[RetrievedProduct]:
        """Embed the query text, run vector search, and return ranked products.

        Args:
            query: Natural language search query from the user.

        Returns:
            list[RetrievedProduct] sorted by similarity_score descending.
        """
        logger.info(
            "retriever retrieve query_preview=%s top_k=%d min_threshold=%.2f",
            query[:80],
            self._top_k,
            self._min_threshold,
        )

        embedding = self._embedder.embed_query(query)
        logger.debug("retriever embedding_dims=%d", len(embedding))

        params: dict[str, Any] = {
            "embedding": embedding,
            "top_k": self._top_k,
            "min_threshold": self._min_threshold,
        }

        rows = self._client.execute_query(VECTOR_SEARCH_PRODUCTS, params)
        products = [_row_to_retrieved_product(row) for row in rows]

        logger.info("retriever returned result_count=%d", len(products))
        return products
