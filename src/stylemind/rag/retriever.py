from __future__ import annotations

import logging
from typing import Any

from stylemind.graph.client import Neo4jClient
from stylemind.graph.queries import VECTOR_SEARCH_AESTHETIC_FALLBACK, VECTOR_SEARCH_PRODUCTS
from stylemind.models.domain import RetrievedProduct
from stylemind.observability import observe
from stylemind.rag.embedder import Embedder

logger = logging.getLogger(__name__)

_FALLBACK_THRESHOLD = 3


def _row_to_retrieved_product(row: dict[str, Any]) -> RetrievedProduct:
    return RetrievedProduct(
        product_id=row["product_id"],
        name=row["name"],
        description=row["description"] or "",
        price_inr=row["price_inr"] or 0,
        category=row["category"] or "",
        brand=row["brand"] or "",
        budget_tier=row["budget_tier"] or "",
        aesthetics=row["aesthetics"] or [],
        occasions=row["occasions"] or [],
        colors=row["colors"] or [],
        seasons=row["seasons"] or [],
        materials=row.get("materials") or [],
        pairs_with=row["pairs_with"] or [],
        similarity_score=row["score"],
    )


class ProductRetriever:
    """Retrieves products via vector similarity search + graph expansion.

    When product vector search returns fewer than 3 results, falls back to
    aesthetic vector search: finds matching aesthetics, then graph-expands
    to products that embody those aesthetics.
    """

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
        logger.info(
            "retriever retrieve query_preview=%s top_k=%d min_threshold=%.2f",
            query[:80],
            self._top_k,
            self._min_threshold,
        )

        embedding = self._embedder.embed_query(query)

        params: dict[str, Any] = {
            "embedding": embedding,
            "top_k": self._top_k,
            "min_threshold": self._min_threshold,
        }

        rows = self._client.execute_query(VECTOR_SEARCH_PRODUCTS, params)
        products = [_row_to_retrieved_product(row) for row in rows]

        if len(products) < _FALLBACK_THRESHOLD:
            logger.info(
                "retriever product_count=%d below threshold=%d, trying aesthetic fallback",
                len(products),
                _FALLBACK_THRESHOLD,
            )
            fallback_products = self._aesthetic_fallback(embedding, products)
            products = self._merge_and_deduplicate(products, fallback_products)

        logger.info("retriever returned result_count=%d", len(products))
        return products

    def _aesthetic_fallback(self, embedding: list[float], existing: list[RetrievedProduct]) -> list[RetrievedProduct]:
        existing_ids = {p.product_id for p in existing}
        params: dict[str, Any] = {
            "embedding": embedding,
            "top_k": self._top_k,
            "top_k_aesthetics": 3,
            "min_threshold": self._min_threshold,
        }
        try:
            rows = self._client.execute_query(VECTOR_SEARCH_AESTHETIC_FALLBACK, params)
            products = [_row_to_retrieved_product(row) for row in rows if row["product_id"] not in existing_ids]
            logger.info("retriever aesthetic_fallback returned %d new products", len(products))
            return products
        except Exception as exc:
            logger.warning("retriever aesthetic_fallback failed error=%s", exc)
            return []

    def _merge_and_deduplicate(
        self, primary: list[RetrievedProduct], fallback: list[RetrievedProduct]
    ) -> list[RetrievedProduct]:
        seen = {p.product_id for p in primary}
        merged = list(primary)
        for p in fallback:
            if p.product_id not in seen:
                seen.add(p.product_id)
                merged.append(p)
        return merged[: self._top_k]
