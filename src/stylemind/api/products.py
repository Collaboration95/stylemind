from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()

FETCH_PRODUCT_NAMES = """
MATCH (p:Product)-[:BELONGS_TO]->(b:Brand)
RETURN p.product_id AS product_id, p.name AS name, b.name AS brand
ORDER BY p.name
"""


@router.get("/products/names")
async def list_product_names(request: Request) -> list[dict[str, str]]:
    """Return all product names with IDs for autocomplete."""
    neo4j_client = getattr(request.app.state, "neo4j", None)
    if neo4j_client is None:
        return []

    try:
        rows = await asyncio.to_thread(neo4j_client.execute_query, FETCH_PRODUCT_NAMES, {})
        return [{"product_id": r["product_id"], "name": r["name"], "brand": r["brand"]} for r in rows]
    except Exception as exc:
        logger.warning("products list_names failed error=%s", exc)
        return []
