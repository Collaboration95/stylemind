from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_product_text(record: dict) -> str:
    """Deterministic text for product embedding."""
    name = record.get("name", "")
    description = record.get("description", "")

    # Get related data from query results
    aesthetics = record.get("aesthetics", [])
    occasions = record.get("occasions", [])
    color_palette = record.get("color_palette", "")

    aesthetic_str = ", ".join(aesthetics) if aesthetics else ""
    occasion_str = ", ".join(occasions) if occasions else ""

    return f"{name}. {description}. Style: {aesthetic_str}. For: {occasion_str}. In {color_palette}."


def build_aesthetic_text(record: dict) -> str:
    """Deterministic text for aesthetic embedding."""
    name = record.get("name", "")
    description = record.get("description", "")
    keywords = record.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else ""
    return f"{name}. {description}. Keywords: {keywords_str}."


FETCH_PRODUCTS = """
MATCH (p:Product)
OPTIONAL MATCH (p)-[:EMBODIES]->(a:Aesthetic)
OPTIONAL MATCH (p)-[:FITS_OCCASION]->(o:Occasion)
RETURN p.product_id AS product_id, p.name AS name, p.description AS description,
       p.color_palette AS color_palette,
       collect(DISTINCT a.name) AS aesthetics,
       collect(DISTINCT o.name) AS occasions
"""

FETCH_AESTHETICS = """
MATCH (a:Aesthetic)
RETURN a.name AS name, a.description AS description, a.keywords AS keywords
"""

SET_PRODUCT_EMBEDDING = """
MATCH (p:Product {product_id: $product_id})
SET p.embedding = $embedding
"""

SET_AESTHETIC_EMBEDDING = """
MATCH (a:Aesthetic {name: $name})
SET a.embedding = $embedding
"""


def embed_products(session, embedder) -> int:
    result = session.run(FETCH_PRODUCTS)
    products = list(result)
    logger.info("Fetched products count=%d", len(products))

    texts = [build_product_text(dict(p)) for p in products]
    embeddings = embedder.embed_batch(texts)

    for product, embedding in zip(products, embeddings, strict=True):
        session.run(
            SET_PRODUCT_EMBEDDING,
            {
                "product_id": product["product_id"],
                "embedding": embedding,
            },
        )

    logger.info("Embedded products count=%d", len(products))
    return len(products)


def embed_aesthetics(session, embedder) -> int:
    result = session.run(FETCH_AESTHETICS)
    aesthetics = list(result)
    logger.info("Fetched aesthetics count=%d", len(aesthetics))

    texts = [build_aesthetic_text(dict(a)) for a in aesthetics]
    embeddings = embedder.embed_batch(texts)

    for aesthetic, embedding in zip(aesthetics, embeddings, strict=True):
        session.run(
            SET_AESTHETIC_EMBEDDING,
            {
                "name": aesthetic["name"],
                "embedding": embedding,
            },
        )

    logger.info("Embedded aesthetics count=%d", len(aesthetics))
    return len(aesthetics)


def main() -> None:
    from stylemind.config import get_config
    from stylemind.rag.embedder import get_embedder

    config = get_config()
    embedder = get_embedder(config.embedding)
    logger.info(
        "Using embedder provider=%s model=%s dims=%d",
        config.embedding.provider,
        config.embedding.model_name,
        embedder.dimensions,
    )

    nc = config.neo4j
    driver = GraphDatabase.driver(nc.uri, auth=(nc.user, nc.password))
    try:
        with driver.session() as session:
            n_products = embed_products(session, embedder)
            n_aesthetics = embed_aesthetics(session, embedder)
        logger.info("Embed complete products=%d aesthetics=%d", n_products, n_aesthetics)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
