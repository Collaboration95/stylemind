from __future__ import annotations

import logging

from neo4j import Driver

from stylemind.models.schemas import OutfitItemSchema, OutfitSuggestion, PersonaSnapshot, ProductSummary
from stylemind.observability import observe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

GET_ANCHOR_PRODUCT = """
MATCH (p:Product {product_id: $product_id})
MATCH (p)-[:BELONGS_TO]->(b:Brand)
OPTIONAL MATCH (p)-[:FITS_OCCASION]->(o:Occasion)
OPTIONAL MATCH (p)-[:BEST_IN_SEASON]->(s:Season)
OPTIONAL MATCH (p)-[:EMBODIES]->(a:Aesthetic)
RETURN p.product_id AS product_id, p.name AS name, p.category AS category,
       b.name AS brand, p.price_inr AS price_inr,
       collect(DISTINCT o.name) AS occasions,
       collect(DISTINCT s.name) AS seasons,
       collect(DISTINCT a.name) AS aesthetics
"""

GET_PAIRS_WITH_COHERENT = """
MATCH (anchor:Product {product_id: $product_id})
OPTIONAL MATCH (anchor)-[:FITS_OCCASION]->(ao:Occasion)
OPTIONAL MATCH (anchor)-[:BEST_IN_SEASON]->(as:Season)
WITH anchor, collect(DISTINCT ao.name) AS anchor_occasions, collect(DISTINCT as.name) AS anchor_seasons
MATCH (anchor)-[:PAIRS_WITH]-(paired:Product)
OPTIONAL MATCH (paired)-[:FITS_OCCASION]->(po:Occasion)
OPTIONAL MATCH (paired)-[:BEST_IN_SEASON]->(ps:Season)
OPTIONAL MATCH (paired)-[:BELONGS_TO]->(pb:Brand)
OPTIONAL MATCH (paired)-[:EMBODIES]->(pa:Aesthetic)
WITH anchor, anchor_occasions, anchor_seasons, paired, pb,
     collect(DISTINCT po.name) AS paired_occasions,
     collect(DISTINCT ps.name) AS paired_seasons,
     collect(DISTINCT pa.name) AS paired_aesthetics
WHERE size([x IN paired_occasions WHERE x IN anchor_occasions]) > 0
  AND size([x IN paired_seasons WHERE x IN anchor_seasons]) > 0
RETURN paired.product_id AS product_id,
       paired.name AS name,
       paired.category AS category,
       pb.name AS brand,
       paired.price_inr AS price_inr,
       paired_occasions AS occasions,
       paired_seasons AS seasons,
       paired_aesthetics AS aesthetics,
       'PAIRS_WITH' AS path_type
"""

GET_AESTHETIC_FALLBACK = """
MATCH (anchor:Product {product_id: $product_id})
OPTIONAL MATCH (anchor)-[:FITS_OCCASION]->(ao:Occasion)
OPTIONAL MATCH (anchor)-[:BEST_IN_SEASON]->(as:Season)
OPTIONAL MATCH (anchor)-[:EMBODIES]->(aa:Aesthetic)
WITH anchor, collect(DISTINCT ao.name) AS anchor_occasions,
     collect(DISTINCT as.name) AS anchor_seasons,
     collect(DISTINCT aa.name) AS anchor_aesthetics
MATCH (candidate:Product)
WHERE candidate.product_id <> anchor.product_id
OPTIONAL MATCH (candidate)-[:FITS_OCCASION]->(co:Occasion)
OPTIONAL MATCH (candidate)-[:BEST_IN_SEASON]->(cs:Season)
OPTIONAL MATCH (candidate)-[:EMBODIES]->(ca:Aesthetic)
OPTIONAL MATCH (candidate)-[:BELONGS_TO]->(cb:Brand)
WITH anchor, anchor_occasions, anchor_seasons, anchor_aesthetics,
     candidate, cb,
     collect(DISTINCT co.name) AS cand_occasions,
     collect(DISTINCT cs.name) AS cand_seasons,
     collect(DISTINCT ca.name) AS cand_aesthetics
WHERE size([x IN cand_occasions WHERE x IN anchor_occasions]) > 0
  AND size([x IN cand_seasons WHERE x IN anchor_seasons]) > 0
  AND size([x IN cand_aesthetics WHERE x IN anchor_aesthetics]) > 0
RETURN candidate.product_id AS product_id,
       candidate.name AS name,
       candidate.category AS category,
       cb.name AS brand,
       candidate.price_inr AS price_inr,
       cand_occasions AS occasions,
       cand_seasons AS seasons,
       cand_aesthetics AS aesthetics,
       'aesthetic_similarity' AS path_type
LIMIT 10
"""

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

MAX_OUTFIT_ITEMS = 4


class OutfitBuilder:
    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    @observe(name="build_outfit")
    def build_outfit(
        self,
        product_id: str,
        user_id: str,
        persona: PersonaSnapshot | None = None,
    ) -> OutfitSuggestion:
        """Build a coherent outfit starting from an anchor product.

        Steps:
        1. Get anchor product details from graph.
        2. Get PAIRS_WITH coherent candidates (season + occasion overlap).
        3. If no candidates: fall back to aesthetic similarity.
        4. Diversify by category slot (max 1 per slot).
        5. Rank by persona if available.
        6. Select 2-4 paired items.
        7. Return OutfitSuggestion.
        """
        logger.info("outfit build_outfit product_id=%s user_id=%s", product_id, user_id)

        # 1. Fetch anchor
        anchor_rows = self._run_query(GET_ANCHOR_PRODUCT, {"product_id": product_id})
        if not anchor_rows:
            raise ValueError(f"Product not found: product_id={product_id}")

        anchor_row = anchor_rows[0]
        anchor_occasions: list[str] = anchor_row.get("occasions") or []
        anchor_seasons: list[str] = anchor_row.get("seasons") or []
        anchor_aesthetics: list[str] = anchor_row.get("aesthetics") or []

        logger.debug(
            "outfit anchor product_id=%s occasions=%s seasons=%s",
            product_id,
            anchor_occasions,
            anchor_seasons,
        )

        # 2. PAIRS_WITH traversal
        candidates = self._run_query(GET_PAIRS_WITH_COHERENT, {"product_id": product_id})
        used_fallback = False

        # 3. Fallback when PAIRS_WITH yields nothing
        if not candidates:
            logger.info("outfit no PAIRS_WITH candidates, using aesthetic fallback product_id=%s", product_id)
            candidates = self._run_query(GET_AESTHETIC_FALLBACK, {"product_id": product_id})
            used_fallback = True

        logger.debug("outfit candidates_count=%d used_fallback=%s", len(candidates), used_fallback)

        # 4. Persona ranking (before diversification to preserve ranked order)
        if persona and persona.preferred_aesthetics:
            candidates = self._rank_by_persona(candidates, persona)

        # 5. Diversify by category slot (max 1 per slot, skip anchor's category slot)
        anchor_category = anchor_row.get("category", "")
        diverse_candidates = self._diversify_by_category(candidates, anchor_category)

        # 6. Select 2-4 paired items
        selected = diverse_candidates[:MAX_OUTFIT_ITEMS]

        # Build output items
        items: list[OutfitItemSchema] = []
        for item in selected:
            path_type: str = item.get("path_type", "aesthetic_similarity")
            items.append(
                OutfitItemSchema(
                    product_id=item["product_id"],
                    name=item["name"],
                    category=item.get("category") or "",
                    brand=item.get("brand") or "",
                    price_inr=item.get("price_inr") or 0,
                    justification=self._make_justification(anchor_row["name"], item, path_type),
                    graph_path=self._make_graph_path(product_id, item["product_id"], path_type),
                )
            )

        anchor_summary = ProductSummary(
            product_id=anchor_row["product_id"],
            name=anchor_row["name"],
            brand=anchor_row.get("brand") or "",
            category=anchor_row.get("category") or "",
            price_inr=anchor_row.get("price_inr") or 0,
            aesthetics=anchor_aesthetics,
        )

        occasion_str = anchor_occasions[0] if anchor_occasions else "Casual"
        season_str = anchor_seasons[0] if anchor_seasons else "Year-round"

        logger.info(
            "outfit built anchor=%s items_count=%d occasion=%s season=%s",
            product_id,
            len(items),
            occasion_str,
            season_str,
        )

        return OutfitSuggestion(
            anchor=anchor_summary,
            items=items,
            occasion=occasion_str,
            season=season_str,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_query(self, query: str, params: dict) -> list[dict]:  # type: ignore[type-arg]
        """Execute a Cypher query via the driver and return list of dicts."""
        with self._driver.session() as session:
            result = session.run(query, params)  # type: ignore[arg-type]
            return [record.data() for record in result]

    def _rank_by_persona(self, candidates: list[dict], persona: PersonaSnapshot) -> list[dict]:  # type: ignore[type-arg]
        """Sort candidates by how well their aesthetics match persona.preferred_aesthetics.

        Candidates with more overlapping aesthetics appear first.
        """
        preferred = set(persona.preferred_aesthetics)

        def overlap_score(item: dict) -> int:  # type: ignore[type-arg]
            item_aesthetics: list[str] = item.get("aesthetics") or []
            return len(set(item_aesthetics) & preferred)

        return sorted(candidates, key=overlap_score, reverse=True)

    def _diversify_by_category(self, candidates: list[dict], anchor_category: str) -> list[dict]:  # type: ignore[type-arg]
        """Return at most one candidate per category slot.

        The anchor already occupies its own category slot, so additional items
        in the same slot are excluded.
        """
        seen_categories: set[str] = {anchor_category} if anchor_category else set()
        result: list[dict] = []  # type: ignore[type-arg]

        for item in candidates:
            cat: str = item.get("category") or ""
            if cat not in seen_categories:
                seen_categories.add(cat)
                result.append(item)

        return result

    def _make_graph_path(self, anchor_id: str, paired_id: str, path_type: str) -> str:
        """Format the graph traversal path for display."""
        if path_type == "PAIRS_WITH":
            return f"{anchor_id} -PAIRS_WITH-> {paired_id}"
        return f"{anchor_id} ~aesthetic~ {paired_id}"

    def _make_justification(self, anchor_name: str, item: dict, path_type: str) -> str:  # type: ignore[type-arg]
        """Generate a simple rule-based justification string."""
        item_occasions: list[str] = item.get("occasions") or []
        item_aesthetics: list[str] = item.get("aesthetics") or []

        if path_type == "PAIRS_WITH":
            occasion_hint = f" for {item_occasions[0]}" if item_occasions else ""
            return f"Directly pairs with {anchor_name}{occasion_hint} via PAIRS_WITH relationship."

        # Aesthetic fallback
        aesthetic_hint = f" ({', '.join(item_aesthetics[:2])} aesthetic)" if item_aesthetics else ""
        return f"Complements {anchor_name}{aesthetic_hint} through shared aesthetic and occasion overlap."
