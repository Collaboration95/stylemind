from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src/ to path so `from stylemind.xxx import ...` works without installing.
# Add project root to path so `from data.enrichment import ...` works.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from neo4j import GraphDatabase  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


CSV_PATH = Path(__file__).parent.parent / "data" / "products_seed.csv"


def parse_row(raw_line: str) -> list[str]:
    """RTL CSV parser: handles unquoted commas in description (rows P013, P030, P038, P042)."""
    tokens = raw_line.strip().split(",")
    if len(tokens) <= 14:
        return tokens
    first_12 = tokens[:12]
    pairs_with = tokens[-1]
    description = ",".join(tokens[12:-1])
    return first_12 + [description, pairs_with]


def parse_csv(csv_path: Path) -> list[dict]:
    """Parse products_seed.csv, returning list of product dicts."""
    products = []
    with open(csv_path) as f:
        lines = f.readlines()

    header = [h.strip() for h in lines[0].split(",")]

    for line in lines[1:]:
        if not line.strip():
            continue
        tokens = parse_row(line)
        if len(tokens) != 14:
            logger.warning("Unexpected column count=%d for line=%s", len(tokens), line[:80])
            continue
        row = dict(zip(header, [t.strip() for t in tokens], strict=True))

        # Remap aesthetic token "Casual" -> "Casual Minimalism" for P037, P038
        if row["product_id"] in ("P037", "P038"):
            parts = [a.strip() for a in row["aesthetic"].split("|")]
            parts = ["Casual Minimalism" if a == "Casual" else a for a in parts]
            row["aesthetic"] = "|".join(parts)

        # Parse price as int
        row["price_inr"] = int(row["price_inr"])

        products.append(row)

    return products


def compute_overlaps_with(products: list[dict]) -> list[tuple[str, str]]:
    """Derive OVERLAPS_WITH from CSV co-occurrence: aesthetics sharing >= 2 products."""
    from collections import defaultdict

    aesthetic_products: dict[str, set[str]] = defaultdict(set)
    for p in products:
        for aesthetic in p["aesthetic"].split("|"):
            aesthetic = aesthetic.strip()
            if aesthetic:
                aesthetic_products[aesthetic].add(p["product_id"])

    aesthetics = list(aesthetic_products.keys())
    overlaps = []
    for i in range(len(aesthetics)):
        for j in range(i + 1, len(aesthetics)):
            a, b = aesthetics[i], aesthetics[j]
            shared = aesthetic_products[a] & aesthetic_products[b]
            if len(shared) >= 2:
                overlaps.append((a, b))
    return overlaps


def seed(driver) -> None:  # type: ignore[type-arg]
    """Run all MERGE operations. Safe to call multiple times (idempotent)."""
    from data.enrichment import (
        AESTHETIC_METADATA,
        BODY_TYPE_METADATA,
        BRAND_METADATA,
        BUDGET_TIER_RANGES,
        COLOR_PALETTE_METADATA,
        MATERIAL_METADATA,
        OCCASION_METADATA,
        SYNTHETIC_PRODUCTS,
    )
    from stylemind.graph import queries as Q

    products = parse_csv(CSV_PATH)
    logger.info("Parsed csv_products=%d", len(products))

    # Add synthetic products (ensure price_inr is int)
    synthetic = []
    for sp in SYNTHETIC_PRODUCTS:
        p = sp.copy()
        p["price_inr"] = int(p["price_inr"])
        synthetic.append(p)
    all_products = products + synthetic
    logger.info(
        "Total products=%d (csv=%d synthetic=%d)",
        len(all_products),
        len(products),
        len(synthetic),
    )

    with driver.session() as session:
        # ------------------------------------------------------------------
        # 1. Nodes
        # ------------------------------------------------------------------

        # Products
        for p in all_products:
            session.run(
                Q.MERGE_PRODUCT,
                {
                    "product_id": p["product_id"],
                    "name": p["name"],
                    "description": p["description"],
                    "price_inr": int(p["price_inr"]),
                    "category": p["category"],
                    "color_palette": p["color_palette"],
                    "material": p["material"],
                    "season": p["season"],
                },
            )
        logger.info("Merged products count=%d", len(all_products))

        # Brands
        for brand_data in BRAND_METADATA.values():
            session.run(Q.MERGE_BRAND, brand_data)
        logger.info("Merged brands count=%d", len(BRAND_METADATA))

        # Aesthetics
        for aesthetic_data in AESTHETIC_METADATA.values():
            session.run(
                Q.MERGE_AESTHETIC,
                {
                    "name": aesthetic_data["name"],
                    "description": aesthetic_data["description"],
                    "keywords": aesthetic_data["keywords"],
                },
            )
        logger.info("Merged aesthetics count=%d", len(AESTHETIC_METADATA))

        # Occasions
        for occ_data in OCCASION_METADATA.values():
            session.run(Q.MERGE_OCCASION, occ_data)
        logger.info("Merged occasions count=%d", len(OCCASION_METADATA))

        # Body types
        for bt_data in BODY_TYPE_METADATA.values():
            session.run(Q.MERGE_BODY_TYPE, bt_data)
        logger.info("Merged body_types count=%d", len(BODY_TYPE_METADATA))

        # Color palettes
        for cp_data in COLOR_PALETTE_METADATA.values():
            session.run(
                Q.MERGE_COLOR_PALETTE,
                {
                    "name": cp_data["name"],
                    "hex_codes": cp_data["hex_codes"],
                },
            )
        logger.info("Merged color_palettes count=%d", len(COLOR_PALETTE_METADATA))

        # Materials
        for mat_data in MATERIAL_METADATA.values():
            session.run(Q.MERGE_MATERIAL, mat_data)
        logger.info("Merged materials count=%d", len(MATERIAL_METADATA))

        # Seasons
        for season_name in ["SS", "AW", "Year-round"]:
            session.run(Q.MERGE_SEASON, {"name": season_name})
        logger.info("Merged seasons count=3")

        # Budget tiers
        for tier_data in BUDGET_TIER_RANGES.values():
            session.run(Q.MERGE_BUDGET_TIER, tier_data)
        logger.info("Merged budget_tiers count=%d", len(BUDGET_TIER_RANGES))

        # ------------------------------------------------------------------
        # 2. Relationships
        # ------------------------------------------------------------------

        for p in all_products:
            pid = p["product_id"]

            # BELONGS_TO Brand
            session.run(Q.REL_PRODUCT_BELONGS_TO_BRAND, {"product_id": pid, "brand_name": p["brand"]})

            # EMBODIES Aesthetic(s) — single value or pipe-separated
            for aesthetic in p["aesthetic"].split("|"):
                aesthetic = aesthetic.strip()
                if aesthetic in AESTHETIC_METADATA:
                    session.run(Q.REL_PRODUCT_EMBODIES_AESTHETIC, {"product_id": pid, "aesthetic_name": aesthetic})

            # FITS_OCCASION — pipe-separated
            for occ in p["occasion"].split("|"):
                occ = occ.strip()
                if occ in OCCASION_METADATA:
                    session.run(Q.REL_PRODUCT_FITS_OCCASION, {"product_id": pid, "occasion_name": occ})

            # SUITS_BODY — pipe-separated
            for bt in p["body_type_fit"].split("|"):
                bt = bt.strip()
                if bt in BODY_TYPE_METADATA:
                    session.run(Q.REL_PRODUCT_SUITS_BODY, {"product_id": pid, "body_type_name": bt})

            # MADE_FROM Material
            mat = p["material"].strip()
            if mat in MATERIAL_METADATA:
                session.run(Q.REL_PRODUCT_MADE_FROM, {"product_id": pid, "material_name": mat})

            # BEST_IN_SEASON — pipe-separated
            for season in p["season"].split("|"):
                season = season.strip()
                if season:
                    session.run(Q.REL_PRODUCT_BEST_IN_SEASON, {"product_id": pid, "season_name": season})

            # IN_COLOR ColorPalette (take first palette if pipe-separated)
            cp = p["color_palette"].split("|")[0].strip()
            if cp:
                session.run(Q.REL_PRODUCT_IN_COLOR, {"product_id": pid, "palette_name": cp})

            # AT_TIER BudgetTier
            tier = p["budget_tier"].strip()
            session.run(Q.REL_PRODUCT_AT_TIER, {"product_id": pid, "tier_label": tier})

        # PAIRS_WITH: collect all canonical pairs, then MERGE one directed edge per pair.
        # Use sorted product IDs to ensure smaller-id → larger-id direction (deduplication).
        pairs: set[tuple[str, str]] = set()
        for p in all_products:
            pid = p["product_id"]
            pw = str(p.get("pairs_with", "")).strip()
            if not pw:
                continue
            for other_id in pw.split("|"):
                other_id = other_id.strip()
                if not other_id:
                    continue
                canonical = tuple(sorted([pid, other_id]))
                pairs.add(canonical)  # type: ignore[arg-type]

        for a_id, b_id in pairs:
            session.run(Q.MERGE_PAIRS_WITH, {"product_id_a": a_id, "product_id_b": b_id})
        logger.info("Merged PAIRS_WITH edges count=%d", len(pairs))

        # Brand KNOWN_FOR Aesthetic (derive from products: which aesthetics does each brand carry)
        brand_aesthetics: dict[str, set[str]] = {}
        for p in all_products:
            brand = p["brand"]
            for aesthetic in p["aesthetic"].split("|"):
                aesthetic = aesthetic.strip()
                if aesthetic in AESTHETIC_METADATA:
                    brand_aesthetics.setdefault(brand, set()).add(aesthetic)

        for brand_name, aesthetics in brand_aesthetics.items():
            for aesthetic_name in aesthetics:
                session.run(
                    Q.REL_BRAND_KNOWN_FOR_AESTHETIC,
                    {"brand_name": brand_name, "aesthetic_name": aesthetic_name},
                )
        logger.info("Merged KNOWN_FOR relationships")

        # Brand AT_TIER
        for brand_data in BRAND_METADATA.values():
            session.run(
                Q.REL_BRAND_AT_TIER,
                {"brand_name": brand_data["name"], "tier_label": brand_data["tier"]},
            )
        logger.info("Merged brand AT_TIER relationships")

        # OVERLAPS_WITH: derived from co-occurrence (bidirectional)
        overlaps = compute_overlaps_with(all_products)
        for aesthetic_a, aesthetic_b in overlaps:
            session.run(Q.MERGE_OVERLAPS_WITH, {"aesthetic_a": aesthetic_a, "aesthetic_b": aesthetic_b})
            session.run(Q.MERGE_OVERLAPS_WITH, {"aesthetic_a": aesthetic_b, "aesthetic_b": aesthetic_a})
        logger.info("Merged OVERLAPS_WITH relationships count=%d", len(overlaps) * 2)

        # Vector indexes (embedding property populated later by embed.py)
        session.run(Q.CREATE_PRODUCT_VECTOR_INDEX)
        session.run(Q.CREATE_AESTHETIC_VECTOR_INDEX)
        logger.info("Vector indexes created (if not exists)")

    logger.info("Seed complete")


def main() -> None:
    from stylemind.config import get_config

    config = get_config()
    nc = config.neo4j
    logger.info("Connecting to neo4j uri=%s", nc.uri)
    driver = GraphDatabase.driver(nc.uri, auth=(nc.user, nc.password))
    try:
        seed(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
