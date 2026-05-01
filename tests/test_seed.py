from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Unit tests — no Neo4j required
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rtl_parser_all_rows_14_fields() -> None:
    """RTL parser on all 45 rows produces exactly 14 fields each."""
    from seed import parse_csv  # type: ignore[import]

    csv_path = Path("data/products_seed.csv")
    products = parse_csv(csv_path)
    assert len(products) == 45
    for p in products:
        assert len(p) == 14, f"Product {p.get('product_id')} has {len(p)} fields"


@pytest.mark.unit
def test_problematic_rows_parse_correctly() -> None:
    """Rows P013, P030, P038, P042 have commas in description — must parse cleanly."""
    from seed import parse_csv  # type: ignore[import]

    csv_path = Path("data/products_seed.csv")
    products = parse_csv(csv_path)
    by_id = {p["product_id"]: p for p in products}

    for pid in ["P013", "P030", "P038", "P042"]:
        assert pid in by_id, f"Product {pid} not found"
        assert len(by_id[pid]["description"]) > 10, f"Description too short for {pid}"
        # pairs_with should be clean product IDs, not part of description
        pw = by_id[pid]["pairs_with"]
        for pair_id in pw.split("|"):
            assert pair_id.strip().startswith("P"), f"Bad pairs_with in {pid}: {pair_id!r}"


@pytest.mark.unit
def test_aesthetic_remapping() -> None:
    """P037 and P038 aesthetic 'Casual' must be remapped to 'Casual Minimalism'."""
    from seed import parse_csv  # type: ignore[import]

    csv_path = Path("data/products_seed.csv")
    products = parse_csv(csv_path)
    by_id = {p["product_id"]: p for p in products}
    # P037 raw: "Y2K|Casual" -> "Y2K|Casual Minimalism"
    assert "Casual Minimalism" in by_id["P037"]["aesthetic"]
    assert "Casual" not in by_id["P037"]["aesthetic"].replace("Casual Minimalism", "")
    # P038 raw: "Streetwear|Casual" -> "Streetwear|Casual Minimalism"
    assert "Casual Minimalism" in by_id["P038"]["aesthetic"]
    assert "Casual" not in by_id["P038"]["aesthetic"].replace("Casual Minimalism", "")


@pytest.mark.unit
def test_compute_overlaps_with() -> None:
    """compute_overlaps_with returns non-empty list of valid aesthetic pairs."""
    from seed import compute_overlaps_with, parse_csv  # type: ignore[import]

    from data.enrichment import SYNTHETIC_PRODUCTS

    csv_path = Path("data/products_seed.csv")
    products = parse_csv(csv_path)
    # Add synthetic products (ensure price_inr is int)
    for sp in SYNTHETIC_PRODUCTS:
        p = sp.copy()
        p["price_inr"] = int(p["price_inr"])
        products.append(p)

    overlaps = compute_overlaps_with(products)
    assert len(overlaps) > 0, "Expected at least one overlapping aesthetic pair"
    for a, b in overlaps:
        assert isinstance(a, str)
        assert isinstance(b, str)
        assert a != b, f"Self-overlap detected: {a}"


@pytest.mark.unit
def test_pipe_separated_materials_produce_multiple_entries() -> None:
    """Products with pipe-separated materials should create separate MADE_FROM entries for each."""
    from seed import parse_csv  # type: ignore[import]

    from data.enrichment import MATERIAL_METADATA

    csv_path = Path("data/products_seed.csv")
    products = parse_csv(csv_path)
    pipe_products = [p for p in products if "|" in p["material"]]
    assert len(pipe_products) > 0, "Expected at least one product with pipe-separated materials"
    for p in pipe_products:
        materials = [m.strip() for m in p["material"].split("|")]
        for mat in materials:
            assert mat in MATERIAL_METADATA, f"Material {mat!r} not in MATERIAL_METADATA (product {p['product_id']})"


@pytest.mark.unit
def test_pipe_separated_color_palettes_valid() -> None:
    """Products with pipe-separated color_palette should have all palettes in COLOR_PALETTE_METADATA."""
    from seed import parse_csv  # type: ignore[import]

    from data.enrichment import COLOR_PALETTE_METADATA

    csv_path = Path("data/products_seed.csv")
    products = parse_csv(csv_path)
    pipe_products = [p for p in products if "|" in p["color_palette"]]
    assert len(pipe_products) > 0, "Expected at least one product with pipe-separated color palettes"
    for p in pipe_products:
        palettes = [cp.strip() for cp in p["color_palette"].split("|")]
        assert len(palettes) >= 2, f"Expected >= 2 palettes for {p['product_id']}, got {palettes}"
        for cp in palettes:
            assert cp in COLOR_PALETTE_METADATA, (
                f"Color palette {cp!r} not in COLOR_PALETTE_METADATA (product {p['product_id']})"
            )
