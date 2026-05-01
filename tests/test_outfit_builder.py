from __future__ import annotations

from unittest.mock import MagicMock  # noqa: I001

import pytest

from stylemind.models.schemas import OutfitSuggestion, PersonaSnapshot
from stylemind.outfit.builder import OutfitBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_driver(query_side_effects: list) -> MagicMock:
    """Build a mock neo4j Driver whose session().run() returns successive record lists.

    Each element in query_side_effects is a list of dicts that the corresponding
    `session.run()` call should return (via record.data()).
    """
    driver = MagicMock()
    session_mock = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session_mock)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    # Each call to session.run() returns a different list of records
    run_return_values = []
    for records_data in query_side_effects:
        result_mock = MagicMock()
        record_mocks = []
        for d in records_data:
            rec = MagicMock()
            rec.data.return_value = d
            record_mocks.append(rec)
        result_mock.__iter__ = MagicMock(return_value=iter(record_mocks))
        run_return_values.append(result_mock)

    session_mock.run.side_effect = run_return_values
    return driver


def _anchor_row(
    product_id: str = "P001",
    name: str = "Anchor Top",
    category: str = "Tops",
    brand: str = "BrandA",
    price_inr: int = 1000,
    occasions: list[str] | None = None,
    seasons: list[str] | None = None,
    aesthetics: list[str] | None = None,
) -> dict:
    return {
        "product_id": product_id,
        "name": name,
        "category": category,
        "brand": brand,
        "price_inr": price_inr,
        "occasions": occasions if occasions is not None else ["Casual"],
        "seasons": seasons if seasons is not None else ["SS"],
        "aesthetics": aesthetics if aesthetics is not None else ["Casual Minimalism"],
    }


def _candidate_row(
    product_id: str = "P002",
    name: str = "Paired Bottom",
    category: str = "Bottoms",
    brand: str = "BrandB",
    price_inr: int = 1500,
    occasions: list[str] | None = None,
    seasons: list[str] | None = None,
    aesthetics: list[str] | None = None,
    path_type: str = "PAIRS_WITH",
) -> dict:
    return {
        "product_id": product_id,
        "name": name,
        "category": category,
        "brand": brand,
        "price_inr": price_inr,
        "occasions": occasions if occasions is not None else ["Casual"],
        "seasons": seasons if seasons is not None else ["SS"],
        "aesthetics": aesthetics if aesthetics is not None else ["Casual Minimalism"],
        "path_type": path_type,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coherence_rejects_season_clash():
    """Anchor is SS-only; AW-only candidate excluded by Cypher → no items → ValueError.

    With the minimum-outfit-size enforcement, build_outfit raises ValueError when
    both the PAIRS_WITH and aesthetic-fallback queries return zero candidates so
    the caller (API layer) can handle the case gracefully.
    """
    driver = _make_driver(
        [
            [_anchor_row(seasons=["SS"])],  # GET_ANCHOR_PRODUCT
            [],  # GET_PAIRS_WITH_COHERENT → empty (AW candidate excluded)
            [],  # GET_AESTHETIC_FALLBACK → empty
        ]
    )
    builder = OutfitBuilder(driver)
    with pytest.raises(ValueError, match="No coherent paired items"):
        builder.build_outfit("P001", "user1")


@pytest.mark.unit
def test_coherence_rejects_occasion_clash():
    """Office anchor + Active-only candidate excluded → no items → ValueError."""
    driver = _make_driver(
        [
            [_anchor_row(occasions=["Office"])],  # anchor
            [],  # PAIRS_WITH empty (Active candidate filtered by Cypher)
            [],  # fallback empty
        ]
    )
    builder = OutfitBuilder(driver)
    with pytest.raises(ValueError, match="No coherent paired items"):
        builder.build_outfit("P001", "user1")


@pytest.mark.unit
def test_category_diversification_max_one_per_slot():
    """Two Top candidates → only one selected; the other (same category as anchor excluded too)."""
    # Anchor is Tops. Two candidates are both Tops → both should be excluded by diversification.
    anchor = _anchor_row(category="Tops")
    top_candidate_1 = _candidate_row(product_id="P002", category="Tops")
    top_candidate_2 = _candidate_row(product_id="P003", name="Another Top", category="Tops")
    bottom_candidate = _candidate_row(product_id="P004", name="Good Bottom", category="Bottoms")

    driver = _make_driver(
        [
            [anchor],
            [top_candidate_1, top_candidate_2, bottom_candidate],  # PAIRS_WITH returns all
        ]
    )
    builder = OutfitBuilder(driver)
    outfit = builder.build_outfit("P001", "user1")

    categories_in_outfit = [item.category for item in outfit.items]
    # Tops duplicates must be removed; only the Bottom should remain
    tops_count = categories_in_outfit.count("Tops")
    assert tops_count == 0, f"Expected 0 Tops items (anchor already occupies that slot), got {tops_count}"
    assert "Bottoms" in categories_in_outfit


@pytest.mark.unit
def test_fallback_triggers_when_no_pairs_with():
    """When PAIRS_WITH query returns empty, the fallback query must be called."""
    anchor = _anchor_row()
    fallback_item = _candidate_row(product_id="P010", path_type="aesthetic_similarity")

    driver = _make_driver(
        [
            [anchor],  # GET_ANCHOR_PRODUCT
            [],  # GET_PAIRS_WITH_COHERENT → empty → triggers fallback
            [fallback_item],  # GET_AESTHETIC_FALLBACK
        ]
    )
    builder = OutfitBuilder(driver)
    outfit = builder.build_outfit("P001", "user1")

    assert len(outfit.items) == 1
    assert outfit.items[0].product_id == "P010"
    # graph_path should use aesthetic notation
    assert "~aesthetic~" in outfit.items[0].graph_path


@pytest.mark.unit
def test_graph_path_string_format_pairs_with():
    """Verify PAIRS_WITH graph_path format: 'P012 -PAIRS_WITH-> P002'."""
    builder = OutfitBuilder(MagicMock())
    path = builder._make_graph_path("P012", "P002", "PAIRS_WITH")
    assert path == "P012 -PAIRS_WITH-> P002"


@pytest.mark.unit
def test_graph_path_string_format_aesthetic():
    """Verify aesthetic fallback graph_path format: 'P012 ~aesthetic~ P002'."""
    builder = OutfitBuilder(MagicMock())
    path = builder._make_graph_path("P012", "P002", "aesthetic_similarity")
    assert path == "P012 ~aesthetic~ P002"


@pytest.mark.unit
def test_persona_ranking_prefers_matching_aesthetics():
    """Candidates matching persona aesthetics should rank before non-matching ones."""
    anchor = _anchor_row()
    matching = _candidate_row(product_id="P002", category="Bottoms", aesthetics=["Old Money"])
    non_matching = _candidate_row(
        product_id="P003", name="Unrelated Shoes", category="Footwear", aesthetics=["Streetwear"]
    )

    driver = _make_driver(
        [
            [anchor],
            [non_matching, matching],  # non-matching returned first by Cypher
        ]
    )
    persona = PersonaSnapshot(preferred_aesthetics=["Old Money"])
    builder = OutfitBuilder(driver)
    outfit = builder.build_outfit("P001", "user1", persona=persona)

    # After persona ranking, matching should appear before non_matching
    product_ids = [item.product_id for item in outfit.items]
    assert product_ids.index("P002") < product_ids.index("P003")


@pytest.mark.unit
def test_outfit_items_respect_max_count():
    """Builder selects at most 4 paired items even when more candidates exist."""
    anchor = _anchor_row()
    candidates = [
        _candidate_row(product_id=f"P{i:03d}", name=f"Item {i}", category=f"Cat{i}")
        for i in range(2, 10)  # 8 candidates
    ]
    driver = _make_driver(
        [
            [anchor],
            candidates,
        ]
    )
    builder = OutfitBuilder(driver)
    outfit = builder.build_outfit("P001", "user1")

    assert len(outfit.items) <= 4


@pytest.mark.unit
def test_outfit_anchor_summary_populated():
    """Anchor ProductSummary fields are correctly populated."""
    anchor = _anchor_row(product_id="P001", name="Nice Top", brand="Zara", price_inr=2000)
    paired = _candidate_row(product_id="P002", name="Paired Bottom")
    driver = _make_driver(
        [
            [anchor],
            [paired],  # PAIRS_WITH returns one item so build succeeds
        ]
    )
    builder = OutfitBuilder(driver)
    outfit = builder.build_outfit("P001", "user1")

    assert outfit.anchor.product_id == "P001"
    assert outfit.anchor.name == "Nice Top"
    assert outfit.anchor.brand == "Zara"
    assert outfit.anchor.price_inr == 2000


# ---------------------------------------------------------------------------
# Integration test (requires running Neo4j)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_build_outfit_p012_integration():
    """Integration test: build outfit for P012 against a running Neo4j instance."""
    import os

    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        pytest.skip("NEO4J_PASSWORD not set; skipping integration test")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except ServiceUnavailable:
        pytest.skip("Neo4j not reachable; skipping integration test")

    try:
        builder = OutfitBuilder(driver)
        outfit = builder.build_outfit("P012", "integration_test_user")
        assert isinstance(outfit, OutfitSuggestion)
        assert outfit.anchor.product_id == "P012"
        assert isinstance(outfit.items, list)
        for item in outfit.items:
            assert "P012" in item.graph_path
            assert "-PAIRS_WITH->" in item.graph_path or "~aesthetic~" in item.graph_path
    finally:
        driver.close()
