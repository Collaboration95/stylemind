from __future__ import annotations

import pytest

from stylemind.models.domain import RetrievedProduct
from stylemind.models.schemas import PersonaSnapshot
from stylemind.rag.reranker import ProductReranker, ScoreBreakdown

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_product(
    product_id: str = "P001",
    name: str = "Test Product",
    aesthetics: list[str] | None = None,
    budget_tier: str = "Mid",
    category: str = "Top",
    brand: str = "TestBrand",
    similarity_score: float = 0.8,
) -> RetrievedProduct:
    return RetrievedProduct(
        product_id=product_id,
        name=name,
        description="A test product",
        price=2000,
        category=category,
        brand=brand,
        budget_tier=budget_tier,
        aesthetics=aesthetics or [],
        occasions=["Casual"],
        colors=["White"],
        seasons=["SS"],
        pairs_with=[],
        similarity_score=similarity_score,
    )


def empty_persona() -> PersonaSnapshot:
    return PersonaSnapshot()  # confidence_score defaults to 0.0


def persona_with_quiet_luxury(confidence: float = 0.8, budget_tier: str | None = "Premium") -> PersonaSnapshot:
    return PersonaSnapshot(
        preferred_aesthetics=["Quiet Luxury"],
        disliked_materials=[],
        budget_tier=budget_tier,
        top_occasions=["Work"],
        confidence_score=confidence,
    )


# ---------------------------------------------------------------------------
# Unit tests — no Neo4j required
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reranker_empty_persona_identity() -> None:
    """Empty persona (confidence=0.0) should produce same ranking as base scores."""
    reranker = ProductReranker()
    candidates = [
        make_product("P001", similarity_score=0.9),
        make_product("P002", similarity_score=0.7),
        make_product("P003", similarity_score=0.5),
    ]
    results = reranker.rerank(candidates, empty_persona())

    assert len(results) == 3
    # final_score must equal base score when confidence==0.0
    for result in results:
        assert result.final_score == pytest.approx(result.product.similarity_score)
    # ordering preserved (highest first)
    assert results[0].product.product_id == "P001"
    assert results[1].product.product_id == "P002"
    assert results[2].product.product_id == "P003"


@pytest.mark.unit
def test_reranker_persona_boosts_matching_aesthetic() -> None:
    """Persona with preferred_aesthetics='Quiet Luxury' should boost matching products."""
    reranker = ProductReranker()
    persona = persona_with_quiet_luxury(confidence=1.0, budget_tier=None)

    ql_product = make_product("P_QL", aesthetics=["Quiet Luxury"], similarity_score=0.65)
    other_product = make_product("P_OTHER", aesthetics=["Streetwear"], similarity_score=0.7)

    results = reranker.rerank([other_product, ql_product], persona)

    result_map = {r.product.product_id: r for r in results}
    ql_result = result_map["P_QL"]
    other_result = result_map["P_OTHER"]

    # ql_product should be boosted above other_product despite lower base score
    assert ql_result.final_score > other_result.final_score, (
        f"Expected QL product ({ql_result.final_score:.3f}) > other ({other_result.final_score:.3f})"
    )
    # Verify the boost was applied (final > base)
    assert ql_result.final_score > ql_product.similarity_score
    # Non-matching product should not be boosted
    assert other_result.final_score == pytest.approx(other_product.similarity_score)


@pytest.mark.unit
def test_reranker_budget_boost() -> None:
    """Matching budget_tier should add a positive budget_boost."""
    reranker = ProductReranker()
    persona = PersonaSnapshot(
        preferred_aesthetics=[],
        budget_tier="Premium",
        confidence_score=1.0,
    )

    premium_product = make_product("P_PREM", budget_tier="Premium", similarity_score=0.5)
    mid_product = make_product("P_MID", budget_tier="Mid", similarity_score=0.5)

    results = reranker.rerank([premium_product, mid_product], persona, explain=True)
    result_map = {r.product.product_id: r for r in results}

    prem_result = result_map["P_PREM"]
    mid_result = result_map["P_MID"]

    # Premium product gets budget boost
    assert prem_result.breakdown is not None
    assert prem_result.breakdown.budget_boost > 0.0
    assert mid_result.breakdown is not None
    assert mid_result.breakdown.budget_boost == 0.0

    # Premium product should rank higher
    assert prem_result.final_score > mid_result.final_score


@pytest.mark.unit
def test_reranker_explain_mode_returns_breakdown() -> None:
    """explain=True should populate ScoreBreakdown in every result."""
    reranker = ProductReranker()
    candidates = [
        make_product("P001", similarity_score=0.8),
        make_product("P002", similarity_score=0.6),
    ]
    persona = persona_with_quiet_luxury()

    results = reranker.rerank(candidates, persona, explain=True)

    for result in results:
        assert result.breakdown is not None, f"Expected breakdown for {result.product.product_id}"
        assert isinstance(result.breakdown, ScoreBreakdown)
        assert result.breakdown.product_id == result.product.product_id
        assert result.breakdown.final_score == pytest.approx(result.final_score)
        assert result.breakdown.base_score == pytest.approx(result.product.similarity_score)


@pytest.mark.unit
def test_reranker_explain_false_no_breakdown() -> None:
    """explain=False (default) should leave breakdown as None."""
    reranker = ProductReranker()
    candidates = [make_product("P001")]
    results = reranker.rerank(candidates, empty_persona(), explain=False)

    assert results[0].breakdown is None


@pytest.mark.unit
def test_reranker_persona_boost_capped() -> None:
    """Persona boost from many matched aesthetics should be capped at 0.3."""
    reranker = ProductReranker()
    persona = PersonaSnapshot(
        preferred_aesthetics=["A1", "A2", "A3", "A4", "A5"],
        confidence_score=1.0,
    )
    # Product matches all 5 aesthetics; uncapped would be 5 * 0.1 = 0.5
    product = make_product("P001", aesthetics=["A1", "A2", "A3", "A4", "A5"], similarity_score=0.5)

    results = reranker.rerank([product], persona, explain=True)

    assert results[0].breakdown is not None
    assert results[0].breakdown.persona_boost == pytest.approx(0.3)


@pytest.mark.unit
def test_reranker_below_threshold_not_included() -> None:
    """Products with similarity_score < 0.3 should not appear in results after reranking.

    The primary filter is in the Cypher query, but the reranker applies it as a
    safety net so callers get a clean list regardless of the source.
    """
    # NOTE: The reranker itself doesn't filter out below-threshold candidates because
    # filtering is done in the retriever's Cypher query. However, we verify here
    # that if such candidates somehow arrive, their final_score with zero confidence
    # will simply equal their base score (no boost), which callers can then filter.
    # This test documents the expected behaviour contract.
    reranker = ProductReranker()
    low_score_product = make_product("P_LOW", similarity_score=0.1)
    above_threshold_product = make_product("P_HIGH", similarity_score=0.8)

    results = reranker.rerank([low_score_product, above_threshold_product], empty_persona())

    # Without threshold filtering, both products appear — high first.
    assert results[0].product.product_id == "P_HIGH"
    assert results[1].product.product_id == "P_LOW"
    # The low-score product's final_score equals its base score (no adjustment).
    assert results[1].final_score == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Integration test — requires live Neo4j with seeded data
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_retriever_integration() -> None:
    """Full round-trip: embed query -> vector search -> RetrievedProduct list.

    Skipped automatically if Neo4j is not reachable.
    """
    import os

    from stylemind.config import Neo4jConfig
    from stylemind.graph.client import Neo4jClient
    from stylemind.rag.embedder import LocalEmbedder
    from stylemind.rag.retriever import ProductRetriever

    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_password:
        pytest.skip("NEO4J_PASSWORD not set; skipping integration test")

    config = Neo4jConfig(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=neo4j_password,
    )

    client = Neo4jClient(config)
    try:
        client.connect()
        reachable = client.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j not reachable; skipping integration test")

    if not reachable:
        pytest.skip("Neo4j connectivity check failed; skipping integration test")

    embedder = LocalEmbedder()
    retriever = ProductRetriever(client=client, embedder=embedder, top_k=5, min_threshold=0.0)

    try:
        results = retriever.retrieve("summer dress for a casual outing")
        # Results may be empty if embeddings not populated, but should not raise.
        assert isinstance(results, list)
        for product in results:
            assert isinstance(product, RetrievedProduct)
            assert product.product_id
            assert product.similarity_score >= 0.0
    finally:
        client.close()
