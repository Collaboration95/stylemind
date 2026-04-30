from __future__ import annotations

import pytest

from stylemind.models.domain import RetrievedProduct
from stylemind.models.schemas import PersonaSnapshot
from stylemind.rag.reranker import ProductReranker


def make_product(
    product_id: str = "P001",
    name: str = "Test Product",
    aesthetics: list[str] | None = None,
    materials: list[str] | None = None,
    budget_tier: str = "Mid",
    category: str = "Top",
    brand: str = "TestBrand",
    similarity_score: float = 0.8,
) -> RetrievedProduct:
    return RetrievedProduct(
        product_id=product_id,
        name=name,
        description="A test product",
        price_inr=2000,
        category=category,
        brand=brand,
        budget_tier=budget_tier,
        aesthetics=aesthetics or [],
        occasions=["Casual"],
        colors=["White"],
        seasons=["SS"],
        materials=materials or [],
        pairs_with=[],
        similarity_score=similarity_score,
    )


@pytest.mark.unit
def test_disliked_material_hard_filtered() -> None:
    reranker = ProductReranker()
    persona = PersonaSnapshot(
        disliked_materials=["Polyester"],
        confidence_score=1.0,
    )
    polyester_product = make_product("P_POLY", materials=["Polyester", "Cotton"], similarity_score=0.8)
    clean_product = make_product("P_CLEAN", materials=["Cotton", "Silk"], similarity_score=0.8)

    results = reranker.rerank([polyester_product, clean_product], persona, explain=True)

    assert len(results) == 1
    assert results[0].product.product_id == "P_CLEAN"


@pytest.mark.unit
def test_disliked_material_filter_case_insensitive() -> None:
    reranker = ProductReranker()
    persona = PersonaSnapshot(
        disliked_materials=["polyester"],
        confidence_score=1.0,
    )
    product = make_product("P_POLY", materials=["Polyester"], similarity_score=0.8)

    results = reranker.rerank([product], persona, explain=True)

    assert len(results) == 0


@pytest.mark.unit
def test_material_penalty_no_match_no_penalty() -> None:
    reranker = ProductReranker()
    persona = PersonaSnapshot(
        disliked_materials=["Polyester"],
        confidence_score=1.0,
    )
    product = make_product("P_COTTON", materials=["Cotton"], similarity_score=0.8)

    results = reranker.rerank([product], persona, explain=True)

    assert results[0].breakdown is not None
    assert results[0].breakdown.persona_penalty == 0.0


@pytest.mark.unit
def test_disliked_product_penalty() -> None:
    reranker = ProductReranker()
    persona = PersonaSnapshot(
        disliked_products=["P001"],
        confidence_score=1.0,
    )
    disliked = make_product("P001", similarity_score=0.8)
    normal = make_product("P002", similarity_score=0.8)

    results = reranker.rerank([disliked, normal], persona, explain=True)
    result_map = {r.product.product_id: r for r in results}

    assert result_map["P001"].breakdown is not None
    assert result_map["P001"].breakdown.persona_penalty > 0.0
    assert result_map["P002"].breakdown is not None
    assert result_map["P002"].breakdown.persona_penalty == 0.0


@pytest.mark.unit
def test_no_penalty_with_zero_confidence() -> None:
    reranker = ProductReranker()
    persona = PersonaSnapshot(
        disliked_materials=["Polyester"],
        disliked_products=["P001"],
        confidence_score=0.0,
    )
    product = make_product("P001", materials=["Polyester"], similarity_score=0.8)

    results = reranker.rerank([product], persona, explain=True)

    assert results[0].breakdown is not None
    assert results[0].breakdown.persona_penalty == 0.0
