from __future__ import annotations

import json
import time

import pytest

from stylemind.models.domain import PersonaSignals, RetrievedProduct
from stylemind.models.schemas import PersonaSnapshot
from stylemind.rag.reranker import ProductReranker

# ---------------------------------------------------------------------------
# Helper fixtures / factories (module-level to avoid repeat construction)
# ---------------------------------------------------------------------------


def _make_10_products() -> list[RetrievedProduct]:
    """Create 10 realistic RetrievedProduct instances for benchmark tests."""
    aesthetics_pool = [
        ["Quiet Luxury"],
        ["Old Money", "Quiet Luxury"],
        ["Streetwear"],
        ["Athleisure"],
        ["Cottagecore"],
        ["Corporate Minimalism", "Old Money"],
        ["Y2K", "Streetwear"],
        ["Coastal Grandma"],
        ["Casual Minimalism"],
        ["Quiet Luxury", "Old Money"],
    ]
    return [
        RetrievedProduct(
            product_id=f"P{i:03d}",
            name=f"Test Product {i}",
            description=f"A test product description number {i} with some detail",
            price_inr=1000 + i * 500,
            category="Tops" if i % 2 == 0 else "Bottoms",
            brand="COS" if i % 3 == 0 else "Arket",
            budget_tier="Mid",
            aesthetics=aesthetics_pool[i % len(aesthetics_pool)],
            occasions=["Office", "Casual"],
            colors=["Classic Neutrals"],
            seasons=["Year-round"],
            pairs_with=[f"P{(i + 1):03d}"],
            similarity_score=0.9 - i * 0.05,
        )
        for i in range(10)
    ]


def _make_persona_snapshot() -> PersonaSnapshot:
    return PersonaSnapshot(
        preferred_aesthetics=["Quiet Luxury", "Old Money"],
        disliked_materials=["Polyester"],
        budget_tier="Mid",
        top_occasions=["Office"],
        confidence_score=0.6,
    )


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_embedding_latency_under_500ms() -> None:
    """Real LocalEmbedder embed_query latency must be < 500ms after warm-up.

    The model loads on first use (cold start is excluded by doing a warm-up call).
    The benchmark measures the second call.
    """
    from stylemind.rag.embedder import LocalEmbedder

    embedder = LocalEmbedder()

    # Warm up — load the model; don't time this
    embedder.embed_query("warm up query")

    # Now benchmark the actual call
    start = time.perf_counter()
    result = embedder.embed_query("show me a minimalist linen dress for the office")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert isinstance(result, list)
    assert len(result) == 384, f"Expected 384-dim embedding, got {len(result)}"
    assert elapsed_ms < 500, f"embed_query took {elapsed_ms:.1f}ms, expected < 500ms"


@pytest.mark.performance
def test_reranker_latency_under_200ms() -> None:
    """ProductReranker.rerank with 10 products and a persona must complete in < 200ms."""
    reranker = ProductReranker(persona_weight=0.3)
    products = _make_10_products()
    persona = _make_persona_snapshot()

    # Warm up
    reranker.rerank(products[:2], persona)

    start = time.perf_counter()
    results = reranker.rerank(products, persona)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(results) == 10
    assert elapsed_ms < 200, f"rerank took {elapsed_ms:.1f}ms, expected < 200ms"


@pytest.mark.performance
def test_generator_format_context_under_10ms() -> None:
    """_format_product_context for 10 products must complete in < 10ms (no LLM call)."""
    from stylemind.rag.generator import _format_product_context

    products = _make_10_products()

    # Warm up
    _format_product_context(products[:2])

    start = time.perf_counter()
    context = _format_product_context(products)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(context) > 0, "Context should not be empty"
    assert "Available products:" in context
    assert elapsed_ms < 10, f"_format_product_context took {elapsed_ms:.2f}ms, expected < 10ms"


@pytest.mark.performance
def test_persona_signals_parsing_under_50ms() -> None:
    """Parsing PersonaSignals from a JSON string must complete in < 50ms."""
    sample_json = json.dumps(
        {
            "liked_aesthetics": ["Quiet Luxury", "Old Money"],
            "disliked_materials": ["Polyester", "Acrylic"],
            "mentioned_occasions": ["Office", "Date Night"],
            "budget_signal": "premium",
            "color_preferences": ["Earthy Neutrals", "Classic Neutrals"],
            "brand_mentions": ["COS", "Arket", "Toteme"],
            "sentiment_on_shown": {"P001": "positive", "P003": "negative"},
            "signal_strength": 0.85,
        }
    )

    # Warm up
    PersonaSignals(**json.loads(sample_json))

    start = time.perf_counter()
    data = json.loads(sample_json)
    signals = PersonaSignals(
        liked_aesthetics=data.get("liked_aesthetics", []),
        disliked_materials=data.get("disliked_materials", []),
        mentioned_occasions=data.get("mentioned_occasions", []),
        budget_signal=data.get("budget_signal"),
        color_preferences=data.get("color_preferences", []),
        brand_mentions=data.get("brand_mentions", []),
        sentiment_on_shown=data.get("sentiment_on_shown", {}),
        signal_strength=float(data.get("signal_strength", 0.5)),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert isinstance(signals, PersonaSignals)
    assert signals.liked_aesthetics == ["Quiet Luxury", "Old Money"]
    assert signals.signal_strength == pytest.approx(0.85)
    assert elapsed_ms < 50, f"PersonaSignals parsing took {elapsed_ms:.2f}ms, expected < 50ms"
