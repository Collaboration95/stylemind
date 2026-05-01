from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stylemind.models.domain import RetrievedProduct
from stylemind.models.schemas import PersonaSnapshot
from stylemind.rag.reranker import RerankResult, ScoreBreakdown

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_app_with_mocks(
    neo4j_ok: bool = True,
    embedder_ok: bool = True,
    persona: PersonaSnapshot | None = None,
    stream_chunks: list[str] | None = None,
) -> FastAPI:
    """Build a FastAPI test app with all dependencies pre-populated on app.state."""
    from fastapi import FastAPI

    test_app = FastAPI()

    # Register routers directly (no lifespan needed)
    from stylemind.api import chat as chat_module
    from stylemind.api import health as health_module
    from stylemind.api import persona as persona_module

    test_app.include_router(chat_module.router)
    test_app.include_router(persona_module.router)
    test_app.include_router(health_module.router)

    # --- Neo4j mock ---
    if neo4j_ok:
        mock_neo4j = MagicMock()
        mock_neo4j.verify_connectivity = MagicMock(return_value=True)
        test_app.state.neo4j = mock_neo4j
    # If neo4j_ok is False, don't set state.neo4j so health check sees None

    # --- Embedder mock ---
    if embedder_ok:
        mock_embedder = MagicMock()
        mock_embedder.embed_query = MagicMock(return_value=[0.0] * 384)
        test_app.state.embedder = mock_embedder
    # If embedder_ok is False, don't set state.embedder

    # --- Persona manager mock ---
    mock_persona_manager = MagicMock()
    resolved_persona = persona if persona is not None else PersonaSnapshot()
    mock_persona_manager.get_persona = MagicMock(return_value=resolved_persona)
    mock_persona_manager.update_persona = MagicMock()
    test_app.state.persona_manager = mock_persona_manager

    # --- PersonaInferenceEngine mock ---
    mock_inference = MagicMock()
    from stylemind.models.domain import PersonaSignals

    mock_inference.extract_signals = MagicMock(return_value=PersonaSignals())
    test_app.state.inference_engine = mock_inference

    # --- Retriever mock ---
    mock_retriever = MagicMock()
    mock_retriever.retrieve = MagicMock(return_value=[])
    test_app.state.retriever = mock_retriever

    # --- Reranker mock ---
    mock_reranker = MagicMock()
    mock_reranker.rerank = MagicMock(return_value=[])
    test_app.state.reranker = mock_reranker

    # --- Generator mock ---
    chunks = stream_chunks if stream_chunks is not None else ["Hello", " world"]

    async def _fake_stream(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    mock_generator = MagicMock()
    mock_generator.stream_response = MagicMock(side_effect=_fake_stream)
    mock_generator.detect_product_interest = MagicMock(return_value=None)
    test_app.state.generator = mock_generator

    # --- OutfitBuilder mock ---
    mock_outfit = MagicMock()
    mock_outfit.build_outfit = MagicMock(return_value=None)
    test_app.state.outfit_builder = mock_outfit

    return test_app


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_sse_stream():
    """POST /chat returns SSE stream with data: chunks and a final [DONE]."""
    app = _make_app_with_mocks(stream_chunks=["Style", " tip"])
    client = TestClient(app, raise_server_exceptions=True)

    payload = {"user_id": "user-1", "message": "What should I wear?"}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Collect all SSE lines
    sse_lines = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert len(sse_lines) >= 1

    # Final chunk must be [DONE]
    assert sse_lines[-1] == "data: [DONE]"

    # Content chunks should include our fake stream output
    content_chunks = [line[len("data: ") :] for line in sse_lines if line != "data: [DONE]"]
    assert "Style" in content_chunks or "Style" in "".join(content_chunks)


@pytest.mark.integration
def test_persona_endpoint_returns_snapshot():
    """GET /persona/{user_id} returns a valid PersonaSnapshot JSON."""
    snapshot = PersonaSnapshot(
        preferred_aesthetics=["Quiet Luxury", "Old Money"],
        disliked_materials=["Polyester"],
        budget_tier="premium",
        top_occasions=["Office"],
        confidence_score=0.75,
    )
    app = _make_app_with_mocks(persona=snapshot)
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/persona/user-1")

    assert response.status_code == 200
    data = response.json()
    assert data["preferred_aesthetics"] == ["Quiet Luxury", "Old Money"]
    assert data["disliked_materials"] == ["Polyester"]
    assert data["budget_tier"] == "premium"
    assert data["top_occasions"] == ["Office"]
    assert data["confidence_score"] == pytest.approx(0.75)


@pytest.mark.integration
def test_health_returns_200_when_healthy():
    """GET /health returns 200 when Neo4j is connected and embedder is loaded."""
    app = _make_app_with_mocks(neo4j_ok=True, embedder_ok=True)
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["neo4j"] is True
    assert data["embedder"] is True


@pytest.mark.integration
def test_health_returns_503_when_neo4j_unavailable():
    """GET /health returns 503 when Neo4j is not available."""
    app = _make_app_with_mocks(neo4j_ok=False, embedder_ok=True)
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["neo4j"] is False


# ---------------------------------------------------------------------------
# Contract / schema tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_persona_snapshot_validates_against_spec():
    """PersonaSnapshot validates all required spec fields with correct types."""
    snapshot = PersonaSnapshot(
        preferred_aesthetics=["Streetwear"],
        disliked_materials=["Nylon"],
        budget_tier="mid",
        top_occasions=["Casual", "Active"],
        confidence_score=0.5,
    )

    assert isinstance(snapshot.preferred_aesthetics, list)
    assert isinstance(snapshot.disliked_materials, list)
    assert isinstance(snapshot.budget_tier, str)
    assert isinstance(snapshot.top_occasions, list)
    assert isinstance(snapshot.confidence_score, float)

    # Default empty state must also be valid
    empty = PersonaSnapshot()
    assert empty.preferred_aesthetics == []
    assert empty.disliked_materials == []
    assert empty.budget_tier is None
    assert empty.top_occasions == []
    assert empty.confidence_score == 0.0


@pytest.mark.unit
def test_persona_endpoint_returns_default_when_no_data():
    """GET /persona/{user_id} returns empty PersonaSnapshot for unknown user."""
    app = _make_app_with_mocks(persona=PersonaSnapshot())
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/persona/unknown-user-xyz")

    assert response.status_code == 200
    data = response.json()
    assert data["preferred_aesthetics"] == []
    assert data["confidence_score"] == 0.0
    assert data["budget_tier"] is None


# ---------------------------------------------------------------------------
# SSE __JSON__ events
# ---------------------------------------------------------------------------


def _make_products():
    return [
        RetrievedProduct(
            product_id="P001",
            name="Linen Trouser",
            description="test",
            price_inr=4200,
            category="Bottoms",
            brand="COS",
            budget_tier="Mid",
            aesthetics=["Quiet Luxury"],
            occasions=["Office"],
            colors=["Neutrals"],
            seasons=["SS"],
            pairs_with=[],
            similarity_score=0.85,
        ),
        RetrievedProduct(
            product_id="P005",
            name="Ribbed Polo",
            description="test",
            price_inr=1800,
            category="Tops",
            brand="Uniqlo",
            budget_tier="Budget",
            aesthetics=["Streetwear"],
            occasions=["Casual"],
            colors=["Neutrals"],
            seasons=["Year-round"],
            pairs_with=[],
            similarity_score=0.78,
        ),
    ]


def _make_rerank_results(products, explain=False):
    results = []
    for p in products:
        breakdown = None
        if explain:
            breakdown = ScoreBreakdown(
                product_id=p.product_id,
                base_score=p.similarity_score,
                persona_boost=0.04,
                persona_penalty=0.0,
                budget_boost=0.0,
                final_score=p.similarity_score + 0.04,
            )
        results.append(
            RerankResult(product=p, final_score=p.similarity_score + (0.04 if explain else 0.0), breakdown=breakdown)
        )
    return results


def _parse_sse_data(response):
    payloads = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            payloads.append(line[6:])
    return payloads


@pytest.mark.integration
def test_chat_emits_sources_json_event():
    products = _make_products()
    rerank_results = _make_rerank_results(products, explain=False)

    app = _make_app_with_mocks(stream_chunks=["Hello"])

    mock_retriever = app.state.retriever
    mock_retriever.retrieve = MagicMock(return_value=products)

    mock_reranker = app.state.reranker
    mock_reranker.rerank = MagicMock(return_value=rerank_results)

    client = TestClient(app, raise_server_exceptions=True)
    response = client.post("/chat", json={"user_id": "u1", "message": "test"})

    assert response.status_code == 200
    payloads = _parse_sse_data(response)

    sources_events = [p for p in payloads if p.startswith("__JSON__") and '"sources"' in p]
    assert len(sources_events) == 1

    import json

    data = json.loads(sources_events[0][8:])
    assert len(data["sources"]) == 2
    assert data["sources"][0]["product_id"] == "P001"
    assert data["sources"][0]["name"] == "Linen Trouser"
    assert data["sources"][0]["brand"] == "COS"
    assert data["sources"][0]["price_inr"] == 4200


@pytest.mark.integration
def test_chat_emits_explain_json_event_when_explain_true():
    products = _make_products()
    rerank_results = _make_rerank_results(products, explain=True)

    app = _make_app_with_mocks(stream_chunks=["Hello"])

    mock_retriever = app.state.retriever
    mock_retriever.retrieve = MagicMock(return_value=products)

    mock_reranker = app.state.reranker
    mock_reranker.rerank = MagicMock(return_value=rerank_results)

    client = TestClient(app, raise_server_exceptions=True)
    response = client.post("/chat", json={"user_id": "u1", "message": "test", "explain": True})

    assert response.status_code == 200
    payloads = _parse_sse_data(response)

    import json

    explain_events = [p for p in payloads if p.startswith("__JSON__") and '"explain"' in p]
    assert len(explain_events) == 1

    data = json.loads(explain_events[0][8:])
    assert len(data["explain"]) == 2
    assert data["explain"][0]["product_id"] == "P001"
    assert data["explain"][0]["base_score"] == pytest.approx(0.85)
    assert data["explain"][0]["persona_boost"] == pytest.approx(0.04)
    assert data["explain"][0]["penalty"] == pytest.approx(0.0)
    assert data["explain"][0]["final_score"] == pytest.approx(0.89)


@pytest.mark.integration
def test_chat_no_explain_event_when_explain_false():
    products = _make_products()
    rerank_results = _make_rerank_results(products, explain=False)

    app = _make_app_with_mocks(stream_chunks=["Hello"])

    mock_retriever = app.state.retriever
    mock_retriever.retrieve = MagicMock(return_value=products)

    mock_reranker = app.state.reranker
    mock_reranker.rerank = MagicMock(return_value=rerank_results)

    client = TestClient(app, raise_server_exceptions=True)
    response = client.post("/chat", json={"user_id": "u1", "message": "test", "explain": False})

    assert response.status_code == 200
    payloads = _parse_sse_data(response)

    explain_events = [p for p in payloads if p.startswith("__JSON__") and '"explain"' in p]
    assert len(explain_events) == 0


@pytest.mark.unit
def test_score_breakdown_to_dict():
    sb = ScoreBreakdown(
        product_id="P001", base_score=0.85, persona_boost=0.04, persona_penalty=0.0, budget_boost=0.02, final_score=0.91
    )
    d = sb.to_dict()
    assert d["product_id"] == "P001"
    assert d["base_score"] == pytest.approx(0.85)
    assert d["persona_boost"] == pytest.approx(0.04)
    assert d["penalty"] == pytest.approx(0.0)
    assert d["budget_boost"] == pytest.approx(0.02)
    assert d["final_score"] == pytest.approx(0.91)


@pytest.mark.unit
def test_health_returns_503_when_embedder_down():
    """GET /health returns 503 when embedder is not loaded."""
    app = _make_app_with_mocks(neo4j_ok=True, embedder_ok=False)
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["embedder"] is False


@pytest.mark.unit
def test_chat_when_generator_is_none():
    """POST /chat when generator is None returns graceful error message."""
    app = _make_app_with_mocks(stream_chunks=["Hello"])

    app.state.generator = None

    client = TestClient(app, raise_server_exceptions=True)
    response = client.post("/chat", json={"user_id": "u1", "message": "hello"})

    assert response.status_code == 200
    assert "not available" in response.text.lower() or "StyleMind generator not available" in response.text


# ---------------------------------------------------------------------------
# API error path tests (#75)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chat_stream_error_mid_response():
    """LLM stream raises mid-response → error event sent and stream ends with [DONE]."""

    async def _failing_stream(*args, **kwargs):
        yield "Start"
        raise RuntimeError("LLM connection lost")

    app = _make_app_with_mocks()
    app.state.generator.stream_response = MagicMock(side_effect=_failing_stream)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/chat", json={"user_id": "u1", "message": "hello"})

    assert response.status_code == 200
    payloads = _parse_sse_data(response)
    assert payloads[-1] == "[DONE]"
    assert any("error" in p.lower() for p in payloads)


@pytest.mark.unit
def test_chat_retriever_failure_still_responds():
    """Retriever throws → chat still returns a response and ends with [DONE]."""
    app = _make_app_with_mocks(stream_chunks=["Fallback response"])
    app.state.retriever.retrieve = MagicMock(side_effect=RuntimeError("Neo4j unavailable"))

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/chat", json={"user_id": "u1", "message": "what to wear?"})

    assert response.status_code == 200
    payloads = _parse_sse_data(response)
    assert payloads[-1] == "[DONE]"
    assert any("Fallback response" in p for p in payloads)


@pytest.mark.unit
def test_chat_persona_inference_failure_still_responds():
    """Persona inference failure → chat still succeeds, no persona signals emitted."""
    app = _make_app_with_mocks(stream_chunks=["Hello"])
    app.state.inference_engine.extract_signals = MagicMock(side_effect=RuntimeError("Extraction failed"))

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/chat", json={"user_id": "u1", "message": "show me dresses"})

    assert response.status_code == 200
    payloads = _parse_sse_data(response)
    assert payloads[-1] == "[DONE]"
    signals_events = [p for p in payloads if "__JSON__" in p and "signals" in p]
    assert len(signals_events) == 0


@pytest.mark.unit
def test_chat_empty_retrieval_still_streams():
    """Retriever returns empty results → response still streams and completes."""
    app = _make_app_with_mocks(stream_chunks=["No products found, but here's advice"])
    app.state.retriever.retrieve = MagicMock(return_value=[])

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/chat", json={"user_id": "u1", "message": "hello"})

    assert response.status_code == 200
    payloads = _parse_sse_data(response)
    assert payloads[-1] == "[DONE]"
    sources_events = [p for p in payloads if "__JSON__" in p and "sources" in p]
    assert len(sources_events) == 0
