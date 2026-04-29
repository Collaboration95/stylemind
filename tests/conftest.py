from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_config() -> Generator[None]:
    from stylemind import config as cfg_module

    cfg_module._reset_config()
    yield
    cfg_module._reset_config()


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
    monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
    monkeypatch.setenv("CHAT_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("EXTRACTION_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CHAT_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-4.1-nano")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-lf-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-lf-secret")


@pytest.fixture
def mock_driver() -> MagicMock:
    driver = MagicMock()
    driver.verify_connectivity = MagicMock()
    driver.execute_query = MagicMock(return_value=MagicMock(records=[]))
    driver.close = MagicMock()
    return driver


@pytest.fixture
def mock_async_driver() -> AsyncMock:
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock(return_value=AsyncMock(records=[]))
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def sample_retrieved_products():
    """Returns a list of 5 RetrievedProduct instances with realistic data."""
    from stylemind.models.domain import RetrievedProduct

    return [
        RetrievedProduct(
            product_id="P001",
            name="Linen Wide-Leg Trouser",
            description="Relaxed wide-leg trouser in breathable linen with a high waist and tailored pleat",
            price_inr=4200,
            category="Bottoms",
            brand="COS",
            budget_tier="Mid",
            aesthetics=["Corporate Minimalism"],
            occasions=["Office"],
            colors=["Earthy Neutrals"],
            seasons=["SS", "Year-round"],
            pairs_with=["P012", "P018", "P031"],
            similarity_score=0.85,
        ),
        RetrievedProduct(
            product_id="P005",
            name="Ribbed Polo Shirt",
            description="Fine ribbed polo in Pima cotton with a subtle tipped collar",
            price_inr=1800,
            category="Tops",
            brand="Uniqlo",
            budget_tier="Budget",
            aesthetics=["Quiet Luxury"],
            occasions=["Casual", "Office"],
            colors=["Classic Neutrals"],
            seasons=["Year-round"],
            pairs_with=["P001", "P011", "P013"],
            similarity_score=0.78,
        ),
        RetrievedProduct(
            product_id="P012",
            name="Silk Slip Cami",
            description="Bias-cut slip cami in washed silk with adjustable spaghetti straps and a deep V",
            price_inr=3800,
            category="Tops",
            brand="COS",
            budget_tier="Mid",
            aesthetics=["Quiet Luxury"],
            occasions=["Date Night", "Casual"],
            colors=["Soft Whites", "Classic Neutrals"],
            seasons=["SS", "AW"],
            pairs_with=["P001", "P002", "P015", "P019"],
            similarity_score=0.72,
        ),
        RetrievedProduct(
            product_id="P019",
            name="Tailored Blazer (Fitted)",
            description="Double-breasted fitted blazer in pressed wool crepe — structured and minimal",
            price_inr=22000,
            category="Outerwear",
            brand="Toteme",
            budget_tier="Luxury",
            aesthetics=["Old Money", "Quiet Luxury"],
            occasions=["Office", "Formal"],
            colors=["Classic Neutrals"],
            seasons=["AW"],
            pairs_with=["P014", "P015", "P012", "P008"],
            similarity_score=0.68,
        ),
        RetrievedProduct(
            product_id="P027",
            name="Pleated Midi Skirt",
            description="Knife-pleated midi skirt in fluid silk blend — movement with restraint",
            price_inr=4800,
            category="Bottoms",
            brand="COS",
            budget_tier="Mid",
            aesthetics=["Quiet Luxury", "Old Money"],
            occasions=["Office", "Date Night"],
            colors=["Classic Neutrals", "Soft Whites"],
            seasons=["SS", "AW"],
            pairs_with=["P005", "P012", "P013", "P018"],
            similarity_score=0.65,
        ),
    ]


@pytest.fixture
def sample_persona_snapshot():
    """Returns a PersonaSnapshot with preferred_aesthetics=['Quiet Luxury'], confidence=0.4."""
    from stylemind.models.schemas import PersonaSnapshot

    return PersonaSnapshot(
        preferred_aesthetics=["Quiet Luxury"],
        disliked_materials=[],
        budget_tier=None,
        top_occasions=[],
        confidence_score=0.4,
    )


@pytest.fixture
def sample_persona_signals():
    """Returns PersonaSignals with liked_aesthetics=['Quiet Luxury'], signal_strength=0.7."""
    from stylemind.models.domain import PersonaSignals

    return PersonaSignals(
        liked_aesthetics=["Quiet Luxury"],
        disliked_materials=[],
        mentioned_occasions=[],
        budget_signal=None,
        color_preferences=[],
        brand_mentions=[],
        sentiment_on_shown={},
        signal_strength=0.7,
    )


@pytest.fixture
def mock_llm_client():
    """Returns a MagicMock mimicking openai.OpenAI with completions.create."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"liked_aesthetics": [], "disliked_materials": [], "mentioned_occasions": [], '
        '"budget_signal": null, "color_preferences": [], "brand_mentions": [], '
        '"sentiment_on_shown": {}, "signal_strength": 0.1}'
    )
    client.chat.completions.create.return_value = mock_response
    return client
