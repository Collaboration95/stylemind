from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stylemind.models.domain import RetrievedProduct
from stylemind.rag.generator import StyleMindGenerator, _format_product_context

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_product(
    product_id: str = "P001",
    name: str = "Classic White Shirt",
    brand: str = "Uniqlo",
    price: int = 2499,
    category: str = "Top",
    aesthetics: list[str] | None = None,
    occasions: list[str] | None = None,
) -> RetrievedProduct:
    return RetrievedProduct(
        product_id=product_id,
        name=name,
        description="A versatile white shirt",
        price=price,
        category=category,
        brand=brand,
        budget_tier="Mid",
        aesthetics=aesthetics or ["Minimalist"],
        occasions=occasions or ["Office", "Casual"],
        colors=["White"],
        seasons=["Year-round"],
        pairs_with=[],
        similarity_score=0.85,
    )


def make_generator() -> StyleMindGenerator:
    from stylemind.config import ChatLLMConfig

    config = ChatLLMConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )
    return StyleMindGenerator(config=config)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_product_context_numbered_list() -> None:
    """Context formatting should produce a numbered list with name, brand, price, aesthetics, occasions."""
    products = [
        make_product("P001", "Classic White Shirt", "Uniqlo", 2499, aesthetics=["Minimalist"], occasions=["Office"]),
        make_product("P002", "Floral Sundress", "Zara", 3999, aesthetics=["Boho"], occasions=["Casual", "Beach"]),
    ]

    context = _format_product_context(products)

    # Check numbered list structure
    assert "1." in context
    assert "2." in context

    # Check product 1 fields
    assert "Classic White Shirt" in context
    assert "Uniqlo" in context
    assert "2,499" in context
    assert "Minimalist" in context
    assert "Office" in context

    # Check product 2 fields
    assert "Floral Sundress" in context
    assert "Zara" in context
    assert "3,999" in context
    assert "Boho" in context
    assert "Beach" in context


@pytest.mark.unit
def test_format_product_context_empty_list() -> None:
    """Empty product list should return a 'no products' message."""
    context = _format_product_context([])
    assert "No products" in context


@pytest.mark.unit
def test_detect_product_interest_catches_interest_phrases() -> None:
    """detect_product_interest should return product_id when user expresses interest in a named product."""
    generator = make_generator()
    products = [
        make_product("P001", "Classic White Shirt"),
        make_product("P002", "Floral Sundress"),
    ]

    # Various interest phrases paired with a product name
    test_cases = [
        ("I love the Classic White Shirt", "P001"),
        ("I like Floral Sundress, can you show me more?", "P002"),
        ("I'm interested in Classic White Shirt", "P001"),
        ("I want the Floral Sundress", "P002"),
        ("Tell me about Classic White Shirt", "P001"),
        ("Show me more like Floral Sundress", "P002"),
        ("I want something similar to Classic White Shirt", "P001"),
    ]

    for message, expected_id in test_cases:
        result = generator.detect_product_interest(message, products)
        assert result == expected_id, f"Expected {expected_id!r} for message {message!r}, got {result!r}"


@pytest.mark.unit
def test_detect_product_interest_returns_none_for_generic_messages() -> None:
    """detect_product_interest should return None for generic messages without product references."""
    generator = make_generator()
    products = [
        make_product("P001", "Classic White Shirt"),
        make_product("P002", "Floral Sundress"),
    ]

    generic_messages = [
        "What's trending this season?",
        "Show me something for a beach vacation",
        "I need an outfit for a wedding",
        "What do you recommend?",
        "Hello!",
    ]

    for message in generic_messages:
        result = generator.detect_product_interest(message, products)
        assert result is None, f"Expected None for message {message!r}, got {result!r}"


@pytest.mark.unit
def test_detect_product_interest_returns_none_when_no_products() -> None:
    """detect_product_interest should return None when the product list is empty."""
    generator = make_generator()
    result = generator.detect_product_interest("I love the Classic White Shirt", [])
    assert result is None


# ---------------------------------------------------------------------------
# Integration tests — mock LLM client
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_stream_response_yields_chunks() -> None:
    """stream_response should yield text chunks from the LLM stream."""
    from stylemind.config import ChatLLMConfig

    config = ChatLLMConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    products = [
        make_product("P001", "Classic White Shirt", "Uniqlo", 2499),
    ]

    # Build mock chunks mimicking the OpenAI streaming response
    def make_chunk(content: str) -> MagicMock:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        return chunk

    expected_chunks = ["Here ", "is ", "a ", "recommendation."]
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = AsyncMock(return_value=iter([make_chunk(c) for c in expected_chunks]))

    # Make the async context manager return our mock stream
    async def fake_aiter(self: object) -> AsyncIterator[MagicMock]:
        for chunk in [make_chunk(c) for c in expected_chunks]:
            yield chunk

    mock_stream.__aiter__ = fake_aiter

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream)

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    with patch("stylemind.rag.generator.AsyncOpenAI") as mock_openai_cls:
        mock_client_instance = MagicMock()
        mock_client_instance.chat = mock_chat
        mock_openai_cls.return_value = mock_client_instance

        generator = StyleMindGenerator(config=config)

        collected: list[str] = []
        async for chunk in generator.stream_response(
            message="What shirts do you recommend?",
            history=[],
            retrieved_products=products,
            outfit=None,
        ):
            collected.append(chunk)

    assert collected == expected_chunks, f"Expected {expected_chunks!r}, got {collected!r}"
    assert len(collected) == 4
