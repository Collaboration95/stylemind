from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from stylemind.config import ChatLLMConfig
from stylemind.models.domain import RetrievedProduct

if TYPE_CHECKING:
    from stylemind.models.schemas import OutfitSuggestion

logger = logging.getLogger(__name__)

_INTEREST_PHRASES = (
    "like",
    "love",
    "interested in",
    "want",
    "show me more",
    "tell me about",
    "that one",
    "this one",
    "similar to",
)

_SYSTEM_PROMPT = """You are StyleMind, a friendly and knowledgeable personal stylist assistant.

Guidelines:
- Only recommend products from the provided context. NEVER invent or hallucinate products.
- When mentioning a product, always include its exact name, brand, and price (in INR).
- NEVER ask the user directly about their style preferences, size, or budget — infer from context.
- When suggesting outfit pairings, explain why each item complements the others (occasion, aesthetic, season).
- Be honest and helpful when no products match the user's request.
- Keep responses concise, warm, and actionable.
- Format prices as ₹X,XXX (e.g., ₹2,500).
"""


def _format_product_context(products: list[RetrievedProduct]) -> str:
    """Format retrieved products as a numbered list for the LLM context."""
    if not products:
        return "No products available in the current context."

    lines: list[str] = ["Available products:"]
    for i, product in enumerate(products, start=1):
        aesthetics = ", ".join(product.aesthetics) if product.aesthetics else "N/A"
        occasions = ", ".join(product.occasions) if product.occasions else "N/A"
        lines.append(
            f"{i}. {product.name} | Brand: {product.brand} | Price: ₹{product.price_inr:,} "
            f"| Category: {product.category} | Aesthetics: {aesthetics} | Occasions: {occasions}"
        )
    return "\n".join(lines)


def _format_outfit_context(outfit: OutfitSuggestion) -> str:
    """Format an outfit suggestion for the LLM context."""
    lines: list[str] = [
        f"\nSuggested outfit for {outfit.occasion} ({outfit.season}):",
        f"  Anchor: {outfit.anchor.name} by {outfit.anchor.brand} — ₹{outfit.anchor.price_inr:,}",
    ]
    for item in outfit.items:
        lines.append(f"  + {item.name} by {item.brand} — ₹{item.price_inr:,} ({item.category}): {item.justification}")
    return "\n".join(lines)


class StyleMindGenerator:
    """LLM response generator with streaming support and citation grounding."""

    def __init__(self, config: ChatLLMConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    async def stream_response(
        self,
        message: str,
        history: list[dict[str, str]],
        retrieved_products: list[RetrievedProduct],
        outfit: OutfitSuggestion | None = None,
    ) -> AsyncGenerator[str]:
        """Stream an LLM response grounded in the retrieved products.

        Args:
            message: Current user message.
            history: Prior conversation turns as list of {"role": ..., "content": ...} dicts.
            retrieved_products: Products from the RAG retriever/reranker.
            outfit: Optional outfit suggestion from the outfit builder.

        Yields:
            Text chunks as they arrive from the LLM stream.
        """
        product_context = _format_product_context(retrieved_products)
        context_block = product_context
        if outfit is not None:
            context_block += "\n" + _format_outfit_context(outfit)

        # Build message list: system + history + context-injected user message
        messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": f"{context_block}\n\nUser: {message}",
            }
        )

        logger.info(
            "generator stream_response product_count=%d has_outfit=%s model=%s",
            len(retrieved_products),
            outfit is not None,
            self._config.model,
        )

        stream = await self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._config.temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def detect_product_interest(
        self,
        message: str,
        products: list[RetrievedProduct],
    ) -> str | None:
        """Detect whether the user is expressing interest in a specific product.

        Uses simple keyword matching against product names and IDs — no LLM call.

        Args:
            message: Raw user message text.
            products: Products currently in context.

        Returns:
            product_id of the matched product, or None if no interest detected.
        """
        if not products:
            return None

        message_lower = message.lower()

        # Check if message contains any interest phrase
        has_interest_phrase = any(phrase in message_lower for phrase in _INTEREST_PHRASES)
        if not has_interest_phrase:
            return None

        # Match against product names (longest match first to avoid partial collisions)
        sorted_products = sorted(products, key=lambda p: len(p.name), reverse=True)
        for product in sorted_products:
            if product.name.lower() in message_lower:
                logger.debug(
                    "generator detect_product_interest matched product_id=%s name=%s",
                    product.product_id,
                    product.name,
                )
                return product.product_id

        # Fall back to product_id mention
        for product in products:
            if product.product_id.lower() in message_lower:
                logger.debug(
                    "generator detect_product_interest matched product_id=%s via id",
                    product.product_id,
                )
                return product.product_id

        return None
