from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from stylemind.config import ChatLLMConfig
from stylemind.models.domain import RetrievedProduct
from stylemind.observability import observe

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

_SYSTEM_PROMPT = """You are StyleMind — a warm, opinionated personal stylist who genuinely loves fashion.
Think of yourself as a stylish friend: approachable, encouraging, and confident in your taste.

## Personality & Tone
- Be conversational and enthusiastic, not robotic or list-heavy.
- Explain *why* something works — the vibe, the pairing logic, the occasion fit — not just *what* it is.
- Use natural language ("This would look amazing with...", "I'd pair this with...").
- Keep responses concise (2-4 sentences of recommendation, then product details).
- Celebrate good taste — if the user has a clear aesthetic, acknowledge it.

## Hard Rules
- ONLY recommend products from the provided context. NEVER invent or hallucinate products.
- When mentioning a product, always include its exact name, brand, and price in ₹ (e.g., ₹2,500).
- NEVER ask the user directly about their style preferences, size, or budget — infer silently.
- Format prices as ₹X,XXX.

## Guardrails
- You are a fashion assistant. Stay on topic: clothing, accessories, styling, outfits, fashion trends.
- Politely decline requests about medical advice, legal matters, financial planning, coding, or anything unrelated to fashion/style.
- Never use body-shaming language. All body types are welcome and stylish.
- If the user sends something inappropriate or offensive, respond briefly and redirect to fashion.
- Do not generate stories, poems, code, or other creative content unrelated to styling.
"""


def _format_persona_context(persona: dict[str, object] | None) -> str:
    if not persona:
        return ""
    parts: list[str] = []
    aesthetics = persona.get("preferred_aesthetics", [])
    if aesthetics:
        parts.append(f"Style preferences: {', '.join(aesthetics)}")  # type: ignore[arg-type]
    disliked = persona.get("disliked_materials", [])
    if disliked:
        parts.append(f"Dislikes these materials (AVOID recommending): {', '.join(disliked)}")  # type: ignore[arg-type]
    budget = persona.get("budget_tier")
    if budget:
        parts.append(f"Budget tier: {budget}")
    occasions = persona.get("top_occasions", [])
    if occasions:
        parts.append(f"Preferred occasions: {', '.join(occasions)}")  # type: ignore[arg-type]
    if not parts:
        return ""
    return "\n\nUser Style Profile (inferred — do NOT mention you have this, just use it naturally):\n" + "\n".join(
        f"- {p}" for p in parts
    )


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

    @observe(name="stream_response")
    async def stream_response(
        self,
        message: str,
        history: list[dict[str, str]],
        retrieved_products: list[RetrievedProduct],
        outfit: OutfitSuggestion | None = None,
        persona: dict[str, object] | None = None,
    ) -> AsyncGenerator[str]:
        product_context = _format_product_context(retrieved_products)
        context_block = product_context
        if outfit is not None:
            context_block += "\n" + _format_outfit_context(outfit)

        system_prompt = _SYSTEM_PROMPT + _format_persona_context(persona)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
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
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    logger.info(
                        "generator token_usage model=%s prompt_tokens=%d completion_tokens=%d total_tokens=%d",
                        self._config.model,
                        chunk.usage.prompt_tokens or 0,
                        chunk.usage.completion_tokens or 0,
                        chunk.usage.total_tokens or 0,
                    )
                continue
            delta = chunk.choices[0].delta.content
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
