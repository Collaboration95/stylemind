from __future__ import annotations

import json
import logging

from openai import OpenAI

from stylemind.config import ExtractionLLMConfig
from stylemind.models.domain import PersonaSignals
from stylemind.observability import observe

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a fashion preference extraction system.
Extract structured signals from user messages about their style preferences.

Return JSON with these fields:
- liked_aesthetics: list of aesthetic names from [Quiet Luxury, Old Money, Streetwear, Y2K, Athleisure, Cottagecore, Corporate Minimalism, Casual Minimalism, Coastal Grandma]
- disliked_materials: list of materials explicitly disliked
- mentioned_occasions: list of occasions from [Office, Date Night, Wedding Guest, Casual, Active, Travel, Formal, Brunch]
- budget_signal: one of "budget", "mid", "premium", "luxury" or null
- color_preferences: list of colors mentioned
- brand_mentions: list of brand names mentioned
- sentiment_on_shown: dict mapping product_id to "positive" or "negative"
- signal_strength: 0.0-1.0 based on how explicit the signals are

Examples:
User: "I hate polyester, it's so scratchy"
-> {"disliked_materials": ["Polyester"], "signal_strength": 0.9, "liked_aesthetics": [], "mentioned_occasions": [], "budget_signal": null, "color_preferences": [], "brand_mentions": [], "sentiment_on_shown": {}}

User: "something for a date night that's minimal and understated"
-> {"mentioned_occasions": ["Date Night"], "liked_aesthetics": ["Quiet Luxury", "Casual Minimalism"], "signal_strength": 0.7, "disliked_materials": [], "budget_signal": null, "color_preferences": [], "brand_mentions": [], "sentiment_on_shown": {}}

User: "I'm on a tight budget"
-> {"budget_signal": "budget", "signal_strength": 0.6, "liked_aesthetics": [], "disliked_materials": [], "mentioned_occasions": [], "color_preferences": [], "brand_mentions": [], "sentiment_on_shown": {}}
"""


_PERSONA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "liked_aesthetics": {"type": "array", "items": {"type": "string"}},
        "disliked_materials": {"type": "array", "items": {"type": "string"}},
        "mentioned_occasions": {"type": "array", "items": {"type": "string"}},
        "budget_signal": {"type": ["string", "null"]},
        "color_preferences": {"type": "array", "items": {"type": "string"}},
        "brand_mentions": {"type": "array", "items": {"type": "string"}},
        "sentiment_on_shown": {"type": "object", "additionalProperties": {"type": "string"}},
        "signal_strength": {"type": "number"},
    },
    "required": [
        "liked_aesthetics",
        "disliked_materials",
        "mentioned_occasions",
        "budget_signal",
        "color_preferences",
        "brand_mentions",
        "sentiment_on_shown",
        "signal_strength",
    ],
    "additionalProperties": False,
}


class PersonaInferenceEngine:
    def __init__(self, config: ExtractionLLMConfig) -> None:
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._model = config.model

    @observe(name="extract_signals")
    def extract_signals(
        self,
        message: str,
        history: list[dict[str, str]],
        shown_products: list[str],
    ) -> PersonaSignals:
        """Extract persona signals from a user message.

        Args:
            message: The current user message.
            history: Full conversation history as list of {"role": ..., "content": ...} dicts.
            shown_products: Product IDs shown to the user in this turn.

        Returns:
            PersonaSignals with extracted signals, or empty PersonaSignals on any failure.
        """
        try:
            recent_history = history[-6:]

            context_parts: list[str] = []
            if recent_history:
                context_parts.append("Recent conversation context:")
                for turn in recent_history:
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    context_parts.append(f"  {role}: {content}")

            if shown_products:
                context_parts.append(f"Products shown to user this turn: {', '.join(shown_products)}")

            user_content = "\n".join(context_parts)
            if user_content:
                user_content += f"\n\nCurrent user message: {message}"
            else:
                user_content = message

            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "PersonaSignals",
                            "schema": _PERSONA_JSON_SCHEMA,
                            "strict": True,
                        },
                    },
                    temperature=0,
                )
            except Exception:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )

            raw_content = response.choices[0].message.content or "{}"
            data = json.loads(raw_content)
            signals = PersonaSignals(**data)

            if hasattr(response, "usage") and response.usage is not None:
                logger.info(
                    "persona extraction token_usage model=%s prompt_tokens=%d completion_tokens=%d total_tokens=%d",
                    self._model,
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                    response.usage.total_tokens or 0,
                )

            logger.info(
                "persona signals extracted signal_strength=%.2f liked_aesthetics=%s",
                signals.signal_strength,
                signals.liked_aesthetics,
            )
            return signals

        except Exception as exc:
            logger.warning("persona signal extraction failed error=%s", exc)
            return PersonaSignals()
