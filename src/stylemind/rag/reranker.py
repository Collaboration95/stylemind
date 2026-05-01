from __future__ import annotations

import logging
from dataclasses import dataclass

from stylemind.models.domain import RetrievedProduct
from stylemind.models.schemas import PersonaSnapshot
from stylemind.observability import observe

logger = logging.getLogger(__name__)

# Boost added per matching aesthetic between product and persona (multiplicative with confidence).
# Capped at _PERSONA_BOOST_CAP so a single highly-matched product can't dominate infinitely.
_PERSONA_BOOST_PER_AESTHETIC = 0.1
_PERSONA_BOOST_CAP = 0.3  # max total boost from aesthetic matching (~3 aesthetic matches)

# Flat boost when a product's budget_tier matches the inferred persona budget tier.
_BUDGET_BOOST = 0.05

# Penalty applied when a product contains a disliked material or is in the disliked_products list.
# Applied per violation type (material + product_id can each contribute independently).
_PERSONA_PENALTY = 0.15


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-product score decomposition for explain-mode responses."""

    product_id: str
    base_score: float
    persona_boost: float
    persona_penalty: float
    budget_boost: float
    final_score: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "product_id": self.product_id,
            "base_score": self.base_score,
            "persona_boost": self.persona_boost,
            "penalty": self.persona_penalty,
            "budget_boost": self.budget_boost,
            "final_score": self.final_score,
        }


@dataclass(frozen=True)
class RerankResult:
    """Product with its final persona-adjusted score and optional score breakdown."""

    product: RetrievedProduct
    final_score: float
    breakdown: ScoreBreakdown | None = None


class ProductReranker:
    """Persona-aware reranker that adjusts vector similarity scores.

    Scoring formula:
        final_score = base_score + persona_boost - persona_penalty + budget_boost

    When persona.confidence_score == 0.0, no boost/penalty is applied and the
    ranking is identical to pure vector similarity ordering.
    """

    @observe(name="rerank")
    def rerank(
        self,
        candidates: list[RetrievedProduct],
        persona: PersonaSnapshot,
        explain: bool = False,
    ) -> list[RerankResult]:
        """Rerank candidates using persona signals.

        Args:
            candidates: Products returned by the vector retriever.
            persona: Current user persona snapshot.
            explain: If True, populate ScoreBreakdown in each RerankResult.

        Returns:
            list[RerankResult] sorted by final_score descending.
        """
        confidence = persona.confidence_score

        # Hard-filter disliked materials before scoring to avoid wasted computation
        if confidence > 0.0 and persona.disliked_materials:
            disliked_lower = {m.lower() for m in persona.disliked_materials}
            before_count = len(candidates)
            candidates = [c for c in candidates if not ({m.lower() for m in c.materials} & disliked_lower)]
            filtered = before_count - len(candidates)
            if filtered:
                logger.info("reranker filtered_disliked_materials=%d", filtered)

        results: list[RerankResult] = []

        for candidate in candidates:
            base_score = candidate.similarity_score

            # --- persona boost ---
            persona_boost = 0.0
            if confidence > 0.0 and persona.preferred_aesthetics:
                matched = set(candidate.aesthetics) & set(persona.preferred_aesthetics)
                raw_boost = len(matched) * _PERSONA_BOOST_PER_AESTHETIC * confidence
                persona_boost = min(raw_boost, _PERSONA_BOOST_CAP)

            # --- persona penalty (disliked products only; materials already filtered) ---
            persona_penalty = 0.0
            if confidence > 0.0 and persona.disliked_products and candidate.product_id in persona.disliked_products:
                persona_penalty = _PERSONA_PENALTY * confidence

            # --- budget boost ---
            budget_boost = 0.0
            if (
                confidence > 0.0
                and persona.budget_tier
                and candidate.budget_tier
                and candidate.budget_tier.lower() == persona.budget_tier.lower()
            ):
                budget_boost = _BUDGET_BOOST * confidence

            final_score = base_score + persona_boost - persona_penalty + budget_boost

            breakdown: ScoreBreakdown | None = None
            if explain:
                breakdown = ScoreBreakdown(
                    product_id=candidate.product_id,
                    base_score=base_score,
                    persona_boost=persona_boost,
                    persona_penalty=persona_penalty,
                    budget_boost=budget_boost,
                    final_score=final_score,
                )

            results.append(RerankResult(product=candidate, final_score=final_score, breakdown=breakdown))

            logger.debug(
                "reranker product_id=%s base=%.3f boost=%.3f penalty=%.3f budget=%.3f final=%.3f",
                candidate.product_id,
                base_score,
                persona_boost,
                persona_penalty,
                budget_boost,
                final_score,
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        logger.info("reranker reranked candidate_count=%d confidence=%.2f", len(results), confidence)
        return results
