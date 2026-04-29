from __future__ import annotations

import logging
from dataclasses import dataclass

from stylemind.models.domain import RetrievedProduct
from stylemind.models.schemas import PersonaSnapshot
from stylemind.observability import observe

logger = logging.getLogger(__name__)

_PERSONA_BOOST_PER_AESTHETIC = 0.1
_PERSONA_BOOST_CAP = 0.3
_BUDGET_BOOST = 0.05
_PERSONA_PENALTY = 0.15


@dataclass(frozen=True)
class ScoreBreakdown:
    product_id: str
    base_score: float
    persona_boost: float
    persona_penalty: float
    budget_boost: float
    final_score: float


@dataclass(frozen=True)
class RerankResult:
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

    def __init__(self, persona_weight: float = 0.3) -> None:
        self._persona_weight = persona_weight

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
        results: list[RerankResult] = []

        for candidate in candidates:
            base_score = candidate.similarity_score

            # --- persona boost ---
            persona_boost = 0.0
            if confidence > 0.0 and persona.preferred_aesthetics:
                matched = set(candidate.aesthetics) & set(persona.preferred_aesthetics)
                raw_boost = len(matched) * _PERSONA_BOOST_PER_AESTHETIC * confidence
                persona_boost = min(raw_boost, _PERSONA_BOOST_CAP)

            # --- persona penalty ---
            persona_penalty = 0.0
            if confidence > 0.0 and persona.disliked_materials:
                # Penalise if any candidate aesthetic or category signals a disliked material context.
                # We compare lowercased tokens broadly so "Cotton" matches "cotton blend" etc.
                disliked_lower = {m.lower() for m in persona.disliked_materials}
                candidate_tokens = {candidate.category.lower(), candidate.brand.lower()}
                for aesthetic in candidate.aesthetics:
                    candidate_tokens.add(aesthetic.lower())
                if candidate_tokens & disliked_lower:
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
