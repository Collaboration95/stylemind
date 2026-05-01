from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any

from neo4j import Driver
from neo4j.exceptions import Neo4jError

from stylemind.models.domain import PersonaSignals
from stylemind.models.schemas import PersonaSnapshot
from stylemind.observability import observe

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.25  # seconds

# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

MERGE_STYLE_PERSONA = """
MERGE (sp:StylePersona {user_id: $user_id})
ON CREATE SET sp.turn_count = 0, sp.created_at = timestamp()
SET sp.turn_count = sp.turn_count + 1
RETURN sp.turn_count AS turn_count
"""

# Batched UNWIND write queries — one round trip per relationship type
BATCH_MERGE_AESTHETICS = """
UNWIND $items AS item
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (a:Aesthetic {name: item.name})
MERGE (sp)-[r:PREFERS]->(a)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

BATCH_MERGE_MATERIALS = """
UNWIND $items AS item
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (m:Material {name: item.name})
MERGE (sp)-[r:DISLIKES]->(m)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

BATCH_MERGE_DISLIKED_PRODUCTS = """
UNWIND $items AS item
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (p:Product {product_id: item.name})
MERGE (sp)-[r:DISLIKES]->(p)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

BATCH_MERGE_BRANDS = """
UNWIND $items AS item
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (b:Brand {name: item.name})
MERGE (sp)-[r:SHOPS_AT]->(b)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

BATCH_MERGE_OCCASIONS = """
UNWIND $items AS item
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (o:Occasion {name: item.name})
MERGE (sp)-[r:INTERESTED_IN]->(o)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

SET_BUDGET_SIGNAL = """
MATCH (sp:StylePersona {user_id: $user_id})
SET sp.budget_signals = CASE
    WHEN sp.budget_signals IS NULL THEN [$budget_entry]
    ELSE sp.budget_signals + [$budget_entry]
END
"""

# Single batched read — one round trip replacing the N+1 pattern.
# Each OPTIONAL MATCH is immediately collected into a list before the next
# OPTIONAL MATCH to prevent cartesian-product row explosion.
GET_PERSONA_ALL = """
MATCH (sp:StylePersona {user_id: $user_id})
OPTIONAL MATCH (sp)-[rp:PREFERS]->(a:Aesthetic)
WITH sp, collect(DISTINCT {aesthetic: a.name, weight: rp.weight, last_seen: rp.last_seen_turn}) AS preferences
OPTIONAL MATCH (sp)-[rd:DISLIKES]->(dm:Material)
WITH sp, preferences,
     collect(DISTINCT {material: dm.name, weight: rd.weight, last_seen: rd.last_seen_turn}) AS dislikes
OPTIONAL MATCH (sp)-[ro:INTERESTED_IN]->(o:Occasion)
WITH sp, preferences, dislikes,
     collect(DISTINCT {occasion: o.name, weight: ro.weight, last_seen: ro.last_seen_turn}) AS occasions
OPTIONAL MATCH (sp)-[rdp:DISLIKES]->(p:Product)
RETURN sp.turn_count AS turn_count,
       sp.budget_signals AS budget_signals,
       preferences,
       dislikes,
       occasions,
       collect(DISTINCT p.product_id) AS disliked_product_ids
"""


_TRANSIENT_ERRORS = (Neo4jError, OSError, TimeoutError)


def _retry[T](
    fn: Callable[[], T],
    attempts: int = _RETRY_ATTEMPTS,
    base_delay: float = _RETRY_BASE_DELAY,
) -> T:
    """Call fn() with exponential backoff on transient Neo4j/network errors only."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < attempts - 1:
                delay = base_delay * (2**attempt)
                logger.warning("retry attempt=%d/%d error=%s sleeping=%.2fs", attempt + 1, attempts, exc, delay)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


class PersonaManager:
    """Manages per-user style persona state in Neo4j."""

    def __init__(self, driver: Driver, decay_rate: float = 0.15, expected_signals_per_turn: float = 3.0) -> None:
        self._driver = driver
        self._decay_rate = decay_rate
        self._expected_signals_per_turn = expected_signals_per_turn

    @observe(name="get_persona")
    def get_persona(self, user_id: str) -> PersonaSnapshot:
        try:
            result = _retry(
                lambda: self._driver.execute_query(GET_PERSONA_ALL, {"user_id": user_id}, database_="neo4j")
            )
            records = [record.data() for record in result.records]
        except Exception as exc:
            logger.warning("persona get_persona query failed user_id=%s error=%s", user_id, exc)
            return PersonaSnapshot()

        if not records or records[0].get("turn_count") is None:
            logger.debug("persona not found for user_id=%s returning default", user_id)
            return PersonaSnapshot()

        row = records[0]
        turn_count: int = row.get("turn_count") or 0
        budget_signals: list[str] = row.get("budget_signals") or []
        preferences: list[dict[str, Any]] = row.get("preferences") or []
        dislikes: list[dict[str, Any]] = row.get("dislikes") or []
        occasions_raw: list[dict[str, Any]] = row.get("occasions") or []
        disliked_products: list[str] = row.get("disliked_product_ids") or []

        # Temporal decay on aesthetics
        decayed_aesthetics: list[tuple[str, float]] = []
        decayed_weights: list[float] = []
        for pref in preferences:
            name = pref.get("aesthetic")
            weight = pref.get("weight")
            last_seen = pref.get("last_seen")
            if name is None or weight is None or last_seen is None:
                continue
            effective = self._apply_decay(float(weight), int(last_seen), turn_count)
            decayed_aesthetics.append((name, effective))
            decayed_weights.append(effective)
        decayed_aesthetics.sort(key=lambda x: x[1], reverse=True)
        preferred_aesthetics = [name for name, _ in decayed_aesthetics[:5]]

        # Disliked materials (persistent, no decay)
        disliked_materials: list[str] = [d["material"] for d in dislikes if d.get("material")]

        # Temporal decay on occasions
        decayed_occasions: list[tuple[str, float]] = []
        for occ in occasions_raw:
            occ_name = occ.get("occasion")
            occ_weight = occ.get("weight")
            occ_last_seen = occ.get("last_seen")
            if occ_name is None or occ_weight is None or occ_last_seen is None:
                continue
            effective = self._apply_decay(float(occ_weight), int(occ_last_seen), turn_count)
            decayed_occasions.append((occ_name, effective))
        decayed_occasions.sort(key=lambda x: x[1], reverse=True)
        top_occasions = [name for name, _ in decayed_occasions[:3]]

        # Budget tier: weighted accumulation (entries stored as "signal:weight" strings).
        # Tiebreak by recency (last position in the list = most recent signal).
        budget_weights: dict[str, float] = {}
        budget_last_seen: dict[str, int] = {}
        for idx, entry in enumerate(budget_signals):
            entry_str = str(entry)
            if ":" in entry_str:
                sig, w_str = entry_str.rsplit(":", 1)
                try:
                    w = float(w_str)
                except ValueError:
                    sig, w = entry_str, 1.0
            else:
                sig, w = entry_str, 1.0
            budget_weights[sig] = budget_weights.get(sig, 0.0) + w
            budget_last_seen[sig] = idx
        budget_tier: str | None = (
            max(budget_weights, key=lambda k: (budget_weights[k], budget_last_seen[k])) if budget_weights else None
        )

        confidence = self._confidence_score(decayed_weights, turn_count)

        logger.info(
            "persona retrieved user_id=%s turn_count=%d confidence=%.2f",
            user_id,
            turn_count,
            confidence,
        )

        return PersonaSnapshot(
            preferred_aesthetics=preferred_aesthetics,
            disliked_materials=disliked_materials,
            budget_tier=budget_tier,
            top_occasions=top_occasions,
            disliked_products=disliked_products,
            confidence_score=confidence,
        )

    @observe(name="update_persona")
    def update_persona(self, user_id: str, signals: PersonaSignals) -> None:
        try:
            with self._driver.session(database="neo4j") as session, session.begin_transaction() as tx:
                # Increment turn_count and get current turn
                result = tx.run(MERGE_STYLE_PERSONA, {"user_id": user_id})
                records = list(result)
                turn_count: int = records[0].data().get("turn_count", 1) if records else 1

                weight = signals.signal_strength

                # Batched PREFERS -> Aesthetic
                if signals.liked_aesthetics:
                    tx.run(
                        BATCH_MERGE_AESTHETICS,
                        {
                            "user_id": user_id,
                            "items": [{"name": a} for a in signals.liked_aesthetics],
                            "weight": weight,
                            "turn": turn_count,
                        },
                    )

                # Batched DISLIKES -> Material
                if signals.disliked_materials:
                    tx.run(
                        BATCH_MERGE_MATERIALS,
                        {
                            "user_id": user_id,
                            "items": [{"name": m} for m in signals.disliked_materials],
                            "weight": weight,
                            "turn": turn_count,
                        },
                    )

                # Batched DISLIKES -> Product (negative sentiment only)
                negative_product_ids = [
                    pid for pid, sentiment in signals.sentiment_on_shown.items() if sentiment == "negative"
                ]
                if negative_product_ids:
                    tx.run(
                        BATCH_MERGE_DISLIKED_PRODUCTS,
                        {
                            "user_id": user_id,
                            "items": [{"name": pid} for pid in negative_product_ids],
                            "weight": weight,
                            "turn": turn_count,
                        },
                    )

                # Batched SHOPS_AT -> Brand
                if signals.brand_mentions:
                    tx.run(
                        BATCH_MERGE_BRANDS,
                        {
                            "user_id": user_id,
                            "items": [{"name": b} for b in signals.brand_mentions],
                            "weight": weight,
                            "turn": turn_count,
                        },
                    )

                # Batched INTERESTED_IN -> Occasion
                if signals.mentioned_occasions:
                    tx.run(
                        BATCH_MERGE_OCCASIONS,
                        {
                            "user_id": user_id,
                            "items": [{"name": o} for o in signals.mentioned_occasions],
                            "weight": weight,
                            "turn": turn_count,
                        },
                    )

                # Budget signal — encoded as "signal:weight" string; Neo4j can't store Maps in property arrays
                if signals.budget_signal:
                    tx.run(
                        SET_BUDGET_SIGNAL,
                        {
                            "user_id": user_id,
                            "budget_entry": f"{signals.budget_signal}:{weight:.2f}",
                        },
                    )

                tx.commit()

            logger.info(
                "persona updated user_id=%s turn=%d aesthetics=%d materials=%d",
                user_id,
                turn_count,
                len(signals.liked_aesthetics),
                len(signals.disliked_materials),
            )

        except Exception as exc:
            logger.warning("persona update_persona failed user_id=%s error=%s", user_id, exc)

    def _apply_decay(self, weight: float, last_seen_turn: int, current_turn: int) -> float:
        """Compute effective weight with exponential temporal decay."""
        delta = max(0, current_turn - last_seen_turn)
        return weight * math.exp(-self._decay_rate * delta)

    def _confidence_score(self, decayed_weights: list[float], turn_count: int) -> float:
        """Confidence = min(1.0, sum(decayed_weights) / (turn_count * expected_signals_per_turn))."""
        if turn_count == 0:
            return 0.0
        return min(1.0, sum(decayed_weights) / (turn_count * self._expected_signals_per_turn))
