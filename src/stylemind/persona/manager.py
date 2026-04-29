from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

from neo4j import Driver

from stylemind.models.schemas import PersonaSignals, PersonaSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher queries for persona management
# ---------------------------------------------------------------------------

MERGE_STYLE_PERSONA = """
MERGE (sp:StylePersona {user_id: $user_id})
ON CREATE SET sp.turn_count = 0, sp.created_at = timestamp()
SET sp.turn_count = sp.turn_count + 1
RETURN sp.turn_count AS turn_count
"""

MERGE_PREFERS_AESTHETIC = """
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (a:Aesthetic {name: $aesthetic_name})
MERGE (sp)-[r:PREFERS]->(a)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

MERGE_DISLIKES_MATERIAL = """
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (m:Material {name: $material_name})
MERGE (sp)-[r:DISLIKES]->(m)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

MERGE_DISLIKES_PRODUCT = """
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (p:Product {product_id: $product_id})
MERGE (sp)-[r:DISLIKES]->(p)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

MERGE_SHOPS_AT_BRAND = """
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (b:Brand {name: $brand_name})
MERGE (sp)-[r:SHOPS_AT]->(b)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""

SET_BUDGET_SIGNAL = """
MATCH (sp:StylePersona {user_id: $user_id})
SET sp.budget_signals = CASE
    WHEN sp.budget_signals IS NULL THEN [$budget_signal]
    ELSE sp.budget_signals + [$budget_signal]
END
"""

GET_PERSONA_DATA = """
MATCH (sp:StylePersona {user_id: $user_id})
OPTIONAL MATCH (sp)-[rp:PREFERS]->(a:Aesthetic)
OPTIONAL MATCH (sp)-[rd:DISLIKES]->(dm:Material)
RETURN sp.turn_count AS turn_count,
       sp.budget_signals AS budget_signals,
       collect(DISTINCT {aesthetic: a.name, weight: rp.weight, last_seen: rp.last_seen_turn}) AS preferences,
       collect(DISTINCT {material: dm.name, weight: rd.weight, last_seen: rd.last_seen_turn}) AS dislikes
"""

GET_OCCASIONS_DATA = """
MATCH (sp:StylePersona {user_id: $user_id})
OPTIONAL MATCH (sp)-[ro:INTERESTED_IN]->(o:Occasion)
RETURN collect(DISTINCT {occasion: o.name, weight: ro.weight, last_seen: ro.last_seen_turn}) AS occasions
"""

MERGE_INTERESTED_IN_OCCASION = """
MATCH (sp:StylePersona {user_id: $user_id})
MATCH (o:Occasion {name: $occasion_name})
MERGE (sp)-[r:INTERESTED_IN]->(o)
ON CREATE SET r.weight = 0, r.last_seen_turn = $turn
SET r.weight = r.weight + $weight, r.last_seen_turn = $turn
"""


class PersonaManager:
    def __init__(self, driver: Driver, decay_rate: float = 0.15, expected_signals_per_turn: float = 3.0) -> None:
        self._driver = driver
        self._decay_rate = decay_rate
        self._expected_signals_per_turn = expected_signals_per_turn

    def get_persona(self, user_id: str) -> PersonaSnapshot:
        """Return persona snapshot. Returns empty default on first turn, NEVER None."""
        try:
            result = self._driver.execute_query(
                GET_PERSONA_DATA,
                {"user_id": user_id},
                database_="neo4j",
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

        # Apply temporal decay to aesthetics
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

        # Disliked materials (no decay — dislikes are persistent)
        disliked_materials: list[str] = []
        for dis in dislikes:
            name = dis.get("material")
            if name:
                disliked_materials.append(name)

        # Budget tier: weighted mode (most frequent budget signal)
        budget_tier: str | None = None
        if budget_signals:
            counter = Counter(budget_signals)
            budget_tier = counter.most_common(1)[0][0]

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
            top_occasions=[],
            confidence_score=confidence,
        )

    def update_persona(self, user_id: str, signals: PersonaSignals) -> None:
        """Merge persona signals into Neo4j graph."""
        try:
            # Increment turn_count and get current turn
            result = self._driver.execute_query(
                MERGE_STYLE_PERSONA,
                {"user_id": user_id},
                database_="neo4j",
            )
            records = [record.data() for record in result.records]
            turn_count: int = records[0].get("turn_count", 1) if records else 1

            weight = signals.signal_strength

            # PREFERS -> Aesthetic
            for aesthetic in signals.liked_aesthetics:
                try:
                    self._driver.execute_query(
                        MERGE_PREFERS_AESTHETIC,
                        {"user_id": user_id, "aesthetic_name": aesthetic, "weight": weight, "turn": turn_count},
                        database_="neo4j",
                    )
                except Exception as exc:
                    logger.warning("persona merge aesthetic failed aesthetic=%s error=%s", aesthetic, exc)

            # DISLIKES -> Material
            for material in signals.disliked_materials:
                try:
                    self._driver.execute_query(
                        MERGE_DISLIKES_MATERIAL,
                        {"user_id": user_id, "material_name": material, "weight": weight, "turn": turn_count},
                        database_="neo4j",
                    )
                except Exception as exc:
                    logger.warning("persona merge material failed material=%s error=%s", material, exc)

            # DISLIKES -> Product (for negative sentiment)
            for product_id, sentiment in signals.sentiment_on_shown.items():
                if sentiment == "negative":
                    try:
                        self._driver.execute_query(
                            MERGE_DISLIKES_PRODUCT,
                            {"user_id": user_id, "product_id": product_id, "weight": weight, "turn": turn_count},
                            database_="neo4j",
                        )
                    except Exception as exc:
                        logger.warning("persona merge dislike product failed product_id=%s error=%s", product_id, exc)

            # SHOPS_AT -> Brand
            for brand in signals.brand_mentions:
                try:
                    self._driver.execute_query(
                        MERGE_SHOPS_AT_BRAND,
                        {"user_id": user_id, "brand_name": brand, "weight": weight, "turn": turn_count},
                        database_="neo4j",
                    )
                except Exception as exc:
                    logger.warning("persona merge brand failed brand=%s error=%s", brand, exc)

            # INTERESTED_IN -> Occasion
            for occasion in signals.mentioned_occasions:
                try:
                    self._driver.execute_query(
                        MERGE_INTERESTED_IN_OCCASION,
                        {"user_id": user_id, "occasion_name": occasion, "weight": weight, "turn": turn_count},
                        database_="neo4j",
                    )
                except Exception as exc:
                    logger.warning("persona merge occasion failed occasion=%s error=%s", occasion, exc)

            # Budget signal: append to list on StylePersona node
            if signals.budget_signal:
                try:
                    self._driver.execute_query(
                        SET_BUDGET_SIGNAL,
                        {"user_id": user_id, "budget_signal": signals.budget_signal},
                        database_="neo4j",
                    )
                except Exception as exc:
                    logger.warning("persona set budget_signal failed budget=%s error=%s", signals.budget_signal, exc)

            logger.info(
                "persona updated user_id=%s turn=%d aesthetics=%d materials=%d",
                user_id,
                turn_count,
                len(signals.liked_aesthetics),
                len(signals.disliked_materials),
            )

        except Exception as exc:
            logger.warning("persona update_persona failed user_id=%s error=%s", user_id, exc)

    def get_persona_snapshot(self, user_id: str) -> PersonaSnapshot:
        """Alias for get_persona - matches spec R5 JSON shape exactly."""
        return self.get_persona(user_id)

    def _apply_decay(self, weight: float, last_seen_turn: int, current_turn: int) -> float:
        """Compute effective weight with exponential temporal decay.

        effective_weight = raw_weight * exp(-decay_rate * (current_turn - last_seen_turn))
        """
        delta = max(0, current_turn - last_seen_turn)
        return weight * math.exp(-self._decay_rate * delta)

    def _confidence_score(self, decayed_weights: list[float], turn_count: int) -> float:
        """Confidence = min(1.0, sum(decayed_weights) / (turn_count * expected_signals_per_turn))."""
        if turn_count == 0:
            return 0.0
        return min(1.0, sum(decayed_weights) / (turn_count * self._expected_signals_per_turn))
