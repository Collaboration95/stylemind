from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from stylemind.models.schemas import PersonaSignals, PersonaSnapshot
from stylemind.persona.manager import PersonaManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(driver: MagicMock, decay_rate: float = 0.15) -> PersonaManager:
    return PersonaManager(driver=driver, decay_rate=decay_rate, expected_signals_per_turn=3.0)


def _make_driver_with_records(records_data: list[dict]) -> MagicMock:
    """Build a mock driver whose execute_query returns given records."""
    driver = MagicMock()
    mock_records = []
    for data in records_data:
        rec = MagicMock()
        rec.data.return_value = data
        mock_records.append(rec)
    mock_result = MagicMock()
    mock_result.records = mock_records
    driver.execute_query.return_value = mock_result
    return driver


def _make_update_driver(turn_count: int = 1) -> MagicMock:
    """Build a driver suitable for update_persona calls."""
    driver = MagicMock()
    mock_rec = MagicMock()
    mock_rec.data.return_value = {"turn_count": turn_count}
    mock_result = MagicMock()
    mock_result.records = [mock_rec]
    driver.execute_query.return_value = mock_result
    return driver


# ---------------------------------------------------------------------------
# PersonaManager — focused unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_first_turn_empty_persona() -> None:
    """get_persona for new user_id returns PersonaSnapshot with confidence=0.0 and empty lists."""
    driver = _make_driver_with_records([])
    manager = _make_manager(driver)

    snapshot = manager.get_persona("brand_new_user")

    assert isinstance(snapshot, PersonaSnapshot)
    assert snapshot.confidence_score == 0.0
    assert snapshot.preferred_aesthetics == []
    assert snapshot.disliked_materials == []
    assert snapshot.budget_tier is None
    assert snapshot.top_occasions == []


@pytest.mark.unit
def test_update_persona_merges_aesthetic_prefers() -> None:
    """update_persona with liked_aesthetics → execute_query called with PREFERS aesthetics merge."""
    driver = _make_update_driver(turn_count=1)
    manager = _make_manager(driver)

    signals = PersonaSignals(liked_aesthetics=["Quiet Luxury"], signal_strength=0.8)
    manager.update_persona("user_001", signals)

    # Find calls that contain PREFERS
    calls_str = [str(call) for call in driver.execute_query.call_args_list]
    prefers_calls = [c for c in calls_str if "PREFERS" in c or "aesthetic_name" in c]
    assert len(prefers_calls) >= 1, "Expected at least one PREFERS/aesthetic_name call"


@pytest.mark.unit
def test_update_persona_merges_disliked_material() -> None:
    """update_persona with disliked_materials → execute_query called with DISLIKES material merge."""
    driver = _make_update_driver(turn_count=1)
    manager = _make_manager(driver)

    signals = PersonaSignals(disliked_materials=["Polyester"], signal_strength=0.9)
    manager.update_persona("user_002", signals)

    calls_str = [str(call) for call in driver.execute_query.call_args_list]
    dislikes_calls = [c for c in calls_str if "material_name" in c or "DISLIKES" in c]
    assert len(dislikes_calls) >= 1, "Expected at least one DISLIKES/material_name call"


@pytest.mark.unit
def test_temporal_decay_reduces_weight() -> None:
    """effective_weight = raw_weight * exp(-0.15 * delta_turn), verify formula directly."""
    driver = MagicMock()
    manager = _make_manager(driver, decay_rate=0.15)

    # Test various delta values
    for delta in [1, 3, 5, 10]:
        raw_weight = 1.0
        last_seen = 10
        current = 10 + delta
        result = manager._apply_decay(raw_weight, last_seen_turn=last_seen, current_turn=current)
        expected = raw_weight * math.exp(-0.15 * delta)
        assert result == pytest.approx(expected, rel=1e-6), (
            f"Decay mismatch at delta={delta}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_confidence_formula_zero_on_first_turn() -> None:
    """After 1 turn with no signals (empty weights), confidence is 0.0."""
    driver = MagicMock()
    manager = _make_manager(driver)

    # No decayed weights, turn_count=1 → confidence = 0.0 / (1 * 3.0) = 0.0
    score = manager._confidence_score([], turn_count=1)
    assert score == pytest.approx(0.0)


@pytest.mark.unit
def test_confidence_grows_with_signals() -> None:
    """After 5 turns with multiple signals, confidence > 0.3."""
    driver = MagicMock()
    manager = _make_manager(driver)

    # 8 signals of weight 0.8 across 5 turns: sum=6.4 / (5*3.0)=0.427 > 0.3
    weights = [0.8] * 8
    score = manager._confidence_score(weights, turn_count=5)
    assert score > 0.3, f"Confidence should exceed 0.3 with 8 signals across 5 turns, got {score}"


@pytest.mark.unit
def test_budget_accumulation() -> None:
    """Multiple turns with same budget_signal → budget_tier set correctly via mode."""
    # Simulate get_persona receiving a record with 3 'premium' and 1 'luxury' signals
    driver = _make_driver_with_records(
        [
            {
                "turn_count": 4,
                "budget_signals": ["premium", "premium", "luxury", "premium"],
                "preferences": [],
                "dislikes": [],
            }
        ]
    )
    manager = _make_manager(driver)

    snapshot = manager.get_persona("user_budget")

    # Most common = "premium" (3 out of 4)
    assert snapshot.budget_tier == "premium", f"Expected 'premium', got {snapshot.budget_tier!r}"


@pytest.mark.unit
def test_get_persona_returns_default_never_none() -> None:
    """Even if DB returns nothing (or raises), get_persona never returns None."""
    # Test 1: empty records
    driver_empty = _make_driver_with_records([])
    manager_empty = _make_manager(driver_empty)
    snapshot_empty = manager_empty.get_persona("ghost_user")
    assert snapshot_empty is not None
    assert isinstance(snapshot_empty, PersonaSnapshot)

    # Test 2: DB raises exception
    driver_broken = MagicMock()
    driver_broken.execute_query.side_effect = RuntimeError("DB connection failed")
    manager_broken = _make_manager(driver_broken)
    snapshot_broken = manager_broken.get_persona("ghost_user_2")
    assert snapshot_broken is not None
    assert isinstance(snapshot_broken, PersonaSnapshot)
    assert snapshot_broken.confidence_score == 0.0

    # Test 3: record with turn_count = None
    driver_null = _make_driver_with_records([{"turn_count": None}])
    manager_null = _make_manager(driver_null)
    snapshot_null = manager_null.get_persona("null_user")
    assert snapshot_null is not None
    assert isinstance(snapshot_null, PersonaSnapshot)
    assert snapshot_null.confidence_score == 0.0


@pytest.mark.unit
def test_update_persona_brand_mention_calls_shops_at() -> None:
    """Brand mention in signals → SHOPS_AT brand merge is called."""
    driver = _make_update_driver(turn_count=2)
    manager = _make_manager(driver)

    signals = PersonaSignals(brand_mentions=["Arket"], signal_strength=0.6)
    manager.update_persona("user_brand", signals)

    calls_str = [str(call) for call in driver.execute_query.call_args_list]
    brand_calls = [c for c in calls_str if "brand_name" in c or "SHOPS_AT" in c]
    assert len(brand_calls) >= 1, "Expected at least one brand/SHOPS_AT call"


@pytest.mark.unit
def test_update_persona_budget_signal_stored() -> None:
    """budget_signal in signals → SET_BUDGET_SIGNAL query is called."""
    driver = _make_update_driver(turn_count=1)
    manager = _make_manager(driver)

    signals = PersonaSignals(budget_signal="luxury", signal_strength=0.7)
    manager.update_persona("user_luxury", signals)

    calls_str = [str(call) for call in driver.execute_query.call_args_list]
    budget_calls = [c for c in calls_str if "budget_signal" in c]
    assert len(budget_calls) >= 1, "Expected at least one budget_signal call"


@pytest.mark.unit
def test_get_persona_with_decayed_aesthetics_sorted() -> None:
    """Aesthetics with higher effective weight appear first in preferred_aesthetics."""
    # Quiet Luxury: weight=3.0, last_seen=5 (delta=0 → high effective)
    # Boho: weight=1.0, last_seen=5 (delta=0 → lower effective)
    driver = _make_driver_with_records(
        [
            {
                "turn_count": 5,
                "budget_signals": None,
                "preferences": [
                    {"aesthetic": "Quiet Luxury", "weight": 3.0, "last_seen": 5},
                    {"aesthetic": "Boho", "weight": 1.0, "last_seen": 5},
                ],
                "dislikes": [],
            }
        ]
    )
    manager = _make_manager(driver)

    snapshot = manager.get_persona("user_sorted")

    # Both should appear
    assert "Quiet Luxury" in snapshot.preferred_aesthetics
    assert "Boho" in snapshot.preferred_aesthetics

    # Quiet Luxury should appear before Boho (higher weight = higher effective score)
    ql_idx = snapshot.preferred_aesthetics.index("Quiet Luxury")
    boho_idx = snapshot.preferred_aesthetics.index("Boho")
    assert ql_idx < boho_idx, "Quiet Luxury (higher weight) should rank before Boho"
