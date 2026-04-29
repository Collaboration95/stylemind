from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stylemind.models.domain import PersonaSignals
from stylemind.models.schemas import PersonaSnapshot
from stylemind.persona.inference import PersonaInferenceEngine  # noqa: I001
from stylemind.persona.manager import PersonaManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inference_engine() -> PersonaInferenceEngine:
    """Build a PersonaInferenceEngine with dummy config (no real API calls)."""
    from stylemind.config import ExtractionLLMConfig

    config = ExtractionLLMConfig(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4.1-nano",
    )
    return PersonaInferenceEngine(config)


def _make_manager(driver: MagicMock) -> PersonaManager:
    return PersonaManager(driver=driver, decay_rate=0.15, expected_signals_per_turn=3.0)


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


# ---------------------------------------------------------------------------
# Inference engine — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_signals_disliked_material():
    """LLM returning disliked_materials=[Polyester] -> PersonaSignals.disliked_materials == ['Polyester']."""
    engine = _make_inference_engine()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"disliked_materials": ["Polyester"], "liked_aesthetics": [], '
        '"mentioned_occasions": [], "budget_signal": null, "color_preferences": [], '
        '"brand_mentions": [], "sentiment_on_shown": {}, "signal_strength": 0.9}'
    )

    with patch.object(engine._client.chat.completions, "create", return_value=mock_response):
        signals = engine.extract_signals("I hate polyester, it's so scratchy", [], [])

    assert signals.disliked_materials == ["Polyester"]
    assert signals.signal_strength == pytest.approx(0.9)


@pytest.mark.unit
def test_extract_signals_occasion():
    """'something for a date night' -> mentioned_occasions includes 'Date Night'."""
    engine = _make_inference_engine()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"mentioned_occasions": ["Date Night"], "liked_aesthetics": ["Quiet Luxury", "Casual Minimalism"], '
        '"disliked_materials": [], "budget_signal": null, "color_preferences": [], '
        '"brand_mentions": [], "sentiment_on_shown": {}, "signal_strength": 0.7}'
    )

    with patch.object(engine._client.chat.completions, "create", return_value=mock_response):
        signals = engine.extract_signals("something for a date night that's minimal", [], [])

    assert "Date Night" in signals.mentioned_occasions


@pytest.mark.unit
def test_extract_signals_llm_failure_returns_empty():
    """When LLM raises an exception, extract_signals returns empty PersonaSignals."""
    engine = _make_inference_engine()

    with patch.object(engine._client.chat.completions, "create", side_effect=RuntimeError("API error")):
        signals = engine.extract_signals("show me something nice", [], [])

    assert signals == PersonaSignals()
    assert signals.liked_aesthetics == []
    assert signals.disliked_materials == []
    assert signals.budget_signal is None


@pytest.mark.unit
def test_extract_signals_budget_signal():
    """'on a tight budget' -> budget_signal == 'budget'."""
    engine = _make_inference_engine()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"budget_signal": "budget", "liked_aesthetics": [], "disliked_materials": [], '
        '"mentioned_occasions": [], "color_preferences": [], '
        '"brand_mentions": [], "sentiment_on_shown": {}, "signal_strength": 0.6}'
    )

    with patch.object(engine._client.chat.completions, "create", return_value=mock_response):
        signals = engine.extract_signals("I'm on a tight budget", [], [])

    assert signals.budget_signal == "budget"


# ---------------------------------------------------------------------------
# PersonaManager — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_persona_returns_default_for_missing_user():
    """When Neo4j returns no records, get_persona returns PersonaSnapshot with confidence=0.0."""
    driver = _make_driver_with_records([])
    manager = _make_manager(driver)

    snapshot = manager.get_persona("unknown_user")

    assert isinstance(snapshot, PersonaSnapshot)
    assert snapshot.confidence_score == 0.0
    assert snapshot.preferred_aesthetics == []
    assert snapshot.disliked_materials == []
    assert snapshot.budget_tier is None


@pytest.mark.unit
def test_get_persona_returns_default_when_turn_count_none():
    """When record has turn_count=None, get_persona returns empty PersonaSnapshot."""
    driver = _make_driver_with_records([{"turn_count": None}])
    manager = _make_manager(driver)

    snapshot = manager.get_persona("user_123")

    assert snapshot.confidence_score == 0.0


@pytest.mark.unit
def test_temporal_decay_math():
    """_apply_decay(1.0, last_seen=5, current=8) ≈ exp(-0.15 * 3)."""
    driver = MagicMock()
    manager = _make_manager(driver)

    result = manager._apply_decay(1.0, last_seen_turn=5, current_turn=8)
    expected = math.exp(-0.15 * 3)

    assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
def test_temporal_decay_no_delta():
    """_apply_decay with same turn returns raw weight unchanged."""
    driver = MagicMock()
    manager = _make_manager(driver)

    result = manager._apply_decay(0.8, last_seen_turn=4, current_turn=4)
    assert result == pytest.approx(0.8, rel=1e-6)


@pytest.mark.unit
def test_temporal_decay_negative_delta_clamped():
    """_apply_decay with current < last_seen clamps delta to 0."""
    driver = MagicMock()
    manager = _make_manager(driver)

    result = manager._apply_decay(1.0, last_seen_turn=10, current_turn=5)
    # delta clamped to 0, so result == weight * exp(0) == 1.0
    assert result == pytest.approx(1.0, rel=1e-6)


@pytest.mark.unit
def test_confidence_score_calculation():
    """min(1.0, sum(weights) / (turns * 3.0)) capped at 1.0."""
    driver = MagicMock()
    manager = _make_manager(driver)

    # 3 weights of 1.0, 2 turns -> 3.0 / (2 * 3.0) = 0.5
    score = manager._confidence_score([1.0, 1.0, 1.0], turn_count=2)
    assert score == pytest.approx(0.5, rel=1e-6)

    # High signals -> capped at 1.0
    score_capped = manager._confidence_score([1.0] * 20, turn_count=1)
    assert score_capped == pytest.approx(1.0, rel=1e-6)

    # Zero turns -> 0.0
    score_zero = manager._confidence_score([], turn_count=0)
    assert score_zero == 0.0


@pytest.mark.unit
def test_confidence_after_5_turns():
    """5 signals with weight 1.0 across 5 turns -> confidence > 0.5."""
    driver = MagicMock()
    manager = _make_manager(driver)

    # 5 signals of weight 1.0 over 5 turns = 5.0 / (5 * 3.0) = 0.333
    # To get > 0.5, we need more signals. With 6 signals:
    # 6 / (5 * 3) = 0.4 — still < 0.5.
    # With 10 signals across 5 turns: 10/(5*3) = 0.667 > 0.5
    score = manager._confidence_score([1.0] * 10, turn_count=5)
    assert score > 0.5


def _make_tx_driver(turn_count: int = 1) -> MagicMock:
    """Build a driver that supports the transaction-based update_persona API."""
    driver = MagicMock()

    mock_record = MagicMock()
    mock_record.data.return_value = {"turn_count": turn_count}
    mock_run_result = MagicMock()
    mock_run_result.__iter__ = MagicMock(return_value=iter([mock_record]))

    mock_tx = MagicMock()
    mock_tx.run.return_value = mock_run_result
    mock_tx.__enter__ = MagicMock(return_value=mock_tx)
    mock_tx.__exit__ = MagicMock(return_value=False)

    mock_session = MagicMock()
    mock_session.begin_transaction.return_value = mock_tx
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    driver.session.return_value = mock_session
    return driver


@pytest.mark.unit
def test_update_persona_calls_merge_style_persona():
    """update_persona opens a session, begins a transaction and calls tx.run at least once."""
    driver = _make_tx_driver(turn_count=1)
    manager = _make_manager(driver)

    signals = PersonaSignals(liked_aesthetics=["Quiet Luxury"], signal_strength=0.8)
    manager.update_persona("user_abc", signals)

    driver.session.assert_called_once_with(database="neo4j")
    mock_session = driver.session.return_value.__enter__.return_value
    mock_session.begin_transaction.assert_called_once()

    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value
    # MERGE_STYLE_PERSONA + at least one BATCH_MERGE_AESTHETICS
    assert mock_tx.run.call_count >= 2


@pytest.mark.unit
def test_update_persona_negative_sentiment_merges_dislikes_product():
    """Negative sentiment_on_shown triggers a DISLIKES product UNWIND batch query."""
    driver = _make_tx_driver(turn_count=1)
    manager = _make_manager(driver)

    signals = PersonaSignals(sentiment_on_shown={"P001": "negative", "P002": "positive"}, signal_strength=0.7)
    manager.update_persona("user_xyz", signals)

    mock_session = driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value

    all_calls_str = [str(c) for c in mock_tx.run.call_args_list]

    # P001 (negative) should appear in a DISLIKES call
    dislike_calls = [c for c in all_calls_str if "P001" in c]
    assert len(dislike_calls) > 0, "Expected a tx.run call containing P001 (negative sentiment)"

    # P002 (positive) must NOT appear in any call
    p002_calls = [c for c in all_calls_str if "P002" in c]
    assert len(p002_calls) == 0, "P002 (positive sentiment) must not appear in any DISLIKES call"


@pytest.mark.unit
def test_get_persona_with_aesthetics_and_decay():
    """Persona with aesthetic preferences returns sorted decayed aesthetics."""
    driver = _make_driver_with_records(
        [
            {
                "turn_count": 5,
                "budget_signals": None,
                "preferences": [
                    {"aesthetic": "Quiet Luxury", "weight": 2.0, "last_seen": 3},
                    {"aesthetic": "Boho", "weight": 1.0, "last_seen": 5},
                ],
                "dislikes": [],
                "occasions": [],
                "disliked_product_ids": [],
            }
        ]
    )
    snapshot = _make_manager(driver).get_persona("user_test")

    assert "Quiet Luxury" in snapshot.preferred_aesthetics or "Boho" in snapshot.preferred_aesthetics
    assert snapshot.confidence_score > 0.0


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_persona_5_turn_evolution():
    """Integration: simulate 5 turns of persona updates and verify state evolution.

    Intercepts both the transaction-based update_persona writes and the single
    execute_query read to simulate an in-memory Neo4j store.
    """
    stored_turn = 0
    stored_aesthetics: dict[str, dict] = {}
    stored_budget_signals: list[dict[str, Any]] = []

    # ---- transaction tx.run interceptor (update_persona writes) ----
    def tx_run_side_effect(query: str, params: dict) -> MagicMock:
        nonlocal stored_turn
        result = MagicMock()

        if "sp.turn_count = sp.turn_count + 1" in query:
            stored_turn += 1
            rec = MagicMock()
            rec.data.return_value = {"turn_count": stored_turn}
            result.__iter__ = MagicMock(return_value=iter([rec]))
        elif "PREFERS" in query and "items" in params:
            for item in params.get("items", []):
                name = item["name"]
                w = params["weight"]
                t = params["turn"]
                if name in stored_aesthetics:
                    stored_aesthetics[name]["weight"] += w
                    stored_aesthetics[name]["last_seen"] = t
                else:
                    stored_aesthetics[name] = {"weight": w, "last_seen": t}
            result.__iter__ = MagicMock(return_value=iter([]))
        elif "budget_entry" in params:
            stored_budget_signals.append(params["budget_entry"])
            result.__iter__ = MagicMock(return_value=iter([]))
        else:
            result.__iter__ = MagicMock(return_value=iter([]))

        return result

    mock_tx = MagicMock()
    mock_tx.run.side_effect = tx_run_side_effect
    mock_tx.__enter__ = MagicMock(return_value=mock_tx)
    mock_tx.__exit__ = MagicMock(return_value=False)

    mock_session = MagicMock()
    mock_session.begin_transaction.return_value = mock_tx
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    # ---- execute_query interceptor (get_persona single read) ----
    def execute_query_side_effect(query: str, params: dict, database_: str = "neo4j") -> MagicMock:
        prefs = [
            {"aesthetic": n, "weight": v["weight"], "last_seen": v["last_seen"]}
            for n, v in stored_aesthetics.items()
        ]
        rec = MagicMock()
        rec.data.return_value = {
            "turn_count": stored_turn,
            "budget_signals": stored_budget_signals if stored_budget_signals else None,
            "preferences": prefs,
            "dislikes": [],
            "occasions": [],
            "disliked_product_ids": [],
        }
        result = MagicMock()
        result.records = [rec]
        return result

    driver = MagicMock()
    driver.session.return_value = mock_session
    driver.execute_query.side_effect = execute_query_side_effect

    manager = _make_manager(driver)

    signals_sequence = [
        PersonaSignals(liked_aesthetics=["Quiet Luxury"], budget_signal="premium", signal_strength=0.8),
        PersonaSignals(liked_aesthetics=["Quiet Luxury", "Old Money"], signal_strength=0.7),
        PersonaSignals(liked_aesthetics=["Old Money"], budget_signal="premium", signal_strength=0.9),
        PersonaSignals(liked_aesthetics=["Quiet Luxury"], signal_strength=0.6),
        PersonaSignals(liked_aesthetics=["Quiet Luxury"], budget_signal="luxury", signal_strength=0.75),
    ]

    user_id = "test_user_evolution"
    for signals in signals_sequence:
        manager.update_persona(user_id, signals)

    snapshot = manager.get_persona(user_id)

    assert "Quiet Luxury" in snapshot.preferred_aesthetics
    assert snapshot.budget_tier == "premium"
    assert snapshot.confidence_score > 0.0
    assert stored_turn == 5
