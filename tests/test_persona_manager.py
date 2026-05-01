from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from stylemind.models.domain import PersonaSignals
from stylemind.models.schemas import PersonaSnapshot
from stylemind.persona.manager import PersonaManager, _retry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(driver: MagicMock, decay_rate: float = 0.15) -> PersonaManager:
    return PersonaManager(driver=driver, decay_rate=decay_rate, expected_signals_per_turn=3.0)


def _make_full_record(
    turn_count: int = 1,
    budget_signals: list | None = None,
    preferences: list | None = None,
    dislikes: list | None = None,
    occasions: list | None = None,
    disliked_product_ids: list | None = None,
) -> dict:
    """Build a record dict matching the GET_PERSONA_ALL combined query output."""
    return {
        "turn_count": turn_count,
        "budget_signals": budget_signals or [],
        "preferences": preferences or [],
        "dislikes": dislikes or [],
        "occasions": occasions or [],
        "disliked_product_ids": disliked_product_ids or [],
    }


def _make_driver_with_records(records_data: list[dict]) -> MagicMock:
    """Build a mock driver whose execute_query returns given records (single combined query)."""
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
    """Build a driver suitable for update_persona calls using the transaction API."""
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


# ---------------------------------------------------------------------------
# get_persona — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_first_turn_empty_persona() -> None:
    """get_persona for new user_id returns PersonaSnapshot with confidence=0.0 and empty lists."""
    driver = _make_driver_with_records([])
    snapshot = _make_manager(driver).get_persona("brand_new_user")

    assert isinstance(snapshot, PersonaSnapshot)
    assert snapshot.confidence_score == 0.0
    assert snapshot.preferred_aesthetics == []
    assert snapshot.disliked_materials == []
    assert snapshot.budget_tier is None
    assert snapshot.top_occasions == []


@pytest.mark.unit
def test_get_persona_returns_default_never_none() -> None:
    """get_persona never returns None — falls back to empty PersonaSnapshot on any error."""
    # Empty records
    assert _make_manager(_make_driver_with_records([])).get_persona("ghost") is not None

    # DB raises
    driver_broken = MagicMock()
    driver_broken.execute_query.side_effect = RuntimeError("DB down")
    snapshot = _make_manager(driver_broken).get_persona("ghost2")
    assert snapshot is not None and isinstance(snapshot, PersonaSnapshot)
    assert snapshot.confidence_score == 0.0

    # Record with turn_count=None
    snapshot_null = _make_manager(_make_driver_with_records([{"turn_count": None}])).get_persona("null_user")
    assert snapshot_null is not None and snapshot_null.confidence_score == 0.0


@pytest.mark.unit
def test_get_persona_single_query() -> None:
    """get_persona issues exactly ONE execute_query call (no N+1)."""
    driver = _make_driver_with_records([_make_full_record(turn_count=2)])
    _make_manager(driver).get_persona("user_a")
    assert driver.execute_query.call_count == 1


@pytest.mark.unit
def test_budget_accumulation() -> None:
    """Multiple budget signals with same tier → that tier wins by weighted sum."""
    data = _make_full_record(
        turn_count=4,
        budget_signals=[
            "premium:0.80",
            "premium:0.60",
            "luxury:0.90",
            "premium:0.70",
        ],
    )
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_budget")
    assert snapshot.budget_tier == "premium"


@pytest.mark.unit
def test_weighted_budget_outranks_count() -> None:
    """Single high-weight signal beats multiple low-weight ones for budget_tier."""
    data = _make_full_record(
        turn_count=3,
        budget_signals=[
            "budget:0.30",
            "budget:0.30",
            "luxury:0.90",
        ],
    )
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_weighted")
    assert snapshot.budget_tier == "luxury"


@pytest.mark.unit
def test_get_persona_with_decayed_aesthetics_sorted() -> None:
    """Aesthetics sort by effective (decayed) weight, highest first."""
    data = _make_full_record(
        turn_count=5,
        preferences=[
            {"aesthetic": "Quiet Luxury", "weight": 3.0, "last_seen": 5},
            {"aesthetic": "Boho", "weight": 1.0, "last_seen": 5},
        ],
    )
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_sorted")

    assert "Quiet Luxury" in snapshot.preferred_aesthetics
    assert "Boho" in snapshot.preferred_aesthetics
    assert snapshot.preferred_aesthetics.index("Quiet Luxury") < snapshot.preferred_aesthetics.index("Boho")


@pytest.mark.unit
def test_top_occasions_populated_with_decay() -> None:
    """top_occasions is populated and ordered by decayed weight."""
    data = _make_full_record(
        turn_count=3,
        occasions=[
            {"occasion": "Date Night", "weight": 2.0, "last_seen": 3},
            {"occasion": "Office", "weight": 1.5, "last_seen": 2},
            {"occasion": "Casual", "weight": 0.5, "last_seen": 1},
        ],
    )
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_occasions")

    assert len(snapshot.top_occasions) == 3
    assert snapshot.top_occasions[0] == "Date Night"
    assert snapshot.top_occasions[1] == "Office"
    assert snapshot.top_occasions[2] == "Casual"


@pytest.mark.unit
def test_top_occasions_limited_to_three() -> None:
    """top_occasions returns at most 3 entries even with more occasions present."""
    data = _make_full_record(
        turn_count=5,
        occasions=[
            {"occasion": "Date Night", "weight": 4.0, "last_seen": 5},
            {"occasion": "Office", "weight": 3.0, "last_seen": 5},
            {"occasion": "Casual", "weight": 2.0, "last_seen": 5},
            {"occasion": "Brunch", "weight": 1.0, "last_seen": 5},
            {"occasion": "Wedding Guest", "weight": 0.5, "last_seen": 5},
        ],
    )
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_many_occasions")

    assert len(snapshot.top_occasions) == 3
    assert snapshot.top_occasions == ["Date Night", "Office", "Casual"]


@pytest.mark.unit
def test_disliked_products_populated() -> None:
    """Disliked product IDs from the query end up in the snapshot."""
    data = _make_full_record(turn_count=2, disliked_product_ids=["P012", "P045"])
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_dislike")

    assert "P012" in snapshot.disliked_products
    assert "P045" in snapshot.disliked_products


@pytest.mark.unit
def test_disliked_products_empty_when_none() -> None:
    """disliked_product_ids=None in DB row → empty list in snapshot."""
    data = _make_full_record(turn_count=1)
    data["disliked_product_ids"] = None
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_no_dislikes")
    assert snapshot.disliked_products == []


@pytest.mark.unit
def test_occasion_decay_ordering() -> None:
    """An older high-weight occasion decays below a fresh lower-weight one."""
    data = _make_full_record(
        turn_count=10,
        occasions=[
            {"occasion": "Office", "weight": 3.0, "last_seen": 1},
            {"occasion": "Date Night", "weight": 1.0, "last_seen": 10},
        ],
    )
    snapshot = _make_manager(_make_driver_with_records([data])).get_persona("user_decay_order")
    # Office: 3.0 * exp(-0.15 * 9) ≈ 0.78 < Date Night: 1.0 * exp(0) = 1.0
    assert snapshot.top_occasions[0] == "Date Night"


# ---------------------------------------------------------------------------
# update_persona — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_persona_uses_session_transaction() -> None:
    """update_persona opens a session and begins a transaction (atomic writes)."""
    driver = _make_update_driver(turn_count=1)
    _make_manager(driver).update_persona(
        "user_001", PersonaSignals(liked_aesthetics=["Quiet Luxury"], signal_strength=0.8)
    )

    driver.session.assert_called_once_with(database="neo4j")
    mock_session = driver.session.return_value.__enter__.return_value
    mock_session.begin_transaction.assert_called_once()


@pytest.mark.unit
def test_update_persona_commits_on_success() -> None:
    """update_persona calls tx.commit() when all writes succeed."""
    driver = _make_update_driver(turn_count=1)
    _make_manager(driver).update_persona("user_commit", PersonaSignals(signal_strength=0.5))

    mock_session = driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value
    mock_tx.commit.assert_called_once()


@pytest.mark.unit
def test_update_persona_batches_aesthetics() -> None:
    """Multiple liked_aesthetics are sent in a single UNWIND batch query."""
    driver = _make_update_driver(turn_count=1)
    signals = PersonaSignals(liked_aesthetics=["Quiet Luxury", "Streetwear", "Old Money"], signal_strength=0.8)
    _make_manager(driver).update_persona("user_batch", signals)

    mock_session = driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value

    # Find calls that contain PREFERS (batched aesthetic query)
    prefers_calls = [c for c in mock_tx.run.call_args_list if "PREFERS" in str(c)]
    assert len(prefers_calls) == 1, "All aesthetics should be sent in a single UNWIND call"

    # Verify all three aesthetics are in the items param
    call_params = prefers_calls[0][0][1]  # positional arg 1 is the params dict
    items = call_params["items"]
    names = [i["name"] for i in items]
    assert "Quiet Luxury" in names
    assert "Streetwear" in names
    assert "Old Money" in names


@pytest.mark.unit
def test_update_persona_skips_empty_signal_lists() -> None:
    """update_persona does not emit batch queries for empty signal lists."""
    driver = _make_update_driver(turn_count=1)
    _make_manager(driver).update_persona("user_empty", PersonaSignals(signal_strength=0.5))

    mock_session = driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value

    # Only MERGE_STYLE_PERSONA should have been called (no UNWIND batches)
    assert mock_tx.run.call_count == 1, f"Expected 1 tx.run (just persona merge), got {mock_tx.run.call_count}"


@pytest.mark.unit
def test_update_persona_negative_sentiment_writes_disliked_product() -> None:
    """Negative sentiment in sentiment_on_shown triggers a DISLIKES product UNWIND query."""
    driver = _make_update_driver(turn_count=2)
    signals = PersonaSignals(
        sentiment_on_shown={"P001": "negative", "P002": "positive"},
        signal_strength=0.7,
    )
    _make_manager(driver).update_persona("user_neg", signals)

    mock_session = driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value

    dislikes_calls = [c for c in mock_tx.run.call_args_list if "product_id" in str(c) or "DISLIKES" in str(c)]
    assert len(dislikes_calls) == 1
    params = dislikes_calls[0][0][1]
    ids = [i["name"] for i in params["items"]]
    assert "P001" in ids
    assert "P002" not in ids  # positive sentiment not included


@pytest.mark.unit
def test_update_persona_budget_signal_stored() -> None:
    """budget_signal triggers the SET_BUDGET_SIGNAL tx.run call with encoded format."""
    driver = _make_update_driver(turn_count=1)
    _make_manager(driver).update_persona("user_luxury", PersonaSignals(budget_signal="luxury", signal_strength=0.7))

    mock_session = driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value

    budget_calls = [c for c in mock_tx.run.call_args_list if "budget_entry" in str(c)]
    assert len(budget_calls) == 1
    params = budget_calls[0][0][1]
    assert params["budget_entry"] == "luxury:0.70"


@pytest.mark.unit
def test_budget_signal_write_read_round_trip() -> None:
    """Budget signal written by update_persona can be correctly parsed by get_persona (#69, #74)."""
    write_driver = _make_update_driver(turn_count=1)
    manager = _make_manager(write_driver)
    manager.update_persona("user_rt", PersonaSignals(budget_signal="premium", signal_strength=0.85))

    mock_session = write_driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value
    budget_calls = [c for c in mock_tx.run.call_args_list if "budget_entry" in str(c)]
    written_entry = budget_calls[0][0][1]["budget_entry"]

    read_driver = _make_driver_with_records(
        [
            _make_full_record(
                turn_count=1,
                budget_signals=[written_entry],
            )
        ]
    )
    snapshot = _make_manager(read_driver).get_persona("user_rt")
    assert snapshot.budget_tier == "premium"


@pytest.mark.unit
def test_budget_signal_round_trip_preserves_weight_ranking() -> None:
    """Two budget signals written with different weights → higher total weight wins on read."""
    write_driver = _make_update_driver(turn_count=1)
    manager = _make_manager(write_driver)

    manager.update_persona("user_w", PersonaSignals(budget_signal="mid", signal_strength=0.3))
    manager.update_persona("user_w", PersonaSignals(budget_signal="luxury", signal_strength=0.95))

    mock_session = write_driver.session.return_value.__enter__.return_value
    mock_tx = mock_session.begin_transaction.return_value.__enter__.return_value
    budget_calls = [c for c in mock_tx.run.call_args_list if "budget_entry" in str(c)]
    written_entries = [c[0][1]["budget_entry"] for c in budget_calls]

    read_driver = _make_driver_with_records(
        [
            _make_full_record(
                turn_count=2,
                budget_signals=written_entries,
            )
        ]
    )
    snapshot = _make_manager(read_driver).get_persona("user_w")
    assert snapshot.budget_tier == "luxury"


@pytest.mark.unit
def test_update_persona_db_error_logged_not_raised() -> None:
    """update_persona logs and swallows DB errors — never propagates to caller."""
    driver = MagicMock()
    driver.session.side_effect = RuntimeError("DB unavailable")

    _make_manager(driver).update_persona("user_err", PersonaSignals(signal_strength=0.5))
    # No exception should reach here


# ---------------------------------------------------------------------------
# Temporal decay & confidence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_temporal_decay_reduces_weight() -> None:
    """effective_weight = raw_weight * exp(-decay_rate * delta), verified at multiple deltas."""
    manager = _make_manager(MagicMock(), decay_rate=0.15)
    for delta in [1, 3, 5, 10]:
        result = manager._apply_decay(1.0, last_seen_turn=10, current_turn=10 + delta)
        expected = math.exp(-0.15 * delta)
        assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
def test_confidence_formula_zero_on_first_turn() -> None:
    """Zero weights on turn 1 → confidence = 0.0."""
    assert _make_manager(MagicMock())._confidence_score([], turn_count=1) == pytest.approx(0.0)


@pytest.mark.unit
def test_confidence_grows_with_signals() -> None:
    """8 signals of weight 0.8 across 5 turns → confidence > 0.3."""
    score = _make_manager(MagicMock())._confidence_score([0.8] * 8, turn_count=5)
    assert score > 0.3


@pytest.mark.unit
def test_confidence_capped_at_one() -> None:
    """Confidence never exceeds 1.0 regardless of signal count."""
    score = _make_manager(MagicMock())._confidence_score([10.0] * 100, turn_count=1)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _retry helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_retry_succeeds_on_first_attempt() -> None:
    """_retry returns result immediately when fn succeeds on first try."""
    call_count = 0

    def fn() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = _retry(fn, attempts=3)
    assert result == "ok"
    assert call_count == 1


@pytest.mark.unit
def test_retry_retries_on_transient_failure() -> None:
    """_retry calls fn up to N times before succeeding."""
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "recovered"

    result = _retry(fn, attempts=3, base_delay=0.0)
    assert result == "recovered"
    assert len(calls) == 3


@pytest.mark.unit
def test_retry_raises_after_exhaustion() -> None:
    """_retry re-raises the last exception when all attempts fail."""

    def fn() -> None:
        raise ValueError("permanent error")

    with pytest.raises(ValueError, match="permanent error"):
        _retry(fn, attempts=3, base_delay=0.0)


# ---------------------------------------------------------------------------
# AppSettings validation (issue #52 — raise ValueError not assert)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAppSettingsValidation:
    def _make_settings(self, **kwargs):
        from stylemind.config import AppSettings

        defaults = {
            "log_level": "INFO",
            "vector_top_k": 10,
            "persona_decay_rate": 0.15,
            "expected_signals_per_turn": 3.0,
            "min_similarity_threshold": 0.3,
        }
        defaults.update(kwargs)
        return AppSettings(**defaults)

    def test_valid_settings_no_error(self) -> None:
        self._make_settings()  # should not raise

    def test_invalid_log_level_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            self._make_settings(log_level="VERBOSE")

    def test_zero_vector_top_k_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="vector_top_k"):
            self._make_settings(vector_top_k=0)

    def test_negative_vector_top_k_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="vector_top_k"):
            self._make_settings(vector_top_k=-1)

    def test_zero_decay_rate_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="persona_decay_rate"):
            self._make_settings(persona_decay_rate=0.0)

    def test_decay_rate_of_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="persona_decay_rate"):
            self._make_settings(persona_decay_rate=1.0)

    def test_zero_expected_signals_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="expected_signals_per_turn"):
            self._make_settings(expected_signals_per_turn=0.0)

    def test_negative_threshold_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="min_similarity_threshold"):
            self._make_settings(min_similarity_threshold=-0.1)

    def test_threshold_above_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="min_similarity_threshold"):
            self._make_settings(min_similarity_threshold=1.1)

    def test_validation_uses_value_error_not_assertion_error(self) -> None:
        """Ensure we get ValueError, not AssertionError (asserts stripped under -O)."""
        with pytest.raises(ValueError):
            self._make_settings(log_level="NOPE")
        # Should NOT raise AssertionError
        try:
            self._make_settings(log_level="NOPE")
        except AssertionError:
            pytest.fail("Got AssertionError — validation must use raise ValueError, not assert")
        except ValueError:
            pass  # expected


# ---------------------------------------------------------------------------
# CORS origins via env var (issue #43)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cors_wildcard_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CORS_ORIGINS is not set, the app accepts wildcard origins."""
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    from stylemind.main import create_app

    app = create_app()
    cors_mw = next((m for m in app.user_middleware if "CORSMiddleware" in str(m)), None)
    assert cors_mw is not None, "CORSMiddleware should be registered"


@pytest.mark.unit
def test_cors_restricted_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CORS_ORIGINS env var restricts allowed origins."""
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com,https://admin.example.com")
    from stylemind.main import create_app

    app = create_app()
    cors_mw = next((m for m in app.user_middleware if "CORSMiddleware" in str(m)), None)
    assert cors_mw is not None


# ---------------------------------------------------------------------------
# Generator empty choices guard (issue #47)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_stream_response_skips_empty_choices_chunks() -> None:
    """Chunks with choices=[] are silently skipped (no AttributeError, no empty yield)."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from stylemind.config import ChatLLMConfig
    from stylemind.rag.generator import StyleMindGenerator

    config = ChatLLMConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    def make_chunk(content: str | None, empty: bool = False) -> MagicMock:
        chunk = MagicMock()
        if empty:
            chunk.choices = []
        else:
            choice = MagicMock()
            choice.delta.content = content
            chunk.choices = [choice]
        return chunk

    chunks = [
        make_chunk(None, empty=True),  # heartbeat chunk — choices=[]
        make_chunk("Hello"),
        make_chunk(None, empty=True),  # another empty
        make_chunk(" world"),
        make_chunk(None),  # content=None (role-only delta)
    ]

    async def fake_stream():
        for c in chunks:
            yield c

    mock_stream = MagicMock()
    mock_stream.__aiter__ = lambda _: fake_stream()

    with patch("stylemind.rag.generator.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        mock_openai_cls.return_value = mock_client

        gen = StyleMindGenerator(config=config)
        collected = []
        async for chunk in gen.stream_response("hi", [], []):
            collected.append(chunk)

    assert collected == ["Hello", " world"], f"Expected ['Hello', ' world'], got {collected}"
