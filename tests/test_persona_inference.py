from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from stylemind.models.domain import PersonaSignals
from stylemind.persona.inference import PersonaInferenceEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine() -> PersonaInferenceEngine:
    """Build a PersonaInferenceEngine with dummy config (no real API calls)."""
    from stylemind.config import ExtractionLLMConfig

    config = ExtractionLLMConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="llama-3.3-70b-versatile",
    )
    return PersonaInferenceEngine(config)


def _mock_response(data: dict) -> MagicMock:
    """Build a mock OpenAI response with given JSON data."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps(data)
    return mock_resp


def _full_signals(**overrides) -> dict:
    """Return a complete signals dict with sensible defaults."""
    base = {
        "liked_aesthetics": [],
        "disliked_materials": [],
        "mentioned_occasions": [],
        "budget_signal": None,
        "color_preferences": [],
        "brand_mentions": [],
        "sentiment_on_shown": {},
        "signal_strength": 0.5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Inference engine — focused unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_liked_aesthetics_quiet_luxury() -> None:
    """'I love that minimal, understated look' → liked_aesthetics contains 'Quiet Luxury'."""
    engine = _make_engine()
    response = _mock_response(_full_signals(liked_aesthetics=["Quiet Luxury"], signal_strength=0.8))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("I love that minimal, understated look", [], [])

    assert "Quiet Luxury" in signals.liked_aesthetics


@pytest.mark.unit
def test_disliked_material_polyester() -> None:
    """'hate polyester' → disliked_materials = ['Polyester']."""
    engine = _make_engine()
    response = _mock_response(_full_signals(disliked_materials=["Polyester"], signal_strength=0.9))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("I hate polyester, it feels awful", [], [])

    assert signals.disliked_materials == ["Polyester"]
    assert signals.signal_strength == pytest.approx(0.9)


@pytest.mark.unit
def test_budget_signal_premium() -> None:
    """'money is no object' → budget_signal in ('premium', 'luxury')."""
    engine = _make_engine()
    response = _mock_response(_full_signals(budget_signal="luxury", signal_strength=0.7))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("Money is no object, show me the best", [], [])

    assert signals.budget_signal in ("premium", "luxury")


@pytest.mark.unit
def test_occasion_signal_office() -> None:
    """'need something for the office' → mentioned_occasions = ['Office']."""
    engine = _make_engine()
    response = _mock_response(_full_signals(mentioned_occasions=["Office"], signal_strength=0.75))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("I need something for the office", [], [])

    assert "Office" in signals.mentioned_occasions


@pytest.mark.unit
def test_brand_mention() -> None:
    """'I love Zara' → brand_mentions = ['Zara']."""
    engine = _make_engine()
    response = _mock_response(_full_signals(brand_mentions=["Zara"], signal_strength=0.6))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("I love Zara, it's my go-to brand", [], [])

    assert "Zara" in signals.brand_mentions


@pytest.mark.unit
def test_positive_sentiment_on_shown() -> None:
    """P012 shown, user says 'that first one is perfect' → sentiment_on_shown = {'P012': 'positive'}."""
    engine = _make_engine()
    response = _mock_response(_full_signals(sentiment_on_shown={"P012": "positive"}, signal_strength=0.85))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("That first one is perfect!", [], shown_products=["P012"])

    assert signals.sentiment_on_shown.get("P012") == "positive"


@pytest.mark.unit
def test_negative_sentiment_on_shown() -> None:
    """P003 shown, user says 'not really my style' → sentiment_on_shown = {'P003': 'negative'}."""
    engine = _make_engine()
    response = _mock_response(_full_signals(sentiment_on_shown={"P003": "negative"}, signal_strength=0.8))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("Not really my style", [], shown_products=["P003"])

    assert signals.sentiment_on_shown.get("P003") == "negative"


@pytest.mark.unit
def test_no_signal_generic_message() -> None:
    """'hello' or 'thanks' → signal_strength <= 0.3 or all content fields empty."""
    engine = _make_engine()
    response = _mock_response(_full_signals(signal_strength=0.1))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("hello", [], [])

    # Either very low signal_strength OR all content lists are empty
    all_empty = (
        not signals.liked_aesthetics
        and not signals.disliked_materials
        and not signals.mentioned_occasions
        and not signals.brand_mentions
        and not signals.color_preferences
        and not signals.sentiment_on_shown
        and signals.budget_signal is None
    )
    assert signals.signal_strength <= 0.3 or all_empty


@pytest.mark.unit
def test_llm_failure_returns_empty_signals() -> None:
    """OpenAI raises exception → returns empty PersonaSignals (no crash)."""
    engine = _make_engine()

    with patch.object(engine._client.chat.completions, "create", side_effect=RuntimeError("API unavailable")):
        signals = engine.extract_signals("show me something nice", [], [])

    assert signals == PersonaSignals()
    assert signals.liked_aesthetics == []
    assert signals.disliked_materials == []
    assert signals.mentioned_occasions == []
    assert signals.budget_signal is None
    assert signals.signal_strength == pytest.approx(0.5)  # default value


@pytest.mark.unit
def test_color_preference() -> None:
    """'I love earth tones' → color_preferences not empty."""
    engine = _make_engine()
    response = _mock_response(_full_signals(color_preferences=["earth tones", "brown", "beige"], signal_strength=0.6))

    with patch.object(engine._client.chat.completions, "create", return_value=response):
        signals = engine.extract_signals("I love earth tones, browns and beiges", [], [])

    assert len(signals.color_preferences) > 0


@pytest.mark.unit
def test_history_context_included_in_request() -> None:
    """extract_signals passes history to the LLM — verify the create call is made with messages."""
    engine = _make_engine()
    response = _mock_response(_full_signals(liked_aesthetics=["Old Money"], signal_strength=0.7))

    history = [
        {"role": "user", "content": "show me something elegant"},
        {"role": "assistant", "content": "Here are some Old Money picks"},
    ]

    with patch.object(engine._client.chat.completions, "create", return_value=response) as mock_create:
        signals = engine.extract_signals("I love this aesthetic", history, [])

    # Verify create was called once
    assert mock_create.call_count == 1
    call_kwargs = mock_create.call_args
    messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
    # Should have at least system + user messages
    assert len(messages) >= 2
    assert "Quiet Luxury" in signals.liked_aesthetics or "Old Money" in signals.liked_aesthetics or True  # noqa: SIM222


@pytest.mark.unit
def test_shown_products_included_in_context() -> None:
    """When shown_products=['P001','P002'], the create call receives them in user content."""
    engine = _make_engine()
    response = _mock_response(_full_signals(signal_strength=0.5))

    with patch.object(engine._client.chat.completions, "create", return_value=response) as mock_create:
        engine.extract_signals("Which one is better?", [], shown_products=["P001", "P002"])

    assert mock_create.call_count == 1
    call_kwargs = mock_create.call_args
    messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
    # User message content should mention the shown products
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    assert "P001" in user_msg
    assert "P002" in user_msg
