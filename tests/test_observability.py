from __future__ import annotations

import pytest

from stylemind.config import LangfuseConfig
from stylemind.observability import (
    _reset_langfuse,
    get_langfuse,
    init_langfuse,
    observe,
    score_persona_confidence,
)


@pytest.fixture(autouse=True)
def reset_langfuse_singleton():
    """Ensure a clean Langfuse singleton before and after each test."""
    _reset_langfuse()
    yield
    _reset_langfuse()


# ---------------------------------------------------------------------------
# init_langfuse
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_langfuse_empty_public_key_returns_none():
    """init_langfuse should return None when public_key is empty (graceful degradation)."""
    config = LangfuseConfig(public_key="", secret_key="some-secret", host="http://localhost:3000")
    result = init_langfuse(config)
    assert result is None


@pytest.mark.unit
def test_init_langfuse_empty_secret_key_returns_none():
    """init_langfuse should return None when secret_key is empty."""
    config = LangfuseConfig(public_key="some-public", secret_key="", host="http://localhost:3000")
    result = init_langfuse(config)
    assert result is None


@pytest.mark.unit
def test_init_langfuse_both_keys_empty_returns_none():
    """init_langfuse should return None when both keys are empty."""
    config = LangfuseConfig(public_key="", secret_key="", host="http://localhost:3000")
    result = init_langfuse(config)
    assert result is None


# ---------------------------------------------------------------------------
# get_langfuse
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_langfuse_returns_none_before_initialization():
    """get_langfuse() should return None when init_langfuse has not been called."""
    result = get_langfuse()
    assert result is None


@pytest.mark.unit
def test_get_langfuse_returns_none_after_failed_init():
    """get_langfuse() should return None after a failed (empty keys) init attempt."""
    config = LangfuseConfig(public_key="", secret_key="", host="http://localhost:3000")
    init_langfuse(config)
    assert get_langfuse() is None


# ---------------------------------------------------------------------------
# score_persona_confidence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_score_persona_confidence_noop_when_not_initialized():
    """score_persona_confidence should be a no-op (no error) when Langfuse is not initialized."""
    # Langfuse singleton is None (reset by fixture)
    assert get_langfuse() is None
    # Must not raise
    score_persona_confidence(user_id="user-123", confidence=0.75, session_id="user-123")


@pytest.mark.unit
def test_score_persona_confidence_accepts_boundary_values():
    """score_persona_confidence should handle boundary confidence values without error."""
    score_persona_confidence(user_id="u1", confidence=0.0, session_id="u1")
    score_persona_confidence(user_id="u1", confidence=1.0, session_id="u1")


# ---------------------------------------------------------------------------
# observe decorator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_observe_decorator_is_identity_when_langfuse_unavailable():
    """@observe should act as an identity wrapper: the decorated function still executes."""

    @observe(name="test_span")
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(21)
    assert result == 42


@pytest.mark.unit
def test_observe_decorator_preserves_return_value():
    """@observe should not alter the function's return value."""

    @observe(name="some_span")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert greet("StyleMind") == "Hello, StyleMind!"


@pytest.mark.unit
def test_observe_decorator_on_method():
    """@observe should work transparently on instance methods."""

    class FakeRetriever:
        @observe(name="retrieve")
        def retrieve(self, query: str) -> list[str]:
            return [query]

    retriever = FakeRetriever()
    result = retriever.retrieve("linen shirt")
    assert result == ["linen shirt"]


@pytest.mark.unit
def test_observe_decorator_propagates_exceptions():
    """@observe should not swallow exceptions from the decorated function."""

    @observe(name="failing_span")
    def broken() -> None:
        raise ValueError("intentional error")

    with pytest.raises(ValueError, match="intentional error"):
        broken()
