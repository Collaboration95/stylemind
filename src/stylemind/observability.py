from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Langfuse import with graceful degradation
# ---------------------------------------------------------------------------

_langfuse_available = False


def _noop_observe(func=None, *, name: str | None = None, **kwargs):  # type: ignore[misc]
    """Identity decorator fallback when Langfuse is unavailable."""

    def decorator(fn):  # type: ignore[misc]
        return fn

    if func is not None:
        return func
    return decorator


try:
    from langfuse import Langfuse as _Langfuse
    from langfuse import observe  # type: ignore[assignment]

    _langfuse_available = True
except ImportError:
    _Langfuse = None  # type: ignore[assignment,misc]
    observe = _noop_observe  # type: ignore[assignment]


# langfuse_context: used to update trace metadata (session_id, user_id) inside an @observe span.
# In Langfuse v4 this is exposed via the module-level `get_client()` singleton.
# We expose a thin wrapper below so callers import from this module (not directly from langfuse).
try:
    from langfuse import get_client as _get_langfuse_client

    class _LangfuseContext:
        """Thin adapter around the Langfuse v4 get_client() singleton."""

        def update_current_trace(self, *, session_id: str | None = None, user_id: str | None = None) -> None:
            """Update the current trace attributes (no-op if not inside an @observe span)."""
            try:
                # In v4, session/user context is set via propagate_attributes or
                # directly on the current OTel span attributes.
                # The recommended approach for updating the ongoing trace is through
                # the span's OTEL attributes that Langfuse maps to its trace.
                from langfuse import propagate_attributes

                # propagate_attributes is a context manager; outside a trace context
                # it is effectively a no-op (Langfuse will associate these baggage
                # attributes with the current trace when inside an @observe span).
                if session_id or user_id:
                    ctx = propagate_attributes(session_id=session_id, user_id=user_id, as_baggage=True)
                    ctx.__enter__()
                    # We intentionally do NOT __exit__ here so the baggage stays
                    # active for the lifetime of this call stack.
            except Exception as exc:
                logger.debug("langfuse_context update_current_trace no-op error=%s", exc)

        def score_current_observation(self, *, name: str, value: float) -> None:
            """Score the current observation/span (no-op if not inside an @observe span)."""
            try:
                client = _get_langfuse_client()
                client.score_current_span(name=name, value=value)
            except Exception as exc:
                logger.debug("langfuse_context score_current_observation no-op error=%s", exc)

    langfuse_context: _LangfuseContext | None = _LangfuseContext()

except ImportError:
    langfuse_context = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_langfuse: Langfuse | None = None
_lock = threading.Lock()


def init_langfuse(config) -> Langfuse | None:  # type: ignore[type-arg]
    """Initialise the Langfuse client from config.

    Returns the client on success, or None for graceful degradation when:
    - Langfuse package is not importable
    - public_key / secret_key are empty
    - Connection fails for any reason

    Args:
        config: LangfuseConfig dataclass with public_key, secret_key, host.

    Returns:
        Langfuse instance, or None.
    """
    global _langfuse

    if not _langfuse_available or _Langfuse is None:
        logger.warning("observability langfuse_available=false package_not_installed")
        return None

    if not config.public_key or not config.secret_key:
        logger.info("observability langfuse_enabled=false reason=missing_keys")
        return None

    try:
        lf = _Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
        )
        with _lock:
            _langfuse = lf
        logger.info("observability langfuse_enabled=true host=%s", config.host)
        return lf
    except Exception as exc:
        logger.warning("observability langfuse_init_failed error=%s", exc)
        return None


def get_langfuse() -> Langfuse | None:
    """Thread-safe accessor for the Langfuse singleton.

    Returns:
        The initialised Langfuse instance, or None if not yet initialised or
        unavailable.
    """
    with _lock:
        return _langfuse


def score_persona_confidence(user_id: str, confidence: float, session_id: str) -> None:
    """Log a persona_confidence custom score via Langfuse.

    This is a no-op when Langfuse is not initialised or unavailable.

    Args:
        user_id: The user identifier.
        confidence: Confidence score in [0.0, 1.0].
        session_id: Langfuse session_id (typically same as user_id).
    """
    lf = get_langfuse()
    if lf is None:
        return

    try:
        if langfuse_context is not None:
            langfuse_context.score_current_observation(name="persona_confidence", value=confidence)
        logger.debug("observability persona_confidence scored user_id=%s confidence=%.3f", user_id, confidence)
    except Exception as exc:
        # Graceful degradation: scoring failure must never crash the app.
        logger.warning("observability score_persona_confidence failed user_id=%s error=%s", user_id, exc)


def _reset_langfuse() -> None:
    """Reset the singleton — used in tests only."""
    global _langfuse
    with _lock:
        _langfuse = None
