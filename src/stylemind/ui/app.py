"""StyleMind Streamlit UI — Direction A · Modern Boutique · Notebook · Compact.

Run:  uv run streamlit run src/stylemind/ui/app.py --server.port 8000

Architecture: this Streamlit process talks to the FastAPI backend on port 8001
via httpx (streaming SSE for /chat, plain GET for /persona, /outfit). The
FastAPI server is started in a background daemon thread on first load (same
pattern as the Rich CLI in __main__.py).

See docs/SPEC.md for the visual contract this code implements.
"""

from __future__ import annotations

import contextlib
import logging
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

import random  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from difflib import get_close_matches  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from stylemind.ui.components import (  # noqa: E402
    BASE_CSS,
    inject_css,
    render_brand_block,
    render_citations,
    render_debug_signals,
    render_explain_table,
    render_help_panel,
    render_outfit_card,
    render_persona_panel,
    render_sidebar_persona,
    render_signals_strip,
    render_system_note,
    render_top_bar,
    render_turn_marker,
    render_user_bubble,
    render_welcome,
)
from stylemind.ui.sse_client import fetch_outfit, fetch_persona, stream_chat  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Backend server management

_API_PORT = int(os.getenv("STYLEMIND_API_PORT", "8001"))
BASE_URL = os.getenv("STYLEMIND_API", f"http://localhost:{_API_PORT}")

_server_lock = threading.Lock()
_server_started = False


def _quiet_logging() -> None:
    os.environ["LOG_LEVEL"] = "ERROR"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger().setLevel(logging.ERROR)
    for name in (
        "stylemind",
        "langfuse",
        "neo4j",
        "sentence_transformers",
        "httpx",
        "httpcore",
        "uvicorn",
        "transformers",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _server_is_up(port: int) -> bool:
    try:
        resp = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
        return resp.status_code == 200
    except httpx.ConnectError, httpx.TimeoutException, httpx.ReadError:
        return False


def _ensure_backend() -> None:
    global _server_started  # noqa: PLW0603
    if _server_started or _server_is_up(_API_PORT):
        _server_started = True
        return

    with _server_lock:
        if _server_started or _server_is_up(_API_PORT):
            _server_started = True
            return

        _quiet_logging()

        import uvicorn
        from dotenv import load_dotenv

        load_dotenv()

        def _run() -> None:
            uvicorn.run(
                "stylemind.main:create_app",
                factory=True,
                host="0.0.0.0",
                port=_API_PORT,
                log_level="error",
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if _server_is_up(_API_PORT):
                _server_started = True
                return
            time.sleep(0.5)
        st.error("FastAPI backend did not start within 60s. Check Neo4j and API keys.")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Config

STARTERS_POOL = [
    "I need something for a date night",
    "Show me minimal summer outfits under 5k",
    "I love the quiet luxury aesthetic",
    "What's good for an office look?",
    "I want casual streetwear vibes",
    "Show me something in earthy tones",
    "I need a wedding guest outfit",
    "What goes well with linen pants?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Page setup

st.set_page_config(
    page_title="StyleMind",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css(BASE_CSS)


# ─────────────────────────────────────────────────────────────────────────────
# Ensure backend is running

_ensure_backend()


# ─────────────────────────────────────────────────────────────────────────────
# Session state


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("user_id", f"usr-{uuid.uuid4().hex[:6]}")
    ss.setdefault("messages", [])
    ss.setdefault("turn_count", 0)
    ss.setdefault("sources_log", [])
    ss.setdefault("signals_log", [])
    ss.setdefault("explain_log", [])
    ss.setdefault("persona", _empty_persona())
    ss.setdefault("explain_on", False)
    ss.setdefault("starters", random.sample(STARTERS_POOL, 3))
    ss.setdefault("pending_message", None)
    ss.setdefault("product_catalog", [])
    ss.setdefault("command_result", None)


def _empty_persona() -> dict[str, Any]:
    return {
        "preferred_aesthetics": [],
        "disliked_materials": [],
        "disliked_products": [],
        "budget_tier": None,
        "top_occasions": [],
        "color_preferences": [],
        "confidence_score": 0.0,
    }


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Product catalog (for /outfit fuzzy matching)


def _load_product_catalog() -> None:
    if st.session_state.product_catalog:
        return
    with contextlib.suppress(Exception), httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{BASE_URL}/products/names")
        if resp.status_code == 200:
            st.session_state.product_catalog = resp.json()


_load_product_catalog()


def _fuzzy_match_product(query: str) -> str | None:
    catalog = st.session_state.product_catalog
    q = query.lower()
    for p in catalog:
        if p["name"].lower() == q:
            return p["product_id"]
    for p in catalog:
        if q in p["name"].lower():
            return p["product_id"]
    close = get_close_matches(q, [p["name"].lower() for p in catalog], n=1, cutoff=0.5)
    if close:
        matched = next((p for p in catalog if p["name"].lower() == close[0]), None)
        if matched:
            return matched["product_id"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Slash command handler


def _handle_slash_command(cmd: str) -> bool:
    """Process slash commands. Returns True if handled."""
    low = cmd.strip().lower()

    if low == "/help":
        st.session_state.command_result = {"type": "help"}
        return True

    if low == "/persona":
        with contextlib.suppress(Exception):
            st.session_state.persona = fetch_persona(BASE_URL, st.session_state.user_id)
        st.session_state.command_result = {"type": "persona", "data": st.session_state.persona}
        return True

    if low == "/debug-dev":
        st.session_state.command_result = {"type": "debug", "data": st.session_state.signals_log}
        return True

    if low == "/clear":
        st.session_state.messages = []
        st.session_state.turn_count = 0
        st.session_state.sources_log = []
        st.session_state.signals_log = []
        st.session_state.explain_log = []
        st.session_state.persona = _empty_persona()
        st.session_state.starters = random.sample(STARTERS_POOL, 3)
        st.session_state.command_result = None
        st.rerun()
        return True

    if low.startswith("/outfit"):
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            catalog = st.session_state.product_catalog
            examples = ""
            if catalog:
                sample = random.sample(catalog, min(3, len(catalog)))
                examples = " Try: " + ", ".join(f'"{p["name"]}"' for p in sample)
            st.session_state.command_result = {"type": "note", "text": f"Usage: /outfit <product name>.{examples}"}
            return True
        query = parts[1].strip()
        product_id = _fuzzy_match_product(query)
        if not product_id:
            catalog = st.session_state.product_catalog
            close = get_close_matches(
                query.lower(),
                [p["name"].lower() for p in catalog],
                n=3,
                cutoff=0.4,
            )
            suggestions = ""
            if close:
                names = [next(p["name"] for p in catalog if p["name"].lower() == c) for c in close]
                suggestions = " Did you mean: " + ", ".join(f'"{n}"' for n in names) + "?"
            st.session_state.command_result = {"type": "note", "text": f'No product matching "{query}".{suggestions}'}
            return True
        try:
            outfit = fetch_outfit(BASE_URL, product_id, st.session_state.user_id)
            st.session_state.command_result = {"type": "outfit", "data": outfit}
        except Exception:
            st.session_state.command_result = {"type": "note", "text": f'Could not build outfit for "{query}".'}
        return True

    return False


def _render_command_result() -> None:
    result = st.session_state.command_result
    if not result:
        return
    t = result["type"]
    if t == "help":
        render_help_panel()
    elif t == "persona":
        render_persona_panel(result["data"])
    elif t == "debug":
        render_debug_signals(result["data"])
    elif t == "outfit":
        render_outfit_card(result["data"])
    elif t == "note":
        render_system_note(result["text"])


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — the stylist's notebook

with st.sidebar:
    render_brand_block()
    render_sidebar_persona(
        persona=st.session_state.persona,
        turn=st.session_state.turn_count,
        user_id=st.session_state.user_id,
    )

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("clear", key="clear_btn", use_container_width=True):
            st.session_state.messages = []
            st.session_state.turn_count = 0
            st.session_state.sources_log = []
            st.session_state.signals_log = []
            st.session_state.explain_log = []
            st.session_state.persona = _empty_persona()
            st.session_state.starters = random.sample(STARTERS_POOL, 3)
            st.rerun()
    with cols[1]:
        st.session_state.explain_on = st.toggle("explain", value=st.session_state.explain_on, key="explain_toggle")


# ─────────────────────────────────────────────────────────────────────────────
# Main pane

render_top_bar(turn_count=st.session_state.turn_count)

# Empty state — welcome + starters
if not st.session_state.messages:
    render_welcome()
    cols = st.columns(3, gap="small")
    for i, starter in enumerate(st.session_state.starters):
        with cols[i]:
            if st.button(
                f"0{i + 1}   {starter}   →",
                key=f"starter_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_message = starter
                st.rerun()

# Render existing transcript
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        turn_idx = (i // 2) + 1
        render_turn_marker(turn_idx)
        render_user_bubble(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="bq-msg-asst-byline">StyleMind</div><div class="bq-msg-asst-text">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            turn = i // 2
            if turn < len(st.session_state.sources_log):
                sources = st.session_state.sources_log[turn]
                if sources:
                    render_citations(sources)
            if st.session_state.explain_on and turn < len(st.session_state.explain_log):
                explain = st.session_state.explain_log[turn]
                if explain:
                    render_explain_table(explain)
            if msg.get("outfit"):
                render_outfit_card(msg["outfit"])
            if turn < len(st.session_state.signals_log):
                signals = st.session_state.signals_log[turn]
                if signals:
                    render_signals_strip(signals)


# ─────────────────────────────────────────────────────────────────────────────
# Command output (persists across reruns until next input)

_render_command_result()

# ─────────────────────────────────────────────────────────────────────────────
# Composer

placeholder = "What are we dressing for?" if not st.session_state.messages else "Anything in earth tones to swap in?"
user_input = st.chat_input(placeholder=placeholder)

message = st.session_state.pending_message or user_input
st.session_state.pending_message = None

if message:
    if message.startswith("/"):
        if _handle_slash_command(message):
            st.rerun()
    else:
        st.session_state.command_result = None

        st.session_state.messages.append({"role": "user", "content": message})
        turn_idx = (len(st.session_state.messages) // 2) + 1
        render_turn_marker(turn_idx)
        render_user_bubble(message)

        with st.chat_message("assistant"):
            st.markdown(
                '<div class="bq-msg-asst-byline">StyleMind</div>',
                unsafe_allow_html=True,
            )

            captured: dict[str, Any] = {"sources": [], "signals": {}, "explain": []}

            def text_gen():
                yield from stream_chat(
                    base_url=BASE_URL,
                    user_id=st.session_state.user_id,
                    message=message,
                    history=st.session_state.messages[:-1],
                    explain=st.session_state.explain_on,
                    captured=captured,
                )

            text_placeholder = st.empty()
            full_text = ""
            for chunk in text_gen():
                full_text += chunk
                text_placeholder.markdown(
                    f'<div class="bq-msg-asst-text">{full_text}<span class="bq-cursor">▍</span></div>',
                    unsafe_allow_html=True,
                )
            text_placeholder.markdown(
                f'<div class="bq-msg-asst-text">{full_text}</div>',
                unsafe_allow_html=True,
            )

            sources = captured["sources"]
            signals = captured["signals"]
            explain = captured["explain"]

            if sources:
                render_citations(sources)
            if st.session_state.explain_on and explain:
                render_explain_table(explain)
            if signals:
                render_signals_strip(signals)

        st.session_state.messages.append({"role": "assistant", "content": full_text})
        st.session_state.sources_log.append(sources)
        st.session_state.signals_log.append(signals)
        st.session_state.explain_log.append(explain)
        st.session_state.turn_count += 1

        with contextlib.suppress(Exception):
            st.session_state.persona = fetch_persona(BASE_URL, st.session_state.user_id)

        st.rerun()
