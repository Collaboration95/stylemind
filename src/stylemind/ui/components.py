"""Reusable render helpers for the StyleMind Streamlit UI.

Each helper renders a chunk of HTML via st.markdown(unsafe_allow_html=True).
The CSS contract lives in BASE_CSS and matches SPEC.md §1–§4.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CSS

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: oklch(0.97 0.008 75);
  --paper: oklch(0.99 0.005 80);
  --ink: oklch(0.22 0.015 60);
  --ink-soft: oklch(0.35 0.014 60);
  --ink-faint: oklch(0.50 0.012 60);
  --rule: oklch(0.88 0.012 70);
  --rule-soft: oklch(0.92 0.008 70);
  --navy: oklch(0.32 0.06 260);
  --navy-soft: oklch(0.92 0.018 260);
  --gold: #d4a73c;
  --gold-soft: oklch(0.94 0.04 80);
  --danger: oklch(0.55 0.14 30);
}

/* App-wide overrides */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--ink);
  font-family: "Inter", system-ui, sans-serif;
  font-size: 14px;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"] { padding-top: 0; }
.block-container { max-width: 960px; padding-top: 14px; padding-bottom: 120px; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--paper) !important;
  border-right: 1px solid var(--rule);
  width: 300px !important; min-width: 300px !important;
}
[data-testid="stSidebar"] > div { padding-top: 0; }

/* Brand block */
.bq-brand { padding: 18px 20px 14px; border-bottom: 1px solid var(--rule-soft);
  display: flex; align-items: center; gap: 10px; }
.bq-brand-mark { width: 28px; height: 28px; border-radius: 8px; background: var(--navy);
  color: var(--gold); display: grid; place-items: center;
  font-family: "Fraunces", serif; font-weight: 600; font-size: 16px; }
.bq-brand-name { font-family: "Fraunces", serif; font-weight: 500;
  font-size: 19px; letter-spacing: -0.01em; color: var(--ink); }
.bq-brand-name em { font-style: italic; font-weight: 400; color: var(--gold); }
.bq-brand-sub { font-size: 10.5px; color: var(--ink-faint);
  letter-spacing: 0.08em; text-transform: uppercase; margin-top: 1px;
  font-family: "JetBrains Mono", monospace; }

/* Section labels */
.bq-section-label { font-family: "Fraunces", serif; font-size: 11px; font-weight: 500;
  color: var(--ink-faint); letter-spacing: 0.14em; text-transform: uppercase;
  display: flex; align-items: center; gap: 8px; margin-top: 22px; margin-bottom: 8px; }
.bq-section-label::after { content: ""; flex: 1; height: 1px; background: var(--rule); }

/* Confidence */
.bq-conf-row { display: flex; justify-content: space-between; align-items: baseline; }
.bq-conf-label { font-family: "Fraunces", serif; font-size: 14px; font-style: italic; color: var(--ink); }
.bq-conf-pct { font-family: "JetBrains Mono", monospace; font-size: 12px; color: var(--ink-faint);
  font-feature-settings: "tnum"; }
.bq-conf-bar { height: 4px; background: var(--rule-soft); border-radius: 2px;
  overflow: hidden; position: relative; margin: 8px 0 4px; }
.bq-conf-fill { position: absolute; inset: 0 auto 0 0;
  background: linear-gradient(90deg, var(--gold), oklch(0.78 0.14 70));
  border-radius: 2px; transition: width .4s ease; }
.bq-conf-ticks { display: flex; justify-content: space-between;
  font-family: "JetBrains Mono", monospace; font-size: 9.5px; color: var(--ink-faint);
  letter-spacing: 0.04em; text-transform: uppercase; }

/* Tags */
.bq-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.bq-tag { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px;
  background: var(--paper); border: 0.5px solid var(--rule); border-radius: 999px;
  font-size: 12.5px; color: var(--ink); line-height: 1.4; }
.bq-tag.gold { background: var(--gold-soft); border-color: oklch(0.88 0.07 78); color: oklch(0.38 0.08 65); }
.bq-tag.navy { background: var(--navy-soft); border-color: oklch(0.85 0.04 260); color: var(--navy); }
.bq-tag.muted { color: var(--ink-faint);
  text-decoration: line-through; text-decoration-color: oklch(0.55 0.13 30 / 0.5); }
.bq-tag-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: 0.6; }

/* Budget */
.bq-budget { display: flex; align-items: baseline; gap: 8px; }
.bq-budget-tier { font-family: "Fraunces", serif; font-size: 18px; font-weight: 500; color: var(--ink); }
.bq-budget-scale { display: flex; gap: 3px; margin-left: auto; }
.bq-budget-pip { width: 18px; height: 4px; border-radius: 1px; background: var(--rule); }
.bq-budget-pip.on { background: var(--gold); }

.bq-empty-row { font-size: 12px; color: var(--ink-faint);
  font-style: italic; font-family: "Fraunces", serif; }

/* Top bar */
.bq-main-bar { padding: 18px 0 14px; margin-bottom: 28px;
  border-bottom: 1px solid var(--rule-soft);
  display: flex; justify-content: space-between; align-items: center; }
.bq-main-title { font-family: "Fraunces", serif; font-size: 20px; font-weight: 500; }
.bq-main-title em { font-style: italic; color: var(--gold); }
.bq-main-meta { font-family: "JetBrains Mono", monospace; font-size: 12px;
  color: var(--ink-faint); letter-spacing: 0.04em; text-transform: uppercase; }

/* Welcome */
.bq-welcome { padding: 40px 0 20px; }
.bq-welcome-eyebrow { font-family: "JetBrains Mono", monospace; font-size: 13px;
  color: var(--gold); letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 18px; }
.bq-welcome-h1 { font-family: "Fraunces", serif; font-size: 54px; font-weight: 400;
  line-height: 1.08; letter-spacing: -0.02em; margin: 0 0 16px; color: var(--ink); }
.bq-welcome-h1 em { font-style: italic; color: var(--gold); }
.bq-welcome-sub { font-size: 18px; color: var(--ink-soft);
  max-width: 540px; margin: 0 0 36px; line-height: 1.55; }

/* Starter buttons (Streamlit-native, restyled) */
.stButton > button {
  background: var(--paper); border: 0.5px solid var(--rule);
  border-radius: 12px; color: var(--ink); font: 400 16px Inter, sans-serif;
  padding: 16px 20px; text-align: left; transition: all .15s ease;
}
.stButton > button:hover {
  border-color: var(--gold); background: oklch(0.98 0.008 75);
  transform: translateY(-1px); color: var(--ink);
}

/* Turn marker */
.bq-turn-marker { font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--ink-faint); letter-spacing: 0.18em; text-transform: uppercase;
  display: flex; align-items: center; gap: 12px; margin: 24px 0 16px; }
.bq-turn-marker::before, .bq-turn-marker::after {
  content: ""; height: 1px; flex: 1; background: var(--rule-soft); }
.bq-turn-marker::before { max-width: 18px; }

/* User bubble */
.bq-msg-user { align-self: flex-end; max-width: 85%; margin: 0 0 16px auto;
  background: var(--navy); color: oklch(0.96 0.005 75);
  padding: 14px 20px; border-radius: 18px 18px 4px 18px;
  font-size: 16px; line-height: 1.5; width: fit-content; }

/* Assistant body */
.bq-msg-asst-byline { display: flex; align-items: center; gap: 8px;
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--gold); letter-spacing: 0.16em; text-transform: uppercase;
  margin-bottom: 10px; }
.bq-msg-asst-byline::before { content: ""; width: 14px; height: 1px; background: var(--gold); }
.bq-msg-asst-text { font-family: "Fraunces", serif; font-size: 20px;
  line-height: 1.55; color: var(--ink); font-weight: 400; margin-bottom: 16px; }
.bq-cursor { color: var(--gold); animation: bq-blink 1s steps(2) infinite; }
@keyframes bq-blink { to { opacity: 0; } }

/* Citations */
.bq-citations { border: 0.5px solid var(--rule); border-radius: 12px;
  background: var(--paper); overflow: hidden; margin-bottom: 14px; }
.bq-citations-hd { padding: 10px 16px; background: oklch(0.95 0.01 75);
  border-bottom: 0.5px solid var(--rule);
  display: flex; justify-content: space-between; align-items: center;
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  color: var(--ink-soft); letter-spacing: 0.14em; text-transform: uppercase; }
.bq-cit { display: grid; grid-template-columns: 36px 1fr auto auto auto;
  align-items: center; gap: 14px; padding: 12px 16px;
  border-bottom: 0.5px solid var(--rule-soft); font-size: 14px; }
.bq-cit:last-child { border-bottom: none; }
.bq-cit:hover { background: oklch(0.98 0.008 75); }
.bq-cit-id { font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--ink-faint); letter-spacing: 0.04em; }
.bq-cit-name { font-weight: 500; color: var(--ink);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bq-cit-name .brand { color: var(--ink-faint); font-weight: 400;
  font-style: italic; font-family: "Fraunces", serif; margin-left: 6px; }
.bq-cit-price { font-family: "JetBrains Mono", monospace; font-size: 13px;
  color: var(--ink); font-feature-settings: "tnum"; }
.bq-cit-score { width: 56px; height: 4px; background: var(--rule-soft);
  border-radius: 2px; position: relative; overflow: hidden; }
.bq-cit-score-fill { position: absolute; inset: 0 auto 0 0;
  background: var(--gold); border-radius: 2px; }
.bq-cit-score-num { font-family: "JetBrains Mono", monospace; font-size: 12px;
  color: var(--ink-faint); width: 34px; text-align: right;
  font-feature-settings: "tnum"; }

/* Outfit */
.bq-outfit { border: 0.5px solid var(--rule); border-radius: 14px;
  background: var(--paper); overflow: hidden; margin-bottom: 14px; }
.bq-outfit-hd { padding: 16px 20px 14px;
  background: linear-gradient(180deg, oklch(0.96 0.01 75), var(--paper));
  border-bottom: 0.5px solid var(--rule);
  display: flex; justify-content: space-between; align-items: flex-start; }
.bq-outfit-eyebrow { font-family: "JetBrains Mono", monospace; font-size: 9.5px;
  color: var(--gold); letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 4px; }
.bq-outfit-title { font-family: "Fraunces", serif; font-size: 22px; font-weight: 400;
  letter-spacing: -0.01em; line-height: 1.15; }
.bq-outfit-title em { font-style: italic; color: var(--gold); }
.bq-outfit-tags { display: flex; gap: 6px; }
.bq-outfit-stamp { padding: 4px 10px; border: 0.5px solid var(--rule);
  border-radius: 999px; font-family: "JetBrains Mono", monospace;
  font-size: 10px; color: var(--ink-soft); letter-spacing: 0.1em; text-transform: uppercase; }
.bq-outfit-anchor { padding: 14px 20px; background: oklch(0.95 0.025 78);
  border-bottom: 0.5px solid oklch(0.88 0.05 78);
  display: flex; justify-content: space-between; align-items: center; }
.bq-outfit-anchor-tag { font-family: "JetBrains Mono", monospace; font-size: 9.5px;
  color: var(--gold); letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 2px; }
.bq-outfit-anchor-name { font-family: "Fraunces", serif; font-size: 17px;
  font-weight: 500; color: var(--ink); }
.bq-outfit-anchor-name .brand { color: var(--ink-faint); font-style: italic;
  font-weight: 400; margin-left: 6px; }
.bq-outfit-anchor-price { font-family: "JetBrains Mono", monospace; font-size: 14px;
  color: var(--ink); font-feature-settings: "tnum"; }
.bq-outfit-item { padding: 14px 20px;
  border-bottom: 0.5px solid var(--rule-soft);
  display: grid; grid-template-columns: 24px 1fr auto;
  column-gap: 14px; row-gap: 6px; align-items: baseline; }
.bq-outfit-item:last-of-type { border-bottom: 0.5px solid var(--rule); }
.bq-outfit-item-plus { font-family: "Fraunces", serif; font-size: 18px; color: var(--gold); }
.bq-outfit-item-cat { font-family: "JetBrains Mono", monospace; font-size: 9.5px;
  color: var(--ink-faint); letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 2px; }
.bq-outfit-item-name { font-size: 14px; font-weight: 500; color: var(--ink); }
.bq-outfit-item-name .brand { color: var(--ink-faint); font-weight: 400;
  font-style: italic; font-family: "Fraunces", serif; margin-left: 6px; }
.bq-outfit-item-price { grid-column: 3; grid-row: 1 / 3;
  font-family: "JetBrains Mono", monospace; font-size: 13px;
  color: var(--ink); font-feature-settings: "tnum"; align-self: center; }
.bq-outfit-item-just { grid-column: 2; grid-row: 2;
  font-size: 12.5px; color: var(--ink-soft);
  font-family: "Fraunces", serif; font-style: italic; line-height: 1.45; }
.bq-outfit-item-graph { grid-column: 2; grid-row: 3;
  font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--ink-faint); }
.bq-outfit-foot { padding: 14px 20px;
  display: flex; justify-content: space-between; align-items: center;
  background: oklch(0.96 0.01 75); }
.bq-outfit-total-label { font-family: "JetBrains Mono", monospace; font-size: 10px;
  color: var(--ink-faint); letter-spacing: 0.14em; text-transform: uppercase; }
.bq-outfit-total-num { font-family: "Fraunces", serif; font-size: 22px;
  font-weight: 500; color: var(--ink); font-feature-settings: "tnum"; }

/* Explain */
.bq-explain { border: 0.5px dashed var(--rule); border-radius: 10px;
  padding: 12px 16px; background: oklch(0.97 0.012 80); margin-bottom: 14px; }
.bq-explain-hd { font-family: "JetBrains Mono", monospace; font-size: 10px;
  color: var(--ink-soft); letter-spacing: 0.14em; text-transform: uppercase;
  margin-bottom: 8px; display: flex; justify-content: space-between; }
.bq-explain-row { display: grid;
  grid-template-columns: 50px 1fr repeat(4, 56px) 60px;
  gap: 6px; align-items: center;
  font-family: "JetBrains Mono", monospace; font-size: 10.5px;
  padding: 4px 0; color: var(--ink-soft); }
.bq-explain-row.head { color: var(--ink-faint); font-size: 9.5px;
  letter-spacing: 0.06em; text-transform: uppercase;
  border-bottom: 0.5px solid var(--rule); padding-bottom: 6px; margin-bottom: 2px; }
.bq-explain-row .num { text-align: right; font-feature-settings: "tnum"; }
.bq-explain-row .final { color: var(--gold); font-weight: 500; }

/* Signals */
.bq-signals { border-left: 2px solid var(--gold); padding: 8px 14px;
  background: oklch(0.97 0.014 78); border-radius: 0 6px 6px 0;
  font-family: "JetBrains Mono", monospace; font-size: 12px;
  color: var(--ink-soft); margin-bottom: 14px; }
.bq-signals strong { color: var(--gold); font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.12em; font-size: 9.5px; }

/* Bottom bar — fully transparent, no dark strip */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
  background: transparent !important;
  background-color: transparent !important;
  border-top: none !important;
}
[data-testid="stBottom"] > div {
  padding: 24px 0 20px !important;
}
[data-testid="stBottom"] .block-container,
[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
  max-width: 760px !important;
  margin: 0 auto !important;
}

/* Composer — ChatGPT-style floating pill */
[data-testid="stChatInput"] {
  background: #ffffff !important;
  background-color: #ffffff !important;
  border: 1px solid #dbd5c9 !important;
  border-radius: 24px !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  min-height: 52px;
  overflow: hidden;
}
/* Force ALL inner wrappers transparent so the white shows through */
[data-testid="stChatInput"] div {
  background: transparent !important;
  background-color: transparent !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: #dbd5c9 !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
[data-testid="stChatInput"] *:focus {
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus {
  outline: none !important;
  box-shadow: none !important;
  border: none !important;
}
[data-testid="stChatInput"] textarea {
  font-family: "Inter", sans-serif !important;
  color: #2a2620 !important;
  font-size: 15px !important;
  background: transparent !important;
  background-color: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #928c84 !important;
  font-family: "Fraunces", serif !important;
  font-style: italic !important;
}
[data-testid="stChatInput"] button {
  background: transparent !important;
  background-color: transparent !important;
  color: #928c84 !important;
}
[data-testid="stChatInput"] button:hover {
  color: #d4a73c !important;
}

/* Command output panels */
.bq-cmd-panel { border: 0.5px solid var(--rule); border-radius: 12px;
  background: var(--paper); padding: 16px 20px; margin-bottom: 14px; }
.bq-cmd-title { font-family: "JetBrains Mono", monospace; font-size: 10px;
  color: var(--gold); letter-spacing: 0.16em; text-transform: uppercase;
  margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.bq-cmd-title::before { content: ""; width: 14px; height: 1px; background: var(--gold); }
.bq-cmd-row { display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-bottom: 0.5px solid var(--rule-soft); }
.bq-cmd-row:last-child { border-bottom: none; }
.bq-cmd-label { font-family: "Fraunces", serif; font-size: 13px; font-weight: 500;
  color: var(--ink-soft); min-width: 120px; }
.bq-cmd-value { font-size: 14px; color: var(--ink); }
.bq-cmd-help-row { display: flex; gap: 12px; align-items: baseline;
  padding: 6px 0; }
.bq-cmd-name { font-family: "JetBrains Mono", monospace; font-size: 13px;
  color: var(--gold); font-weight: 500; min-width: 160px; }
.bq-cmd-desc { font-size: 13px; color: var(--ink-soft); }
.bq-system-note { font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--ink-faint); padding: 8px 14px; margin: 8px 0;
  border-left: 2px solid var(--rule); }

/* Debug signals table */
.bq-debug-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.bq-debug-table th { font-family: "JetBrains Mono", monospace; font-size: 10px;
  color: var(--ink-faint); letter-spacing: 0.1em; text-transform: uppercase;
  text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--rule); }
.bq-debug-table td { padding: 8px 8px; border-bottom: 0.5px solid var(--rule-soft);
  font-size: 12px; color: var(--ink-soft); vertical-align: top; }
.bq-debug-table td:first-child { font-family: "JetBrains Mono", monospace;
  font-weight: 500; color: var(--ink); }

/* Hide Streamlit footer/menu */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
"""


def inject_css(css: str) -> None:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers


def _inr(n: int | None) -> str:
    if n is None:
        return ""
    return "₹" + f"{n:,}"


CONFIDENCE_LABELS = [
    (0.20, "Learning…"),
    (0.40, "Getting to know you"),
    (0.60, "Building your profile"),
    (0.80, "Personalized"),
    (1.01, "Dialed in"),
]

BUDGET_TIERS = ["Budget", "Mid", "Premium", "Luxury"]


def _confidence_label(score: float) -> str:
    for cutoff, label in CONFIDENCE_LABELS:
        if score < cutoff:
            return label
    return "Dialed in"


# ─────────────────────────────────────────────────────────────────────────────
# Renderers


def render_brand_block() -> None:
    st.markdown(
        """
        <div class="bq-brand">
          <div class="bq-brand-mark">S</div>
          <div>
            <div class="bq-brand-name">Style<em>Mind</em></div>
            <div class="bq-brand-sub">Personal stylist · in residence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_persona(*, persona: dict[str, Any], turn: int, user_id: str) -> None:
    score = float(persona.get("confidence_score") or 0.0)
    pct = round(score * 100)
    label = _confidence_label(score)
    aesthetics = persona.get("preferred_aesthetics") or []
    occasions = persona.get("top_occasions") or []
    colors = persona.get("color_preferences") or []
    dislikes = persona.get("disliked_materials") or []
    budget = persona.get("budget_tier")
    budget_idx = BUDGET_TIERS.index(budget) if budget in BUDGET_TIERS else -1

    parts = []
    # Confidence
    parts.append(
        f"""
        <div class="bq-section-label">Confidence</div>
        <div class="bq-conf-row">
          <div class="bq-conf-label">{label}</div>
          <div class="bq-conf-pct">{pct}%</div>
        </div>
        <div class="bq-conf-bar"><div class="bq-conf-fill" style="width:{pct}%;"></div></div>
        <div class="bq-conf-ticks"><span>learning</span><span>dialed in</span></div>
        """
    )
    # Aesthetics
    parts.append('<div class="bq-section-label">Aesthetics</div>')
    if aesthetics:
        pills = "".join(f'<span class="bq-tag gold"><span class="bq-tag-dot"></span>{a}</span>' for a in aesthetics)
        parts.append(f'<div class="bq-tags">{pills}</div>')
    else:
        parts.append('<div class="bq-empty-row">— Not yet placed.</div>')

    # Budget
    parts.append('<div class="bq-section-label">Budget</div>')
    pips = "".join(f'<div class="bq-budget-pip{" on" if i <= budget_idx else ""}"></div>' for i in range(4))
    if budget:
        parts.append(
            f'<div class="bq-budget"><div class="bq-budget-tier">{budget}</div>'
            f'<div class="bq-budget-scale">{pips}</div></div>'
        )
    else:
        parts.append(
            '<div class="bq-budget"><div class="bq-empty-row">Reading the room…</div>'
            f'<div class="bq-budget-scale">{pips}</div></div>'
        )

    # Occasions
    parts.append('<div class="bq-section-label">Occasions</div>')
    if occasions:
        pills = "".join(f'<span class="bq-tag navy">{o}</span>' for o in occasions)
        parts.append(f'<div class="bq-tags">{pills}</div>')
    else:
        parts.append('<div class="bq-empty-row">— No scenes yet.</div>')

    # Colors
    if colors:
        parts.append('<div class="bq-section-label">Colors</div>')
        pills = "".join(f'<span class="bq-tag">{c}</span>' for c in colors)
        parts.append(f'<div class="bq-tags">{pills}</div>')

    # Dislikes
    parts.append('<div class="bq-section-label">Crossed out</div>')
    if dislikes:
        pills = "".join(f'<span class="bq-tag muted">{d}</span>' for d in dislikes)
        parts.append(f'<div class="bq-tags">{pills}</div>')
    else:
        parts.append('<div class="bq-empty-row">— Nothing struck through.</div>')

    # Footer
    parts.append(
        f'<div style="margin-top:24px;padding-top:12px;border-top:1px solid var(--rule-soft);'
        f"font-family:JetBrains Mono,monospace;font-size:10.5px;color:var(--ink-faint);"
        f'letter-spacing:0.04em;">turn {turn:02d} · {user_id}</div>'
    )

    st.markdown(
        f'<div style="padding:0 20px 16px;">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_top_bar(*, turn_count: int) -> None:
    if turn_count == 0:
        meta = "session · fresh"
    else:
        meta = f"session · {turn_count} turn{'s' if turn_count != 1 else ''}"
    st.markdown(
        f"""
        <div class="bq-main-bar">
          <div class="bq-main-title">Today's <em>fitting</em></div>
          <div class="bq-main-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        """
        <div class="bq-welcome">
          <div class="bq-welcome-eyebrow">⌁ A new fitting</div>
          <h1 class="bq-welcome-h1">What are we <em>dressing for</em><br>today?</h1>
          <p class="bq-welcome-sub">
            Tell me where you're headed, what you're avoiding, who you're trying to be.
            I'll read between the lines — no quizzes, no checkboxes.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_turn_marker(turn_idx: int) -> None:
    st.markdown(
        f'<div class="bq-turn-marker">Turn {turn_idx:02d}</div>',
        unsafe_allow_html=True,
    )


def render_user_bubble(text: str) -> None:
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="bq-msg-user">{safe}</div>', unsafe_allow_html=True)


def render_citations(sources: list[dict[str, Any]]) -> None:
    rows = []
    for s in sources:
        score = float(s.get("score", 0))
        rows.append(
            '<div class="bq-cit">'
            f'<span class="bq-cit-id">{s.get("product_id", "")}</span>'
            f'<span class="bq-cit-name">{s.get("name", "")}'
            f'<span class="brand">{s.get("brand", "")}</span></span>'
            f'<span class="bq-cit-price">{_inr(s.get("price_inr"))}</span>'
            f'<span class="bq-cit-score">'
            f'<span class="bq-cit-score-fill" style="width:{score * 100:.0f}%;"></span></span>'
            f'<span class="bq-cit-score-num">{score:.2f}</span>'
            "</div>"
        )
    body = "".join(rows)
    st.markdown(
        '<div class="bq-citations">'
        '<div class="bq-citations-hd">'
        f"<span>{len(sources)} pulled · re-ranked for you</span>"
        "<span>relevance</span>"
        "</div>"
        f"{body}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_explain_table(rows: list[dict[str, Any]]) -> None:
    body = []
    for r in rows:
        body.append(
            '<div class="bq-explain-row">'
            f'<span>{r.get("product_id", "")}</span><span></span>'
            f'<span class="num">{r.get("base_score", 0):.2f}</span>'
            f'<span class="num">+{r.get("persona_boost", 0):.2f}</span>'
            f'<span class="num">−{r.get("penalty", 0):.2f}</span>'
            f'<span class="num">+{r.get("budget_boost", 0):.2f}</span>'
            f'<span class="num final">{r.get("final_score", 0):.2f}</span>'
            "</div>"
        )
    st.markdown(
        '<div class="bq-explain">'
        '<div class="bq-explain-hd">'
        "<span>Score breakdown · explain=true</span>"
        "<span>persona-aware</span>"
        "</div>"
        '<div class="bq-explain-row head">'
        '<span>id</span><span></span>'
        '<span class="num">base</span><span class="num">+pers</span>'
        '<span class="num">−pen</span><span class="num">+budg</span>'
        '<span class="num">final</span>'
        "</div>"
        f"{''.join(body)}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_outfit_card(outfit: dict[str, Any]) -> None:
    anchor = outfit.get("anchor", {})
    items = outfit.get("items", [])
    occasion = outfit.get("occasion", "")
    season = outfit.get("season", "")
    total = (anchor.get("price_inr") or 0) + sum(it.get("price_inr", 0) for it in items)

    items_html = []
    for it in items:
        items_html.append(
            '<div class="bq-outfit-item">'
            '<span class="bq-outfit-item-plus">＋</span>'
            "<div>"
            f'<div class="bq-outfit-item-cat">{it.get("category", "")}</div>'
            f'<div class="bq-outfit-item-name">{it.get("name", "")}'
            f'<span class="brand">{it.get("brand", "")}</span></div>'
            "</div>"
            f'<span class="bq-outfit-item-price">{_inr(it.get("price_inr"))}</span>'
            f'<div class="bq-outfit-item-just">"{it.get("justification", "")}"</div>'
            f'<div class="bq-outfit-item-graph">{it.get("graph_path", "")}</div>'
            "</div>"
        )

    season_label = "summer" if season == "SS" else "winter"
    st.markdown(
        '<div class="bq-outfit">'
        '<div class="bq-outfit-hd">'
        "<div>"
        '<div class="bq-outfit-eyebrow">A complete look · graph-built</div>'
        f'<div class="bq-outfit-title">Around the <em>{anchor.get("name", "")}</em></div>'
        "</div>"
        '<div class="bq-outfit-tags">'
        f'<span class="bq-outfit-stamp">{occasion}</span>'
        f'<span class="bq-outfit-stamp">{season} · {season_label}</span>'
        "</div>"
        "</div>"
        '<div class="bq-outfit-anchor">'
        "<div>"
        '<div class="bq-outfit-anchor-tag">Anchor</div>'
        f'<div class="bq-outfit-anchor-name">{anchor.get("name", "")}'
        f'<span class="brand">{anchor.get("brand", "")}</span></div>'
        "</div>"
        f'<div class="bq-outfit-anchor-price">{_inr(anchor.get("price_inr"))}</div>'
        "</div>"
        f"{''.join(items_html)}"
        '<div class="bq-outfit-foot">'
        f'<span class="bq-outfit-total-label">Complete look · {len(items) + 1} pieces</span>'
        f'<span class="bq-outfit-total-num">{_inr(total)}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_signals_strip(signals: dict[str, Any]) -> None:
    parts: list[str] = []
    if signals.get("liked_aesthetics"):
        parts.append("aesthetics: " + ", ".join(signals["liked_aesthetics"]))
    if signals.get("mentioned_occasions"):
        parts.append("occasions: " + ", ".join(signals["mentioned_occasions"]))
    if signals.get("budget_signal"):
        parts.append(f"budget: {signals['budget_signal']}")
    if signals.get("color_preferences"):
        parts.append("colors: " + ", ".join(signals["color_preferences"]))
    if signals.get("sentiment_on_shown"):
        s = ", ".join(f"{k}:{v}" for k, v in signals["sentiment_on_shown"].items())
        if s:
            parts.append(f"sentiment: {s}")
    weight = signals.get("signal_strength")
    weight_str = f' <span style="opacity:.6">(weight {weight:.1f})</span>' if weight is not None else ""
    if not parts:
        return
    st.markdown(
        '<div class="bq-signals">'
        "<strong>Quietly noted</strong>"
        f'<div style="margin-top:4px;">{"  ·  ".join(parts)}{weight_str}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_help_panel() -> None:
    commands = [
        ("/persona", "Show your current inferred style persona"),
        ("/outfit &lt;name&gt;", "Build a complete outfit around a product"),
        ("/debug-dev", "Show all persona signals extracted this session"),
        ("/clear", "Clear conversation history and start fresh"),
        ("/help", "Show this help panel"),
    ]
    rows = "".join(
        f'<div class="bq-cmd-help-row">'
        f'<span class="bq-cmd-name">{name}</span>'
        f'<span class="bq-cmd-desc">{desc}</span>'
        f"</div>"
        for name, desc in commands
    )
    st.markdown(
        '<div class="bq-cmd-panel">'
        '<div class="bq-cmd-title">Available Commands</div>'
        f"{rows}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_persona_panel(persona: dict[str, Any]) -> None:
    score = float(persona.get("confidence_score") or 0.0)
    pct = round(score * 100)
    label = _confidence_label(score)
    aesthetics = persona.get("preferred_aesthetics") or []
    occasions = persona.get("top_occasions") or []
    colors = persona.get("color_preferences") or []
    dislikes = persona.get("disliked_materials") or []
    budget = persona.get("budget_tier")

    def _pills(items: list[str], cls: str = "") -> str:
        if not items:
            return '<span style="font-style:italic;color:var(--ink-faint);">none yet</span>'
        return " ".join(f'<span class="bq-tag {cls}">{x}</span>' for x in items)

    rows = (
        '<div class="bq-cmd-row">'
        '<span class="bq-cmd-label">Aesthetics</span>'
        f'<div class="bq-tags">{_pills(aesthetics, "gold")}</div>'
        "</div>"
        '<div class="bq-cmd-row">'
        '<span class="bq-cmd-label">Occasions</span>'
        f'<div class="bq-tags">{_pills(occasions, "navy")}</div>'
        "</div>"
        '<div class="bq-cmd-row">'
        '<span class="bq-cmd-label">Budget</span>'
        f'<span class="bq-cmd-value">{budget or "unknown"}</span>'
        "</div>"
        '<div class="bq-cmd-row">'
        '<span class="bq-cmd-label">Colors</span>'
        f'<div class="bq-tags">{_pills(colors)}</div>'
        "</div>"
        '<div class="bq-cmd-row">'
        '<span class="bq-cmd-label">Dislikes</span>'
        f'<div class="bq-tags">{_pills(dislikes, "muted")}</div>'
        "</div>"
        '<div class="bq-cmd-row">'
        '<span class="bq-cmd-label">Confidence</span>'
        f'<span class="bq-cmd-value">{label} ({pct}%)</span>'
        "</div>"
    )
    st.markdown(
        '<div class="bq-cmd-panel">'
        '<div class="bq-cmd-title">Your Style Persona</div>'
        f"{rows}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_debug_signals(signals_log: list[dict[str, Any]]) -> None:
    if not signals_log:
        st.markdown(
            '<div class="bq-system-note">No persona signals extracted yet. Chat first!</div>',
            unsafe_allow_html=True,
        )
        return
    rows = []
    for idx, s in enumerate(signals_log, 1):
        if not s:
            continue
        turn = s.get("turn", idx)
        aesthetics = ", ".join(f"+{a}" for a in s.get("liked_aesthetics", [])) or "—"
        materials = ", ".join(f"−{m}" for m in s.get("disliked_materials", [])) or "—"
        budget = s.get("budget_signal") or "—"
        occasions = ", ".join(s.get("mentioned_occasions", [])) or "—"
        colors = ", ".join(s.get("color_preferences", [])) or "—"
        brands = ", ".join(s.get("brand_mentions", [])) or "—"
        strength = f'{s.get("signal_strength", 0):.2f}'
        rows.append(
            f"<tr><td>{turn}</td><td>{aesthetics}</td><td>{materials}</td>"
            f"<td>{budget}</td><td>{occasions}</td><td>{colors}</td>"
            f"<td>{brands}</td><td>{strength}</td></tr>"
        )
    st.markdown(
        '<div class="bq-cmd-panel">'
        '<div class="bq-cmd-title">Persona Signal Log</div>'
        '<table class="bq-debug-table">'
        "<thead><tr>"
        "<th>Turn</th><th>Aesthetics</th><th>Materials</th>"
        "<th>Budget</th><th>Occasions</th><th>Colors</th>"
        "<th>Brands</th><th>Strength</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_system_note(text: str) -> None:
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="bq-system-note">{safe}</div>',
        unsafe_allow_html=True,
    )
