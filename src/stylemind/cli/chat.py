from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

import httpx
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

_STARTERS = [
    "I need something for a date night",
    "Show me minimal summer outfits under 5k",
    "I love the quiet luxury aesthetic",
    "What's good for an office look?",
    "I want casual streetwear vibes",
    "Show me something in earthy tones",
    "I need a wedding guest outfit",
    "What goes well with linen pants?",
]

_HELP_TEXT = """
[bold cyan]Available Commands[/bold cyan]

  [green]/help[/green]              Show this help message
  [green]/persona[/green]           Show your current inferred style persona
  [green]/outfit[/green] [dim]<name>[/dim]     Build a complete outfit around a product
  [green]/debug-dev[/green]         Show all persona signals extracted this session (dev tool)
  [green]/clear[/green]             Clear conversation history and start fresh
  [green]/exit[/green]              Exit the chat (also: /quit, quit, exit)

[dim]Tab-complete product names when typing![/dim]
"""

_CONFIDENCE_LABELS = [
    (0.0, "[dim]learning...[/dim]"),
    (0.2, "[yellow]getting to know you[/yellow]"),
    (0.4, "[green]building your profile[/green]"),
    (0.6, "[green]personalized[/green]"),
    (0.8, "[bold green]dialed in[/bold green]"),
]


def _confidence_label(score: float) -> str:
    label = _CONFIDENCE_LABELS[0][1]
    for threshold, text in _CONFIDENCE_LABELS:
        if score >= threshold:
            label = text
    return label


class ProductNameCompleter(Completer):
    """Complete product names anywhere in the input, not just at the start."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text:
            return
        text_lower = text.lower()
        for name in self._names:
            name_lower = name.lower()
            for i in range(len(text)):
                if i > 0 and text[i - 1] != " ":
                    continue
                tail = text_lower[i:]
                if tail and name_lower.startswith(tail) and name_lower != tail:
                    yield Completion(name[len(tail) :], start_position=0)
                    break


@dataclass
class TurnSignals:
    turn: int
    liked_aesthetics: list[str] = field(default_factory=list)
    disliked_materials: list[str] = field(default_factory=list)
    mentioned_occasions: list[str] = field(default_factory=list)
    budget_signal: str | None = None
    color_preferences: list[str] = field(default_factory=list)
    brand_mentions: list[str] = field(default_factory=list)
    signal_strength: float = 0.0


class ChatCLI:
    def __init__(self, base_url: str, user_id: str, explain: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.explain = explain
        self.console = Console()
        self._history: list[dict[str, str]] = []
        self._turn_count = 0
        self._signal_log: list[TurnSignals] = []
        self._last_confidence: float = 0.0
        self._product_catalog: list[dict[str, str]] = []
        self._completer: ProductNameCompleter | None = None
        self._starters: list[str] = []

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_product_catalog(self) -> None:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/products/names")
                if resp.status_code == 200:
                    self._product_catalog = resp.json()
                    names = [p["name"] for p in self._product_catalog]
                    self._completer = ProductNameCompleter(names)
                    logger.debug("cli loaded %d product names for autocomplete", len(names))
        except Exception as exc:
            logger.debug("cli product catalog load failed error=%s", exc)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._load_product_catalog()

        self._starters = random.sample(_STARTERS, min(3, len(_STARTERS)))

        welcome_lines = [
            "",
            "[bold cyan]StyleMind[/bold cyan] — Your Personal Fashion Stylist",
            "",
            "Hi! I'm StyleMind. I learn your style as we chat — no questionnaires,",
            "just natural conversation. Tell me what you're looking for and I'll",
            "find pieces that match your vibe.",
            "",
            f"[dim]Session:[/dim] [bold]{self.user_id}[/bold] [dim](new session — your style starts fresh!)[/dim]",
            "",
            "[dim]Try one of these to get started:[/dim]",
        ]
        for i, s in enumerate(self._starters, 1):
            welcome_lines.append(f'  [green]{i}.[/green] "{s}"')
        welcome_lines.append("")
        welcome_lines.append("[dim]Type[/dim] [green]/help[/green] [dim]for all commands.[/dim]")
        welcome_lines.append("")

        self.console.print("\n".join(welcome_lines))

        while True:
            try:
                prompt_text = f"You (turn {self._turn_count + 1}): " if self._turn_count > 0 else "You: "
                user_input = prompt(prompt_text, completer=self._completer).strip()
            except EOFError, KeyboardInterrupt:
                self._exit_with_summary()
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in {"quit", "exit", "/quit", "/exit"}:
                self._exit_with_summary()
                break

            if cmd == "/help":
                self.console.print(_HELP_TEXT)
                continue

            if cmd == "/persona":
                self._show_persona()
                continue

            if cmd == "/debug-dev":
                self._show_debug_dev()
                continue

            if cmd == "/clear":
                self._history.clear()
                self._turn_count = 0
                self._signal_log.clear()
                self._last_confidence = 0.0
                self.console.print("[dim]Conversation history cleared. Fresh start![/dim]")
                continue

            if cmd.startswith("/outfit"):
                self._handle_outfit_command(user_input)
                continue

            # Numbered starter shortcut (1, 2, 3)
            if cmd in {"1", "2", "3"} and self._turn_count == 0 and self._starters:
                idx = int(cmd) - 1
                if idx < len(self._starters):
                    self.console.print(f"[dim]> {self._starters[idx]}[/dim]")
                    self._send_message(self._starters[idx])
                    continue

            self._send_message(user_input)

    # ------------------------------------------------------------------
    # /outfit command
    # ------------------------------------------------------------------

    def _handle_outfit_command(self, raw_input: str) -> None:
        parts = raw_input.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            self.console.print("[dim]Usage: /outfit <product name>[/dim]")
            if self._product_catalog:
                sample = random.sample(self._product_catalog, min(3, len(self._product_catalog)))
                self.console.print("[dim]Examples:[/dim]")
                for p in sample:
                    self.console.print(f"  [green]/outfit {p['name']}[/green]")
            return

        query = parts[1].strip()
        product_id = self._fuzzy_match_product(query)
        if not product_id:
            self.console.print(f'[red]No product matching "{query}" found.[/red]')
            close = get_close_matches(
                query.lower(), [p["name"].lower() for p in self._product_catalog], n=3, cutoff=0.4
            )
            if close:
                self.console.print("[dim]Did you mean:[/dim]")
                for name in close:
                    matched = next((p for p in self._product_catalog if p["name"].lower() == name), None)
                    if matched:
                        self.console.print(f"  [green]/outfit {matched['name']}[/green]")
            return

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{self.base_url}/outfit/{product_id}",
                    params={"user_id": self.user_id},
                )
                resp.raise_for_status()
                outfit = resp.json()
                self._render_outfit(outfit)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                self.console.print(f'[red]Product "{query}" not found in graph.[/red]')
            else:
                self.console.print(f"[red]Outfit build failed: HTTP {exc.response.status_code}[/red]")
        except httpx.RequestError as exc:
            self.console.print(f"[red]Connection error: {exc}[/red]")

    def _fuzzy_match_product(self, query: str) -> str | None:
        query_lower = query.lower()
        for p in self._product_catalog:
            if p["name"].lower() == query_lower:
                return p["product_id"]
        for p in self._product_catalog:
            if query_lower in p["name"].lower():
                return p["product_id"]
        close = get_close_matches(query_lower, [p["name"].lower() for p in self._product_catalog], n=1, cutoff=0.5)
        if close:
            matched = next((p for p in self._product_catalog if p["name"].lower() == close[0]), None)
            if matched:
                return matched["product_id"]
        return None

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    def _send_message(self, message: str) -> None:
        self._turn_count += 1
        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "message": message,
            "history": self._history,
            "explain": self.explain,
        }

        self.console.print()
        status = self.console.status("[dim]Thinking...[/dim]", spinner="dots")
        status.start()

        full_text = ""
        first_chunk = True
        start = time.monotonic()
        try:
            with (
                httpx.Client(timeout=60.0) as client,
                client.stream("POST", f"{self.base_url}/chat", json=payload) as response,
            ):
                response.raise_for_status()
                for chunk in self._parse_sse_stream(response):
                    if first_chunk:
                        status.stop()
                        self.console.print("[bold green]StyleMind:[/bold green]", end=" ")
                        first_chunk = False
                    self.console.print(chunk, end="", highlight=False)
                    full_text += chunk
        except httpx.HTTPStatusError as exc:
            status.stop()
            self.console.print(f"[red]HTTP error {exc.response.status_code}[/red]")
            return
        except httpx.RequestError as exc:
            status.stop()
            self.console.print(f"[red]Connection error: {exc}[/red]")
            return

        if first_chunk:
            status.stop()

        elapsed = time.monotonic() - start
        self.console.print()

        self._update_confidence()
        conf_label = _confidence_label(self._last_confidence)
        self.console.print(f"[dim]  {elapsed:.1f}s | persona: {conf_label}[/dim]")

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": full_text})

    def _update_confidence(self) -> None:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{self.base_url}/persona/{self.user_id}")
                if resp.status_code == 200:
                    self._last_confidence = resp.json().get("confidence_score", 0.0)
        except Exception:
            pass

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[str]:
        """Yield text chunks; handle structured 'event: json' events inline.

        SSE event type field distinguishes structured payloads from LLM text,
        eliminating any risk of collision with LLM-generated content.
        """
        event_type: str | None = None
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                event_type = None
                continue
            if line.startswith("event: "):
                event_type = line[7:].strip()
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    return
                if event_type == "json":
                    self._handle_structured(data)
                    event_type = None
                else:
                    yield data

    # ------------------------------------------------------------------
    # Structured payload handling (products / outfit / signals)
    # ------------------------------------------------------------------

    def _handle_structured(self, json_str: str) -> None:
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Could not decode structured payload")
            return

        sources: list[dict[str, Any]] = payload.get("sources", [])
        explain: list[dict[str, Any]] = payload.get("explain", [])
        outfit: dict[str, Any] | None = payload.get("outfit")
        signals: dict[str, Any] | None = payload.get("signals")

        if sources:
            self._render_sources(sources)
        if explain:
            self._render_explain(explain)
        if outfit:
            self._render_outfit(outfit)
        if signals:
            self._capture_signals(signals)

    def _capture_signals(self, signals: dict[str, Any]) -> None:
        ts = TurnSignals(
            turn=self._turn_count,
            liked_aesthetics=signals.get("liked_aesthetics", []),
            disliked_materials=signals.get("disliked_materials", []),
            mentioned_occasions=signals.get("mentioned_occasions", []),
            budget_signal=signals.get("budget_signal"),
            color_preferences=signals.get("color_preferences", []),
            brand_mentions=signals.get("brand_mentions", []),
            signal_strength=signals.get("signal_strength", 0.0),
        )
        self._signal_log.append(ts)

    def _render_explain(self, explain: list[dict[str, Any]]) -> None:
        content_lines = []
        for e in explain:
            pid = e.get("product_id", "?")
            base = e.get("base_score", 0.0)
            boost = e.get("persona_boost", 0.0)
            penalty = e.get("penalty", 0.0)
            budget = e.get("budget_boost", 0.0)
            final = e.get("final_score", 0.0)
            content_lines.append(
                f"• [bold]{pid}[/bold]  base={base:.3f}  boost={boost:+.3f}  penalty={penalty:+.3f}  budget={budget:+.3f}  →  [cyan]{final:.3f}[/cyan]"
            )
        self.console.print()
        panel = Panel("\n".join(content_lines), title="[bold yellow]Score Breakdown[/bold yellow]", expand=False)
        self.console.print(panel)

    def _render_sources(self, sources: list[dict[str, Any]]) -> None:
        content_lines = []
        for s in sources:
            name = s.get("name", "Unknown")
            brand = s.get("brand", "")
            price = s.get("price_inr", 0)
            score = s.get("score", 0.0)
            content_lines.append(f"• [bold]{name}[/bold] by {brand} — ₹{price:,}  (score: {score:.2f})")
        self.console.print()
        panel = Panel("\n".join(content_lines), title="[bold blue]Product Citations[/bold blue]", expand=False)
        self.console.print(panel)

    def _render_outfit(self, outfit: dict[str, Any]) -> None:
        table = Table(
            title=f"Outfit Suggestion — {outfit.get('occasion', '')} / {outfit.get('season', '')}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Item", style="bold")
        table.add_column("Category")
        table.add_column("Brand")
        table.add_column("Price (₹)", justify="right")
        table.add_column("Justification")

        anchor = outfit.get("anchor", {})
        if anchor:
            table.add_row(
                anchor.get("name", ""),
                anchor.get("category", ""),
                anchor.get("brand", ""),
                f"₹{anchor.get('price_inr', 0):,}",
                "[dim]anchor[/dim]",
            )
        for item in outfit.get("items", []):
            table.add_row(
                item.get("name", ""),
                item.get("category", ""),
                item.get("brand", ""),
                f"₹{item.get('price_inr', 0):,}",
                item.get("justification", ""),
            )
        self.console.print(table)

    # ------------------------------------------------------------------
    # Persona display
    # ------------------------------------------------------------------

    def _show_persona(self) -> None:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/persona/{self.user_id}")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            self.console.print(f"[red]HTTP error {exc.response.status_code}[/red]")
            return
        except httpx.RequestError as exc:
            self.console.print(f"[red]Connection error: {exc}[/red]")
            return
        self._render_persona_panel(data)

    def _render_persona_panel(self, data: dict[str, Any]) -> None:
        aesthetics = ", ".join(data.get("preferred_aesthetics", [])) or "[dim]none yet[/dim]"
        disliked = ", ".join(data.get("disliked_materials", [])) or "[dim]none[/dim]"
        budget = data.get("budget_tier") or "[dim]unknown[/dim]"
        occasions = ", ".join(data.get("top_occasions", [])) or "[dim]none yet[/dim]"
        confidence = data.get("confidence_score", 0.0)
        conf_label = _confidence_label(confidence)

        lines = [
            f"[green]Aesthetics:[/green]       {aesthetics}",
            f"[red]Disliked materials:[/red] {disliked}",
            f"[yellow]Budget tier:[/yellow]      {budget}",
            f"[blue]Occasions:[/blue]        {occasions}",
            f"[magenta]Confidence:[/magenta]       {confidence:.2f} ({conf_label})",
        ]
        panel = Panel("\n".join(lines), title="[bold]Your Style Persona[/bold]", expand=False)
        self.console.print(panel)

    # ------------------------------------------------------------------
    # /debug-dev — session signal log
    # ------------------------------------------------------------------

    def _show_debug_dev(self) -> None:
        if not self._signal_log:
            self.console.print("[dim]No persona signals extracted yet. Chat first![/dim]")
            return

        table = Table(title="Persona Signal Log (this session)", show_header=True, header_style="bold yellow")
        table.add_column("Turn", justify="right", style="bold")
        table.add_column("Aesthetics")
        table.add_column("Materials")
        table.add_column("Budget")
        table.add_column("Occasions")
        table.add_column("Colors")
        table.add_column("Brands")
        table.add_column("Strength", justify="right")

        for ts in self._signal_log:
            aesthetics = ", ".join(f"+{a}" for a in ts.liked_aesthetics) if ts.liked_aesthetics else "[dim]-[/dim]"
            materials = ", ".join(f"-{m}" for m in ts.disliked_materials) if ts.disliked_materials else "[dim]-[/dim]"
            budget = ts.budget_signal or "[dim]-[/dim]"
            occasions = ", ".join(f"+{o}" for o in ts.mentioned_occasions) if ts.mentioned_occasions else "[dim]-[/dim]"
            colors = ", ".join(ts.color_preferences) if ts.color_preferences else "[dim]-[/dim]"
            brands = ", ".join(ts.brand_mentions) if ts.brand_mentions else "[dim]-[/dim]"
            strength = f"{ts.signal_strength:.2f}"
            table.add_row(str(ts.turn), aesthetics, materials, budget, occasions, colors, brands, strength)

        self.console.print(table)

    # ------------------------------------------------------------------
    # Exit with session summary
    # ------------------------------------------------------------------

    def _exit_with_summary(self) -> None:
        if self._turn_count > 0 and self._signal_log:
            self.console.print()
            self.console.print("[bold cyan]Session Summary[/bold cyan]")
            self.console.print(f"[dim]Turns: {self._turn_count} | Signals extracted: {len(self._signal_log)}[/dim]")
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(f"{self.base_url}/persona/{self.user_id}")
                    if resp.status_code == 200:
                        self._render_persona_panel(resp.json())
            except Exception:
                pass
        self.console.print("[dim]Goodbye! Your style persona is saved for next time.[/dim]")
