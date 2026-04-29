from __future__ import annotations

import json
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import httpx
from prompt_toolkit import prompt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

_WELCOME = """
[bold cyan]StyleMind Fashion Assistant[/bold cyan]
User ID: [bold]{user_id}[/bold]

Type [green]/help[/green] to see available commands.
Type your fashion question to get started!
"""

_HELP_TEXT = """
[bold cyan]Available Commands[/bold cyan]

  [green]/help[/green]        Show this help message
  [green]/persona[/green]     Show your current inferred style persona
  [green]/debug-dev[/green]   Show all persona signals extracted this session (dev tool)
  [green]/clear[/green]       Clear conversation history and start fresh
  [green]/exit[/green]        Exit the chat (also: /quit, quit, exit)
"""


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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.console.print(_WELCOME.format(user_id=self.user_id))
        while True:
            try:
                user_input = prompt("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in {"quit", "exit", "/quit", "/exit"}:
                self.console.print("[dim]Goodbye![/dim]")
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
                self.console.print("[dim]Conversation history cleared.[/dim]")
                continue

            self._send_message(user_input)

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
        self.console.print("[bold green]StyleMind:[/bold green]", end=" ")

        full_text = ""
        try:
            with (
                httpx.Client(timeout=60.0) as client,
                client.stream("POST", f"{self.base_url}/chat", json=payload) as response,
            ):
                response.raise_for_status()
                for chunk in self._parse_sse_stream(response):
                    if chunk.startswith("__JSON__"):
                        self._handle_structured(chunk[8:])
                    else:
                        self.console.print(chunk, end="", highlight=False)
                        full_text += chunk
        except httpx.HTTPStatusError as exc:
            self.console.print(f"[red]HTTP error {exc.response.status_code}[/red]")
            return
        except httpx.RequestError as exc:
            self.console.print(f"[red]Connection error: {exc}[/red]")
            return

        self.console.print()

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": full_text})

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[str]:
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
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

        panel = Panel(
            "\n".join(content_lines),
            title="[bold yellow]Score Breakdown[/bold yellow]",
            expand=False,
        )
        self.console.print(panel)

    def _render_sources(self, sources: list[dict[str, Any]]) -> None:
        content_lines = []
        for s in sources:
            name = s.get("name", "Unknown")
            brand = s.get("brand", "")
            price = s.get("price_inr", 0)
            score = s.get("score", 0.0)
            content_lines.append(f"• [bold]{name}[/bold] by {brand} — ₹{price:,}  (score: {score:.2f})")

        panel = Panel(
            "\n".join(content_lines),
            title="[bold blue]Product Citations[/bold blue]",
            expand=False,
        )
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

        aesthetics = ", ".join(data.get("preferred_aesthetics", [])) or "[dim]none yet[/dim]"
        disliked = ", ".join(data.get("disliked_materials", [])) or "[dim]none[/dim]"
        budget = data.get("budget_tier") or "[dim]unknown[/dim]"
        occasions = ", ".join(data.get("top_occasions", [])) or "[dim]none yet[/dim]"
        confidence = data.get("confidence_score", 0.0)

        lines = [
            f"[green]Aesthetics:[/green]       {aesthetics}",
            f"[red]Disliked materials:[/red] {disliked}",
            f"[yellow]Budget tier:[/yellow]      {budget}",
            f"[blue]Occasions:[/blue]        {occasions}",
            f"[magenta]Confidence:[/magenta]       {confidence:.2f}",
        ]

        panel = Panel(
            "\n".join(lines),
            title="[bold]Your Style Persona[/bold]",
            expand=False,
        )
        self.console.print(panel)

    # ------------------------------------------------------------------
    # /debug-dev — session signal log
    # ------------------------------------------------------------------

    def _show_debug_dev(self) -> None:
        if not self._signal_log:
            self.console.print("[dim]No persona signals extracted yet. Chat first![/dim]")
            return

        table = Table(
            title="Persona Signal Log (this session)",
            show_header=True,
            header_style="bold yellow",
        )
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
