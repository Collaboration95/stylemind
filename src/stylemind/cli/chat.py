from __future__ import annotations

import json
import logging
from collections.abc import Generator
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

Commands:
  [green]/persona[/green]  — show your current style persona
  [green]quit[/green] / [green]exit[/green] — exit the chat

Type your fashion question to get started!
"""


class ChatCLI:
    def __init__(self, base_url: str, user_id: str, explain: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.explain = explain
        self.console = Console()
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.console.print(_WELCOME.format(user_id=self.user_id))
        while True:
            try:
                user_input = prompt("You: ").strip()
            except EOFError, KeyboardInterrupt:
                self.console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in {"quit", "exit"}:
                self.console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/persona":
                self._show_persona()
                continue

            self._send_message(user_input)

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    def _send_message(self, message: str) -> None:
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
                        # structured response payload
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

        self.console.print()  # newline after streamed response

        # Update conversation history
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": full_text})

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[str]:
        """Parse SSE data lines from a streaming httpx response."""
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
    # Structured payload handling (products / outfit)
    # ------------------------------------------------------------------

    def _handle_structured(self, json_str: str) -> None:
        """Render product citations, explain breakdowns, and outfit suggestions if present."""
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Could not decode structured payload")
            return

        sources: list[dict[str, Any]] = payload.get("sources", [])
        explain: list[dict[str, Any]] = payload.get("explain", [])
        outfit: dict[str, Any] | None = payload.get("outfit")

        if sources:
            self._render_sources(sources)

        if explain:
            self._render_explain(explain)

        if outfit:
            self._render_outfit(outfit)

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
