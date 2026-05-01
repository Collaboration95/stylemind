"""SSE streaming client + persona fetch.

The /chat endpoint emits text chunks as unnamed SSE data events and structured
payloads as named 'event: json' events. We yield text for st.write_stream-style
consumption and capture the JSON payloads into a dict supplied by the caller.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx


def stream_chat(
    *,
    base_url: str,
    user_id: str,
    message: str,
    history: list[dict[str, str]],
    explain: bool,
    captured: dict[str, Any],
) -> Iterator[str]:
    """Yield text chunks. Capture sources/signals/explain into ``captured``."""
    payload = {
        "user_id": user_id,
        "message": message,
        "history": [{"role": m["role"], "content": m["content"]} for m in history],
        "explain": explain,
    }

    captured.setdefault("sources", [])
    captured.setdefault("signals", {})
    captured.setdefault("explain", [])

    with (
        httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client,
        client.stream("POST", f"{base_url}/chat", json=payload) as resp,
    ):
        resp.raise_for_status()
        event_type: str | None = None
        for raw in resp.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode().strip()
            if not line:
                event_type = None
                continue
            if line.startswith("event: "):
                event_type = line[7:].strip()
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            if event_type == "json":
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    event_type = None
                    continue
                if "sources" in evt:
                    captured["sources"].extend(evt["sources"])
                if "signals" in evt:
                    captured["signals"] = evt["signals"]
                if "explain" in evt:
                    captured["explain"].extend(evt["explain"])
                event_type = None
            else:
                yield data


def fetch_persona(base_url: str, user_id: str) -> dict[str, Any]:
    """GET /persona/{user_id} with a small grace window for the fire-and-forget
    persona persistence triggered by /chat[DONE]."""
    time.sleep(0.15)
    with httpx.Client(timeout=5.0) as client:
        r = client.get(f"{base_url}/persona/{user_id}")
        r.raise_for_status()
        return r.json()


def fetch_outfit(base_url: str, product_id: str, user_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=15.0) as client:
        r = client.get(
            f"{base_url}/outfit/{product_id}",
            params={"user_id": user_id},
        )
        r.raise_for_status()
        return r.json()
