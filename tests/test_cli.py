from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stylemind.cli.chat import ChatCLI

# ---------------------------------------------------------------------------
# Unit tests — SSE parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseSseStream:
    def _cli(self) -> ChatCLI:
        return ChatCLI(base_url="http://localhost:8000", user_id="test1234")

    def _make_response(self, lines: list[str]) -> MagicMock:
        mock = MagicMock()
        mock.iter_lines.return_value = iter(lines)
        return mock

    def test_single_chunk(self) -> None:
        cli = self._cli()
        response = self._make_response(["data: hello world"])
        result = list(cli._parse_sse_stream(response))
        assert result == ["hello world"]

    def test_multiple_chunks(self) -> None:
        cli = self._cli()
        response = self._make_response(["data: chunk one", "data: chunk two", "data: chunk three"])
        result = list(cli._parse_sse_stream(response))
        assert result == ["chunk one", "chunk two", "chunk three"]

    def test_done_terminates_stream(self) -> None:
        cli = self._cli()
        response = self._make_response(["data: hello", "data: [DONE]", "data: should not appear"])
        result = list(cli._parse_sse_stream(response))
        assert result == ["hello"]

    def test_empty_lines_skipped(self) -> None:
        cli = self._cli()
        response = self._make_response(["", "data: hello", "", "data: world", ""])
        result = list(cli._parse_sse_stream(response))
        assert result == ["hello", "world"]

    def test_non_data_lines_skipped(self) -> None:
        cli = self._cli()
        response = self._make_response(
            [
                "event: message",
                "id: 1",
                "data: actual content",
                ": keep-alive comment",
            ]
        )
        result = list(cli._parse_sse_stream(response))
        assert result == ["actual content"]

    def test_empty_stream(self) -> None:
        cli = self._cli()
        response = self._make_response([])
        result = list(cli._parse_sse_stream(response))
        assert result == []

    def test_only_done(self) -> None:
        cli = self._cli()
        response = self._make_response(["data: [DONE]"])
        result = list(cli._parse_sse_stream(response))
        assert result == []

    def test_whitespace_trimmed_from_lines(self) -> None:
        cli = self._cli()
        response = self._make_response(["  data: trimmed  ", "data: normal"])
        result = list(cli._parse_sse_stream(response))
        # Lines are fully stripped so "  data: trimmed  " -> "data: trimmed" -> data="trimmed"
        assert result == ["trimmed", "normal"]

    def test_data_with_json_content(self) -> None:
        cli = self._cli()
        response = self._make_response(['data: {"key": "value"}'])
        result = list(cli._parse_sse_stream(response))
        assert result == ['{"key": "value"}']


# ---------------------------------------------------------------------------
# E2E smoke test — requires a running server
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCliSmoke:
    def test_server_health(self) -> None:
        """Verify the server is running and healthy."""
        import httpx

        try:
            resp = httpx.get("http://localhost:8000/health", timeout=3.0)
        except httpx.ConnectError:
            pytest.skip("No server running at localhost:8000")

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"

    def test_send_message_streams(self) -> None:
        """Send one message, confirm SSE chunks arrive."""
        import httpx

        payload = {
            "user_id": "smoke001",
            "message": "Suggest a casual outfit for a weekend brunch.",
            "history": [],
            "explain": False,
        }

        try:
            with (
                httpx.Client(timeout=30.0) as client,
                client.stream("POST", "http://localhost:8000/chat", json=payload) as response,
            ):
                if response.status_code != 200:
                    pytest.skip(f"Server returned {response.status_code}")
                chunks = []
                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunks.append(data)
        except httpx.ConnectError:
            pytest.skip("No server running at localhost:8000")

        assert len(chunks) > 0, "Expected at least one SSE chunk"

    def test_persona_endpoint(self) -> None:
        """Verify /persona/{user_id} returns a valid snapshot."""
        import httpx

        try:
            resp = httpx.get("http://localhost:8000/persona/smoke001", timeout=5.0)
        except httpx.ConnectError:
            pytest.skip("No server running at localhost:8000")

        assert resp.status_code == 200
        data = resp.json()
        assert "preferred_aesthetics" in data
        assert "confidence_score" in data
