from __future__ import annotations

_config = None


def _reset_config() -> None:
    global _config
    _config = None
