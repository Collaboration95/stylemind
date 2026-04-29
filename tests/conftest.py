from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_config() -> Generator[None, None, None]:
    from stylemind import config as cfg_module

    cfg_module._reset_config()
    yield
    cfg_module._reset_config()


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
    monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
    monkeypatch.setenv("CHAT_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("EXTRACTION_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CHAT_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-4.1-nano")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-lf-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-lf-secret")


@pytest.fixture
def mock_driver() -> MagicMock:
    driver = MagicMock()
    driver.verify_connectivity = MagicMock()
    driver.execute_query = MagicMock(return_value=MagicMock(records=[]))
    driver.close = MagicMock()
    return driver


@pytest.fixture
def mock_async_driver() -> AsyncMock:
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock(return_value=AsyncMock(records=[]))
    driver.close = AsyncMock()
    return driver
