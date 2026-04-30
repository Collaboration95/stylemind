from __future__ import annotations

import pytest

from stylemind.config import _reset_config, get_config


@pytest.mark.unit
class TestGetRequiredVariable:
    def test_missing_required_var_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
        monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            get_config()

    def test_missing_chat_api_key_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
        monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
        monkeypatch.delenv("CHAT_API_KEY", raising=False)

        with pytest.raises(ValueError, match="CHAT_API_KEY"):
            get_config()

    def test_missing_extraction_api_key_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
        monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
        monkeypatch.delenv("EXTRACTION_API_KEY", raising=False)

        with pytest.raises(ValueError, match="EXTRACTION_API_KEY"):
            get_config()


@pytest.mark.unit
class TestConfigLoadsFromEnv:
    def test_config_loads_required_and_optional_vars(self, mock_env: None) -> None:
        config = get_config()

        assert config.neo4j.uri == "bolt://localhost:7687"
        assert config.neo4j.user == "neo4j"
        assert config.neo4j.password == "test_password"
        assert config.chat_llm.api_key == "test-chat-key"
        assert config.chat_llm.base_url == "https://api.groq.com/openai/v1"
        assert config.chat_llm.model == "llama-3.3-70b-versatile"
        assert config.extraction_llm.api_key == "test-extraction-key"
        assert config.extraction_llm.base_url == "https://api.groq.com/openai/v1"
        assert config.extraction_llm.model == "llama-3.3-70b-versatile"
        assert config.embedding.provider == "local"
        assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding.dimensions == 384
        assert config.langfuse.public_key == "test-lf-public"
        assert config.langfuse.secret_key == "test-lf-secret"


@pytest.mark.unit
class TestOptionalVarsUseDefaults:
    def test_optional_vars_use_defaults_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
        monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
        monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
        # Remove optional vars to ensure defaults are used
        for var in [
            "NEO4J_URI",
            "NEO4J_USER",
            "CHAT_BASE_URL",
            "CHAT_MODEL",
            "CHAT_TEMPERATURE",
            "EXTRACTION_BASE_URL",
            "EXTRACTION_MODEL",
            "EMBEDDING_PROVIDER",
            "EMBEDDING_MODEL",
            "EMBEDDING_DIMENSIONS",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_HOST",
            "LOG_LEVEL",
            "VECTOR_TOP_K",
            "PERSONA_DECAY_RATE",
            "EXPECTED_SIGNALS_PER_TURN",
            "MIN_SIMILARITY_THRESHOLD",
        ]:
            monkeypatch.delenv(var, raising=False)

        config = get_config()

        assert config.neo4j.uri == "bolt://localhost:7687"
        assert config.neo4j.user == "neo4j"
        assert config.chat_llm.base_url == "https://api.groq.com/openai/v1"
        assert config.chat_llm.model == "llama-3.3-70b-versatile"
        assert config.chat_llm.temperature == 0.7
        assert config.extraction_llm.base_url == "https://api.groq.com/openai/v1"
        assert config.extraction_llm.model == "llama-3.3-70b-versatile"
        assert config.embedding.provider == "local"
        assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding.dimensions == 384
        assert config.langfuse.public_key == ""
        assert config.langfuse.secret_key == ""
        assert config.langfuse.host == "http://localhost:3000"
        assert config.settings.log_level == "INFO"
        assert config.settings.vector_top_k == 10
        assert config.settings.persona_decay_rate == 0.15
        assert config.settings.expected_signals_per_turn == 3.0
        assert config.settings.min_similarity_threshold == 0.3


@pytest.mark.unit
class TestSingletonBehavior:
    def test_two_calls_return_same_object(self, mock_env: None) -> None:
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config_allows_fresh_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEO4J_PASSWORD", "first-password")
        monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
        monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")

        config1 = get_config()
        assert config1.neo4j.password == "first-password"

        _reset_config()
        monkeypatch.setenv("NEO4J_PASSWORD", "second-password")

        config2 = get_config()
        assert config2.neo4j.password == "second-password"
        assert config1 is not config2


@pytest.mark.unit
class TestTypeConversions:
    def test_temperature_is_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
        monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
        monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
        monkeypatch.setenv("CHAT_TEMPERATURE", "0.5")

        config = get_config()
        assert isinstance(config.chat_llm.temperature, float)
        assert config.chat_llm.temperature == 0.5

    def test_dimensions_is_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
        monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
        monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")

        config = get_config()
        assert isinstance(config.embedding.dimensions, int)
        assert config.embedding.dimensions == 768

    def test_vector_top_k_is_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
        monkeypatch.setenv("CHAT_API_KEY", "test-chat-key")
        monkeypatch.setenv("EXTRACTION_API_KEY", "test-extraction-key")
        monkeypatch.setenv("VECTOR_TOP_K", "20")

        config = get_config()
        assert isinstance(config.settings.vector_top_k, int)
        assert config.settings.vector_top_k == 20
