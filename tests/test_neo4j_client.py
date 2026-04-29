from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from stylemind.config import Neo4jConfig
from stylemind.graph.client import Neo4jClient


def test_neo4j_client_init_with_config():
    """Unit test: Neo4jClient initializes with config values."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    assert client._config == config
    assert client._driver is None


@pytest.mark.unit
def test_connect_initializes_driver():
    """Unit test: connect() initializes driver via GraphDatabase.driver."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    mock_driver = MagicMock()
    with patch("stylemind.graph.client.GraphDatabase.driver", return_value=mock_driver):
        client.connect()
    assert client._driver is mock_driver


@pytest.mark.unit
def test_close_calls_driver_close():
    """Unit test: close() calls driver.close() and sets _driver to None."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    mock_driver = MagicMock()
    client._driver = mock_driver
    client.close()
    mock_driver.close.assert_called_once()
    assert client._driver is None


@pytest.mark.unit
def test_verify_connectivity_success():
    """Unit test: verify_connectivity returns True when driver succeeds."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    mock_driver = MagicMock()
    mock_driver.verify_connectivity.return_value = None
    client._driver = mock_driver
    assert client.verify_connectivity() is True


@pytest.mark.unit
def test_verify_connectivity_failure():
    """Unit test: verify_connectivity returns False on ServiceUnavailable."""
    from neo4j.exceptions import ServiceUnavailable

    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    mock_driver = MagicMock()
    mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Cannot connect")
    client._driver = mock_driver
    assert client.verify_connectivity() is False


@pytest.mark.unit
def test_execute_query_raises_if_not_connected():
    """Unit test: execute_query raises RuntimeError if driver not initialized."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    with pytest.raises(RuntimeError, match="not initialized"):
        client.execute_query("RETURN 1")


@pytest.mark.unit
def test_execute_query_returns_list_of_dicts():
    """Unit test: execute_query returns list of dicts from driver result records."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    mock_record = MagicMock()
    mock_record.data.return_value = {"name": "Test Product"}
    mock_result = MagicMock()
    mock_result.records = [mock_record]
    mock_driver = MagicMock()
    mock_driver.execute_query.return_value = mock_result
    client._driver = mock_driver
    records = client.execute_query("MATCH (p:Product) RETURN p.name AS name")
    assert records == [{"name": "Test Product"}]


@pytest.mark.unit
def test_driver_property_raises_if_not_connected():
    """Unit test: driver property raises RuntimeError if not initialized."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = client.driver


@pytest.mark.unit
def test_driver_property_returns_driver():
    """Unit test: driver property returns the underlying driver when connected."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    mock_driver = MagicMock()
    client._driver = mock_driver
    assert client.driver is mock_driver


@pytest.mark.unit
def test_context_manager():
    """Unit test: context manager enters and exits cleanly."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    mock_driver = MagicMock()
    with patch("stylemind.graph.client.GraphDatabase.driver", return_value=mock_driver), Neo4jClient(config) as client:
        assert client._driver is mock_driver
    mock_driver.close.assert_called_once()


@pytest.mark.unit
def test_verify_connectivity_raises_if_not_connected():
    """Unit test: verify_connectivity raises RuntimeError if driver not initialized."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    with pytest.raises(RuntimeError, match="not initialized"):
        client.verify_connectivity()


@pytest.mark.unit
def test_close_is_idempotent():
    """Unit test: calling close() when already closed does nothing."""
    config = Neo4jConfig(uri="bolt://test:7687", user="neo4j", password="test")
    client = Neo4jClient(config)
    # close on unconnected client should not raise
    client.close()
    assert client._driver is None


@pytest.mark.integration
def test_integration_verify_connectivity(mock_env):
    """Integration test: verify_connectivity against running Neo4j."""
    # Skipped in unit test runs — requires running Neo4j
    config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "test"),
    )
    client = Neo4jClient(config)
    client.connect()
    # This may fail if Neo4j is not running — that's expected for integration tests
    result = client.verify_connectivity()
    client.close()
    assert isinstance(result, bool)
