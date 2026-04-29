from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from stylemind.config import Neo4jConfig, get_config

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Wraps the official neo4j driver with lifecycle management and DI support."""

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: Driver | None = None

    def connect(self) -> None:
        """Initialize the driver. Call once at application startup."""
        self._driver = GraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
        )
        logger.info("neo4j driver initialized uri=%s user=%s", self._config.uri, self._config.user)

    def close(self) -> None:
        """Close the driver. Call at application shutdown."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("neo4j driver closed")

    def verify_connectivity(self) -> bool:
        """Health check — returns True if Neo4j is reachable."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        try:
            self._driver.verify_connectivity()
            logger.info("neo4j connectivity verified uri=%s", self._config.uri)
            return True
        except ServiceUnavailable as exc:
            logger.error("neo4j connectivity failed error=%s", exc)
            return False

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as list of dicts."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        parameters = parameters or {}
        logger.debug("neo4j execute_query query_preview=%s param_count=%d", query[:80], len(parameters))
        result = self._driver.execute_query(query, parameters, database_=database)  # type: ignore[arg-type]
        records = [record.data() for record in result.records]
        logger.debug("neo4j query returned row_count=%d", len(records))
        return records

    @property
    def driver(self) -> Driver:
        """Direct driver access for advanced use cases (e.g. neo4j-graphrag)."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self._driver

    def __enter__(self) -> Neo4jClient:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def get_neo4j_client(request: Request) -> Neo4jClient:
    """FastAPI dependency: returns Neo4jClient stored on app.state."""
    return request.app.state.neo4j  # type: ignore[no-any-return]


def get_client_from_env() -> Neo4jClient:
    """Convenience factory for scripts and CLI: builds a client from environment config."""
    return Neo4jClient(get_config().neo4j)
