from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return service health status, including Neo4j and embedder checks."""
    neo4j_ok = False
    embedder_ok = False

    # Check Neo4j connectivity
    neo4j_client = getattr(request.app.state, "neo4j", None)
    if neo4j_client is not None:
        try:
            neo4j_ok = neo4j_client.verify_connectivity()
        except Exception as exc:
            logger.warning("health neo4j check failed error=%s", exc)
            neo4j_ok = False

    # Check embedder initialisation
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is not None:
        embedder_ok = True

    healthy = neo4j_ok and embedder_ok
    http_status = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    payload = {"status": "ok" if healthy else "degraded", "neo4j": neo4j_ok, "embedder": embedder_ok}

    logger.info("health check status=%s neo4j=%s embedder=%s", payload["status"], neo4j_ok, embedder_ok)
    return JSONResponse(content=payload, status_code=http_status)
