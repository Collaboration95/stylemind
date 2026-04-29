from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from stylemind.api import chat as chat_router
from stylemind.api import health as health_router
from stylemind.api import persona as persona_router
from stylemind.config import get_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown lifecycle."""
    config = get_config()
    logging.basicConfig(level=config.settings.log_level)

    # --- Startup ---

    # 1. Neo4j client
    from stylemind.graph.client import Neo4jClient

    neo4j_client = Neo4jClient(config.neo4j)
    neo4j_client.connect()
    app.state.neo4j = neo4j_client
    logger.info("lifespan neo4j connected uri=%s", config.neo4j.uri)

    # 2. Embedder
    from stylemind.rag.embedder import LocalEmbedder

    embedder = LocalEmbedder(model_name=config.embedding.model_name)
    app.state.embedder = embedder
    logger.info("lifespan embedder initialised model=%s", config.embedding.model_name)

    # 3. ProductRetriever
    from stylemind.rag.retriever import ProductRetriever

    retriever = ProductRetriever(
        client=neo4j_client,
        embedder=embedder,
        top_k=config.settings.vector_top_k,
        min_threshold=config.settings.min_similarity_threshold,
    )
    app.state.retriever = retriever

    # 4. PersonaAwareReranker
    from stylemind.rag.reranker import ProductReranker

    reranker = ProductReranker()
    app.state.reranker = reranker

    # 5. StyleMindGenerator (chat LLM)
    from stylemind.rag.generator import StyleMindGenerator

    generator = StyleMindGenerator(config.chat_llm)
    app.state.generator = generator

    # 6. PersonaManager
    from stylemind.persona.manager import PersonaManager

    persona_manager = PersonaManager(
        driver=neo4j_client.driver,
        decay_rate=config.settings.persona_decay_rate,
        expected_signals_per_turn=config.settings.expected_signals_per_turn,
    )
    app.state.persona_manager = persona_manager

    # 7. PersonaInferenceEngine (extraction LLM)
    from stylemind.persona.inference import PersonaInferenceEngine

    inference_engine = PersonaInferenceEngine(config.extraction_llm)
    app.state.inference_engine = inference_engine

    # 8. OutfitBuilder
    from stylemind.outfit.builder import OutfitBuilder

    outfit_builder = OutfitBuilder(driver=neo4j_client.driver)
    app.state.outfit_builder = outfit_builder

    logger.info("lifespan startup complete")

    yield

    # --- Shutdown ---
    neo4j_client.close()
    logger.info("lifespan shutdown complete")


def create_app() -> FastAPI:
    """Factory function: creates and configures the FastAPI application."""
    app = FastAPI(
        title="StyleMind",
        description="RAG-powered fashion styling assistant with Neo4j knowledge graph",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS (permissive for dev; tighten in prod via env)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request-ID logging middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: object) -> Response:
        request_id = str(uuid.uuid4())
        start = time.monotonic()
        logger.info(
            "request started request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        response: Response = await call_next(request)  # type: ignore[operator]
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "request finished request_id=%s status=%d duration_ms=%.1f",
            request_id,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # Routers
    app.include_router(chat_router.router)
    app.include_router(persona_router.router)
    app.include_router(health_router.router)

    return app


# Module-level app instance for uvicorn / __main__
app = create_app()
