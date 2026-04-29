from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from stylemind.models.schemas import ChatRequest, PersonaSnapshot
from stylemind.observability import langfuse_context, score_persona_confidence

logger = logging.getLogger(__name__)

router = APIRouter()


async def _sse_stream(
    request: Request,
    chat_request: ChatRequest,
) -> AsyncGenerator[str]:
    """Core SSE generator: retrieve → rerank → (outfit) → stream → fire-and-forget persona update."""
    state = request.app.state

    retriever = getattr(state, "retriever", None)
    reranker = getattr(state, "reranker", None)
    generator = getattr(state, "generator", None)
    persona_manager = getattr(state, "persona_manager", None)
    inference_engine = getattr(state, "inference_engine", None)
    outfit_builder = getattr(state, "outfit_builder", None)

    # Set Langfuse session_id so all spans for this user are grouped together
    if langfuse_context is not None:
        with contextlib.suppress(Exception):
            langfuse_context.update_current_trace(session_id=chat_request.user_id, user_id=chat_request.user_id)

    # 1. Get current persona (returns empty default on first turn, never None)
    persona: PersonaSnapshot = PersonaSnapshot()
    if persona_manager is not None:
        try:
            persona = persona_manager.get_persona(chat_request.user_id)
        except Exception as exc:
            logger.warning("chat get_persona failed user_id=%s error=%s", chat_request.user_id, exc)

    # 2. Retrieve candidate products
    retrieved_products = []
    if retriever is not None:
        try:
            retrieved_products = retriever.retrieve(chat_request.message)
        except Exception as exc:
            logger.warning("chat retrieve failed user_id=%s error=%s", chat_request.user_id, exc)

    # 3. Rerank with persona signals
    reranked_products = retrieved_products
    if reranker is not None and retrieved_products:
        try:
            rerank_results = reranker.rerank(retrieved_products, persona, explain=chat_request.explain)
            reranked_products = [r.product for r in rerank_results]
        except Exception as exc:
            logger.warning("chat rerank failed user_id=%s error=%s", chat_request.user_id, exc)

    # 4. Detect product interest → conditionally build outfit
    outfit = None
    if generator is not None and outfit_builder is not None and reranked_products:
        try:
            matched_product_id = generator.detect_product_interest(chat_request.message, reranked_products)
            if matched_product_id:
                try:
                    outfit = outfit_builder.build_outfit(
                        product_id=matched_product_id,
                        user_id=chat_request.user_id,
                        persona=persona,
                    )
                    logger.info("chat outfit built product_id=%s user_id=%s", matched_product_id, chat_request.user_id)
                except Exception as exc:
                    logger.warning("chat outfit build failed product_id=%s error=%s", matched_product_id, exc)
        except Exception as exc:
            logger.warning("chat detect_product_interest failed user_id=%s error=%s", chat_request.user_id, exc)

    # 5. Stream LLM response
    if generator is not None:
        try:
            async for chunk in generator.stream_response(
                message=chat_request.message,
                history=chat_request.history,
                retrieved_products=reranked_products,
                outfit=outfit,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as exc:
            logger.error("chat stream_response failed user_id=%s error=%s", chat_request.user_id, exc)
            yield "data: Sorry, an error occurred while generating the response.\n\n"
    else:
        yield "data: StyleMind generator not available.\n\n"

    yield "data: [DONE]\n\n"

    # 6. Fire-and-forget async persona update (does NOT block response)
    if inference_engine is not None and persona_manager is not None:
        shown_product_ids = [p.product_id for p in reranked_products]

        async def _update_persona() -> None:
            try:
                signals = inference_engine.extract_signals(
                    message=chat_request.message,
                    history=chat_request.history,
                    shown_products=shown_product_ids,
                )
                persona_manager.update_persona(chat_request.user_id, signals)
                logger.info("chat persona updated user_id=%s", chat_request.user_id)

                # Log persona confidence as a custom Langfuse score
                updated_persona = persona_manager.get_persona(chat_request.user_id)
                score_persona_confidence(
                    user_id=chat_request.user_id,
                    confidence=updated_persona.confidence_score,
                    session_id=chat_request.user_id,
                )
            except Exception as exc:
                logger.warning("chat persona update failed user_id=%s error=%s", chat_request.user_id, exc)

        asyncio.ensure_future(_update_persona())


@router.post("/chat")
async def chat(chat_request: ChatRequest, request: Request) -> StreamingResponse:
    """Stream a styled fashion recommendation response via Server-Sent Events."""
    logger.info("chat request user_id=%s message_preview=%s", chat_request.user_id, chat_request.message[:80])
    return StreamingResponse(
        _sse_stream(request, chat_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
