from __future__ import annotations

import asyncio
import contextlib
import json
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
            persona = await asyncio.to_thread(persona_manager.get_persona, chat_request.user_id)
        except Exception as exc:
            logger.warning("chat get_persona failed user_id=%s error=%s", chat_request.user_id, exc)

    # 2. Retrieve candidate products
    retrieved_products = []
    if retriever is not None:
        try:
            retrieved_products = await asyncio.to_thread(retriever.retrieve, chat_request.message)
        except Exception as exc:
            logger.warning("chat retrieve failed user_id=%s error=%s", chat_request.user_id, exc)

    # 3. Rerank with persona signals
    reranked_products = retrieved_products
    rerank_results: list = []
    if reranker is not None and retrieved_products:
        try:
            rerank_results = await asyncio.to_thread(reranker.rerank, retrieved_products, persona, chat_request.explain)
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
                    outfit = await asyncio.to_thread(
                        outfit_builder.build_outfit,
                        matched_product_id,
                        chat_request.user_id,
                        persona,
                    )
                    logger.info("chat outfit built product_id=%s user_id=%s", matched_product_id, chat_request.user_id)
                except Exception as exc:
                    logger.warning("chat outfit build failed product_id=%s error=%s", matched_product_id, exc)
        except Exception as exc:
            logger.warning("chat detect_product_interest failed user_id=%s error=%s", chat_request.user_id, exc)

    # 5. Stream LLM response (with persona context for personalized tone)
    persona_dict = persona.model_dump() if persona.confidence_score > 0.0 else None
    if generator is not None:
        try:
            async for chunk in generator.stream_response(
                message=chat_request.message,
                history=chat_request.history,
                retrieved_products=reranked_products,
                outfit=outfit,
                persona=persona_dict,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as exc:
            logger.error("chat stream_response failed user_id=%s error=%s", chat_request.user_id, exc)
            yield "data: Sorry, an error occurred while generating the response.\n\n"
    else:
        yield "data: StyleMind generator not available.\n\n"

    # 6. Emit structured JSON events (before [DONE])
    if reranked_products:
        sources_payload = [
            {
                "product_id": p.product_id,
                "name": p.name,
                "brand": p.brand,
                "price_inr": p.price_inr,
                "score": p.similarity_score,
            }
            for p in reranked_products
        ]
        yield f"data: __JSON__{json.dumps({'sources': sources_payload})}\n\n"

    if chat_request.explain and rerank_results:
        explain_payload = [r.breakdown.to_dict() for r in rerank_results if r.breakdown is not None]
        if explain_payload:
            yield f"data: __JSON__{json.dumps({'explain': explain_payload})}\n\n"

    yield "data: [DONE]\n\n"

    # 7. Persona update — extract signals, persist, and return signals to client for /debug-dev
    if inference_engine is not None and persona_manager is not None:
        shown_product_ids = [p.product_id for p in reranked_products]

        try:
            signals = await asyncio.to_thread(
                inference_engine.extract_signals,
                chat_request.message,
                chat_request.history,
                shown_product_ids,
            )

            signals_payload = {
                "signals": {
                    "liked_aesthetics": signals.liked_aesthetics,
                    "disliked_materials": signals.disliked_materials,
                    "mentioned_occasions": signals.mentioned_occasions,
                    "budget_signal": signals.budget_signal,
                    "color_preferences": signals.color_preferences,
                    "brand_mentions": signals.brand_mentions,
                    "sentiment_on_shown": signals.sentiment_on_shown,
                    "signal_strength": signals.signal_strength,
                }
            }
            yield f"data: __JSON__{json.dumps(signals_payload)}\n\n"

            async def _persist_persona() -> None:
                try:
                    await asyncio.to_thread(persona_manager.update_persona, chat_request.user_id, signals)
                    logger.info("chat persona updated user_id=%s", chat_request.user_id)
                    updated_persona = await asyncio.to_thread(persona_manager.get_persona, chat_request.user_id)
                    score_persona_confidence(
                        user_id=chat_request.user_id,
                        confidence=updated_persona.confidence_score,
                        session_id=chat_request.user_id,
                    )
                except Exception as exc:
                    logger.warning("chat persona persist failed user_id=%s error=%s", chat_request.user_id, exc)

            asyncio.create_task(_persist_persona())
        except Exception as exc:
            logger.warning("chat persona extraction failed user_id=%s error=%s", chat_request.user_id, exc)


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
