from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator

import orjson
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from stylemind.models.schemas import ChatRequest, PersonaSnapshot
from stylemind.observability import langfuse_context, score_persona_confidence

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse_text(chunk: str) -> str:
    """Encode a text chunk as a valid SSE data event, handling embedded newlines.

    Per SSE spec, multiple data: fields within one event are concatenated with LF
    on the client side, preserving newlines without breaking SSE framing.
    """
    return "".join(f"data: {line}\n" for line in chunk.split("\n")) + "\n"


def _sse_json(payload: dict) -> str:
    """Encode a structured payload as a named SSE 'json' event."""
    return f"event: json\ndata: {orjson.dumps(payload).decode()}\n\n"


async def _sse_stream(
    request: Request,
    chat_request: ChatRequest,
) -> AsyncGenerator[str]:
    """Core SSE generator: retrieve → rerank → (outfit) → stream → fire-and-forget persona update."""
    turn_start = time.monotonic()
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

    # Convert HistoryMessage objects to plain dicts for downstream LLM clients
    history_dicts = [m.model_dump() for m in chat_request.history]

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

    # 2b. Score retrieval quality in Langfuse
    if retrieved_products and langfuse_context is not None:
        scores = [p.similarity_score for p in retrieved_products]
        mean_sim = sum(scores) / len(scores)
        max_sim = max(scores)
        with contextlib.suppress(Exception):
            langfuse_context.score_current_observation(name="retrieval_mean_similarity", value=mean_sim)
            langfuse_context.score_current_observation(name="retrieval_max_similarity", value=max_sim)
        logger.info(
            "chat retrieval_quality product_count=%d mean_sim=%.3f max_sim=%.3f",
            len(retrieved_products),
            mean_sim,
            max_sim,
        )

    # 3. Rerank with persona signals
    reranked_products = retrieved_products
    rerank_results: list = []
    if reranker is not None and retrieved_products:
        try:
            rerank_results = await asyncio.to_thread(reranker.rerank, retrieved_products, persona, chat_request.explain)
            reranked_products = [r.product for r in rerank_results]
        except Exception as exc:
            logger.warning("chat rerank failed user_id=%s error=%s", chat_request.user_id, exc)

    # 4. Detect product interest → conditionally build outfit.
    # Skip products explicitly disliked by the user — they should not anchor outfit building.
    outfit = None
    if generator is not None and outfit_builder is not None and reranked_products:
        try:
            disliked_ids = set(persona.disliked_products)
            interest_candidates = [p for p in reranked_products if p.product_id not in disliked_ids]
            matched_product_id = generator.detect_product_interest(chat_request.message, interest_candidates)
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
                history=history_dicts,
                retrieved_products=reranked_products,
                outfit=outfit,
                persona=persona_dict,
            ):
                yield _sse_text(chunk)
        except Exception as exc:
            logger.error("chat stream_response failed user_id=%s error=%s", chat_request.user_id, exc)
            yield _sse_text("Sorry, an error occurred while generating the response.")
    else:
        yield _sse_text("StyleMind generator not available.")

    # 6. Emit structured JSON events (before [DONE]).
    # Uses named SSE event type 'json' to avoid collision with LLM text output.
    if reranked_products:
        # Use final_score from rerank results when available, fall back to similarity_score
        score_by_id = {r.product.product_id: r.final_score for r in rerank_results} if rerank_results else {}
        sources_payload = [
            {
                "product_id": p.product_id,
                "name": p.name,
                "brand": p.brand,
                "price_inr": p.price_inr,
                "score": score_by_id.get(p.product_id, p.similarity_score),
            }
            for p in reranked_products
        ]
        with contextlib.suppress(Exception):
            yield _sse_json({"sources": sources_payload})

    if chat_request.explain and rerank_results:
        explain_payload = [r.breakdown.to_dict() for r in rerank_results if r.breakdown is not None]
        if explain_payload:
            with contextlib.suppress(Exception):
                yield _sse_json({"explain": explain_payload})

    # 6b. Extract and emit persona signals BEFORE [DONE] so the CLI receives them
    extracted_signals = None
    if inference_engine is not None and persona_manager is not None:
        shown_product_ids = [p.product_id for p in reranked_products]

        try:
            extracted_signals = await asyncio.to_thread(
                inference_engine.extract_signals,
                chat_request.message,
                history_dicts,
                shown_product_ids,
            )

            signals_payload = {
                "signals": {
                    "liked_aesthetics": extracted_signals.liked_aesthetics,
                    "disliked_materials": extracted_signals.disliked_materials,
                    "mentioned_occasions": extracted_signals.mentioned_occasions,
                    "budget_signal": extracted_signals.budget_signal,
                    "color_preferences": extracted_signals.color_preferences,
                    "brand_mentions": extracted_signals.brand_mentions,
                    "sentiment_on_shown": extracted_signals.sentiment_on_shown,
                    "signal_strength": extracted_signals.signal_strength,
                }
            }
            with contextlib.suppress(Exception):
                yield _sse_json(signals_payload)
        except Exception as exc:
            logger.warning("chat persona extraction failed user_id=%s error=%s", chat_request.user_id, exc)

    yield "data: [DONE]\n\n"

    # 7. Score response latency in Langfuse
    turn_elapsed_ms = (time.monotonic() - turn_start) * 1000
    if langfuse_context is not None:
        with contextlib.suppress(Exception):
            langfuse_context.score_current_observation(name="response_latency_ms", value=turn_elapsed_ms)
    logger.info("chat response_latency_ms=%.1f user_id=%s", turn_elapsed_ms, chat_request.user_id)

    # 8. Fire-and-forget persona persistence (after [DONE] — only DB write, no client output)
    if extracted_signals is not None and persona_manager is not None:

        async def _persist_persona() -> None:
            try:
                await asyncio.to_thread(persona_manager.update_persona, chat_request.user_id, extracted_signals)
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
