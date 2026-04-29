from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from stylemind.models.schemas import OutfitSuggestion, PersonaSnapshot

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/outfit/{product_id}", response_model=OutfitSuggestion)
async def build_outfit(product_id: str, request: Request, user_id: str = "anonymous") -> OutfitSuggestion:
    """Build a coherent outfit around the given anchor product."""
    outfit_builder = getattr(request.app.state, "outfit_builder", None)
    persona_manager = getattr(request.app.state, "persona_manager", None)

    if outfit_builder is None:
        raise HTTPException(status_code=503, detail="Outfit builder not available")

    persona = PersonaSnapshot()
    if persona_manager is not None:
        try:
            persona = await asyncio.to_thread(persona_manager.get_persona, user_id)
        except Exception as exc:
            logger.warning("outfit get_persona failed user_id=%s error=%s", user_id, exc)

    try:
        outfit = await asyncio.to_thread(outfit_builder.build_outfit, product_id, user_id, persona)
        logger.info("outfit built product_id=%s user_id=%s items=%d", product_id, user_id, len(outfit.items))
        return outfit
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("outfit build failed product_id=%s error=%s", product_id, exc)
        raise HTTPException(status_code=500, detail="Failed to build outfit") from exc
