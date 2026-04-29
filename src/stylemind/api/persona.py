from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from stylemind.models.schemas import PersonaSnapshot

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/persona/{user_id}", response_model=PersonaSnapshot)
async def get_persona(user_id: str, request: Request) -> PersonaSnapshot:
    """Return the current persona snapshot for a user.

    Returns an empty default PersonaSnapshot when the user has no persona data yet.
    """
    persona_manager = getattr(request.app.state, "persona_manager", None)
    if persona_manager is None:
        logger.warning("persona get_persona persona_manager not initialised, returning default user_id=%s", user_id)
        return PersonaSnapshot()

    try:
        snapshot = persona_manager.get_persona(user_id)
        logger.info("persona get_persona user_id=%s confidence=%.2f", user_id, snapshot.confidence_score)
        return snapshot
    except Exception as exc:
        logger.warning("persona get_persona failed user_id=%s error=%s, returning default", user_id, exc)
        return PersonaSnapshot()
