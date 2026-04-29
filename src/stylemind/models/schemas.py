from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)
    explain: bool = False


class ProductSummary(BaseModel):
    product_id: str
    name: str
    brand: str
    category: str
    price_inr: int
    aesthetics: list[str] = Field(default_factory=list)
    score: float = 0.0


class OutfitItemSchema(BaseModel):
    product_id: str
    name: str
    category: str
    brand: str
    price_inr: int
    justification: str
    graph_path: str


class OutfitSuggestion(BaseModel):
    anchor: ProductSummary
    items: list[OutfitItemSchema]
    occasion: str
    season: str


class ChatResponse(BaseModel):
    user_id: str
    response: str
    sources: list[ProductSummary] = Field(default_factory=list)
    outfit: OutfitSuggestion | None = None


class PersonaSnapshot(BaseModel):
    """Matches spec R5 exactly."""

    preferred_aesthetics: list[str] = Field(default_factory=list)
    disliked_materials: list[str] = Field(default_factory=list)
    budget_tier: str | None = None
    top_occasions: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0


class PersonaSignals(BaseModel):
    liked_aesthetics: list[str] = Field(default_factory=list)
    disliked_materials: list[str] = Field(default_factory=list)
    mentioned_occasions: list[str] = Field(default_factory=list)
    budget_signal: str | None = None  # "budget", "mid", "premium", "luxury"
    color_preferences: list[str] = Field(default_factory=list)
    brand_mentions: list[str] = Field(default_factory=list)
    sentiment_on_shown: dict[str, str] = Field(default_factory=dict)  # product_id -> "positive"/"negative"
    signal_strength: float = 0.5
