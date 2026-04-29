from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Incoming chat request with user message and optional history."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    message: str = Field(max_length=2000, min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)
    explain: bool = False


class ProductSummary(BaseModel):
    """Compact product representation for API responses."""

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
    """Complete outfit suggestion with anchor and paired items."""

    anchor: ProductSummary
    items: list[OutfitItemSchema]
    occasion: str
    season: str


class PersonaSnapshot(BaseModel):
    """Matches spec R5 exactly."""

    model_config = ConfigDict(extra="ignore")

    preferred_aesthetics: list[str] = Field(default_factory=list)
    disliked_materials: list[str] = Field(default_factory=list)
    disliked_products: list[str] = Field(default_factory=list)
    budget_tier: str | None = None
    top_occasions: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
