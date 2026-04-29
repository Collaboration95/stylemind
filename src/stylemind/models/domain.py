from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProductRecord:
    product_id: str
    name: str
    category: str
    brand: str
    price_inr: int
    budget_tier: str
    aesthetic: str
    occasion: str  # pipe-separated e.g. "Office|Date Night"
    body_type_fit: str  # pipe-separated e.g. "Hourglass|Rectangle"
    color_palette: str
    material: str
    season: str  # pipe-separated e.g. "SS|Year-round"
    description: str
    pairs_with: str  # pipe-separated product IDs


@dataclass(frozen=True)
class BrandRecord:
    name: str
    tier: str
    country_of_origin: str


@dataclass(frozen=True)
class AestheticRecord:
    name: str
    description: str
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedProduct:
    product_id: str
    name: str
    description: str
    price: int
    category: str
    brand: str
    budget_tier: str
    aesthetics: list[str] = field(default_factory=list)
    occasions: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    seasons: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    pairs_with: list[str] = field(default_factory=list)
    similarity_score: float = 0.0


@dataclass(frozen=True)
class OutfitItem:
    product_id: str
    name: str
    category: str
    brand: str
    price_inr: int
    justification: str
    graph_path: str


@dataclass(frozen=True)
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class ConversationState:
    user_id: str
    history: list[ConversationTurn] = field(default_factory=list)
    turn_count: int = 0
