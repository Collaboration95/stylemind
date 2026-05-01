from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProductRecord:
    """Raw product row from the seed CSV, before graph insertion."""

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
class RetrievedProduct:
    """Product returned by vector search with expanded graph relationships."""

    product_id: str
    name: str
    description: str
    price_inr: int
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
class PersonaSignals:
    """Structured signals extracted from a single chat turn by the PersonaInferenceEngine.

    signal_strength reflects how much persona information was present in the turn
    (0.0 = generic message, 1.0 = highly specific style intent).
    sentiment_on_shown maps product_id → "positive" | "negative" | "neutral".
    """

    liked_aesthetics: list[str] = field(default_factory=list)
    disliked_materials: list[str] = field(default_factory=list)
    mentioned_occasions: list[str] = field(default_factory=list)
    budget_signal: str | None = None
    color_preferences: list[str] = field(default_factory=list)
    brand_mentions: list[str] = field(default_factory=list)
    sentiment_on_shown: dict[str, str] = field(default_factory=dict)
    signal_strength: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.signal_strength <= 1.0):
            raise ValueError(f"signal_strength must be in [0.0, 1.0], got {self.signal_strength}")
