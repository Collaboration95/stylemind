from __future__ import annotations

import pytest

from stylemind.models.domain import ConversationState, ConversationTurn, ProductRecord
from stylemind.models.enums import Aesthetic, BodyType, BudgetTier, Category, ColorPalette, Occasion, Season
from stylemind.models.schemas import PersonaSignals, PersonaSnapshot

# ---------------------------------------------------------------------------
# Enum exhaustiveness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_budget_tier_values() -> None:
    assert len(BudgetTier) == 4
    assert BudgetTier.BUDGET == "Budget"
    assert BudgetTier.MID == "Mid"
    assert BudgetTier.PREMIUM == "Premium"
    assert BudgetTier.LUXURY == "Luxury"


@pytest.mark.unit
def test_occasion_values() -> None:
    assert len(Occasion) == 6
    assert Occasion.CASUAL == "Casual"
    assert Occasion.OFFICE == "Office"
    assert Occasion.DATE_NIGHT == "Date Night"
    assert Occasion.WEEKEND_BRUNCH == "Weekend Brunch"
    assert Occasion.ACTIVE == "Active"
    assert Occasion.FORMAL == "Formal"


@pytest.mark.unit
def test_aesthetic_values() -> None:
    assert len(Aesthetic) == 9
    assert Aesthetic.OLD_MONEY == "Old Money"
    assert Aesthetic.QUIET_LUXURY == "Quiet Luxury"
    assert Aesthetic.COASTAL_GRANDMA == "Coastal Grandma"
    assert Aesthetic.COTTAGECORE == "Cottagecore"
    assert Aesthetic.CORPORATE_MINIMALISM == "Corporate Minimalism"
    assert Aesthetic.STREETWEAR == "Streetwear"
    assert Aesthetic.ATHLEISURE == "Athleisure"
    assert Aesthetic.Y2K == "Y2K"
    assert Aesthetic.CASUAL_MINIMALISM == "Casual Minimalism"


@pytest.mark.unit
def test_season_values() -> None:
    assert len(Season) == 3
    assert Season.SS == "SS"
    assert Season.AW == "AW"
    assert Season.YEAR_ROUND == "Year-round"


@pytest.mark.unit
def test_category_values() -> None:
    assert len(Category) == 7
    assert Category.TOPS == "Tops"
    assert Category.BOTTOMS == "Bottoms"
    assert Category.FOOTWEAR == "Footwear"
    assert Category.OUTERWEAR == "Outerwear"
    assert Category.DRESSES == "Dresses"
    assert Category.BAGS == "Bags"
    assert Category.ACCESSORIES == "Accessories"


@pytest.mark.unit
def test_body_type_values() -> None:
    assert len(BodyType) == 6
    assert BodyType.ALL == "All"
    assert BodyType.HOURGLASS == "Hourglass"
    assert BodyType.ATHLETIC == "Athletic"
    assert BodyType.RECTANGLE == "Rectangle"
    assert BodyType.PEAR == "Pear"
    assert BodyType.INVERTED_TRIANGLE == "Inverted Triangle"


@pytest.mark.unit
def test_color_palette_values() -> None:
    assert len(ColorPalette) == 7
    assert ColorPalette.CLASSIC_NEUTRALS == "Classic Neutrals"
    assert ColorPalette.EARTHY_NEUTRALS == "Earthy Neutrals"
    assert ColorPalette.SOFT_WHITES == "Soft Whites"
    assert ColorPalette.SPORT_MONOCHROMES == "Sport Monochromes"
    assert ColorPalette.URBAN_BRIGHTS == "Urban Brights"
    assert ColorPalette.WARM_NUDES == "Warm Nudes"
    assert ColorPalette.MONOCHROMES == "Monochromes"


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_product_record_is_immutable() -> None:
    record = ProductRecord(
        product_id="P001",
        name="Test Top",
        category="Tops",
        brand="TestBrand",
        price_inr=999,
        budget_tier="Budget",
        aesthetic="Streetwear",
        occasion="Casual",
        body_type_fit="All",
        color_palette="Classic Neutrals",
        material="Cotton",
        season="SS",
        description="A test top.",
        pairs_with="P002|P003",
    )
    with pytest.raises(AttributeError):
        record.name = "Modified"  # type: ignore[misc]


@pytest.mark.unit
def test_conversation_state_default_turn_count() -> None:
    state = ConversationState(user_id="user-123")
    assert state.turn_count == 0
    assert state.history == []


@pytest.mark.unit
def test_conversation_state_with_turns() -> None:
    turns = [
        ConversationTurn(role="user", content="Hello"),
        ConversationTurn(role="assistant", content="Hi there!"),
    ]
    state = ConversationState(user_id="user-456", history=turns, turn_count=1)
    assert state.turn_count == 1
    assert len(state.history) == 2
    assert state.history[0].role == "user"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_persona_snapshot_json_shape() -> None:
    snapshot = PersonaSnapshot()
    data = snapshot.model_dump()

    assert "preferred_aesthetics" in data
    assert "disliked_materials" in data
    assert "budget_tier" in data
    assert "top_occasions" in data
    assert "confidence_score" in data

    assert data["preferred_aesthetics"] == []
    assert data["disliked_materials"] == []
    assert data["budget_tier"] is None
    assert data["top_occasions"] == []
    assert data["confidence_score"] == 0.0


@pytest.mark.unit
def test_persona_signals_default_values_are_empty() -> None:
    signals = PersonaSignals()

    assert signals.liked_aesthetics == []
    assert signals.disliked_materials == []
    assert signals.mentioned_occasions == []
    assert signals.budget_signal is None
    assert signals.color_preferences == []
    assert signals.brand_mentions == []
    assert signals.sentiment_on_shown == {}
    assert signals.signal_strength == 0.5
