from __future__ import annotations

import pytest

from data.enrichment import (
    AESTHETIC_METADATA,
    BODY_TYPE_METADATA,
    BRAND_METADATA,
    BUDGET_TIER_RANGES,
    COLOR_PALETTE_METADATA,
    MATERIAL_METADATA,
    OCCASION_METADATA,
    SYNTHETIC_PRODUCTS,
)

REQUIRED_PRODUCT_FIELDS = [
    "product_id",
    "name",
    "category",
    "brand",
    "price_inr",
    "budget_tier",
    "aesthetic",
    "occasion",
    "body_type_fit",
    "color_palette",
    "material",
    "season",
    "description",
    "pairs_with",
]


@pytest.mark.unit
def test_aesthetic_metadata_has_9_aesthetics() -> None:
    assert len(AESTHETIC_METADATA) == 9


@pytest.mark.unit
def test_each_aesthetic_has_required_keys() -> None:
    for name, meta in AESTHETIC_METADATA.items():
        assert "name" in meta, f"{name} missing 'name'"
        assert "description" in meta, f"{name} missing 'description'"
        assert "keywords" in meta, f"{name} missing 'keywords'"
        assert len(meta["keywords"]) >= 3, f"{name} has fewer than 3 keywords"


@pytest.mark.unit
def test_all_12_brands_in_brand_metadata() -> None:
    expected_brands = {
        "COS",
        "Arket",
        "Massimo Dutti",
        "New Balance",
        "Mango",
        "Zara",
        "Uniqlo",
        "Toteme",
        "H&M",
        "Levi's",
        "FabIndia",
        "Auralee",
    }
    assert set(BRAND_METADATA.keys()) == expected_brands


@pytest.mark.unit
def test_synthetic_products_have_all_14_fields() -> None:
    for product in SYNTHETIC_PRODUCTS:
        for field in REQUIRED_PRODUCT_FIELDS:
            assert field in product, f"Product {product.get('product_id', '?')} missing field '{field}'"
        assert len(product) == 14, f"Product {product['product_id']} has {len(product)} fields, expected 14"


@pytest.mark.unit
def test_6_synthetic_products_3_fabindia_3_auralee() -> None:
    assert len(SYNTHETIC_PRODUCTS) == 6
    fabindia = [p for p in SYNTHETIC_PRODUCTS if p["brand"] == "FabIndia"]
    auralee = [p for p in SYNTHETIC_PRODUCTS if p["brand"] == "Auralee"]
    assert len(fabindia) == 3
    assert len(auralee) == 3


@pytest.mark.unit
def test_synthetic_product_ids_start_with_f_or_a() -> None:
    for product in SYNTHETIC_PRODUCTS:
        pid = product["product_id"]
        assert pid[0] in ("F", "A"), f"Product ID {pid} does not start with F or A"


@pytest.mark.unit
def test_material_parent_material_groups() -> None:
    assert MATERIAL_METADATA["Cotton Rib"]["parent_material"] == "Cotton"
    assert MATERIAL_METADATA["Full-grain Leather"]["parent_material"] == "Leather"
    assert MATERIAL_METADATA["Suede"]["parent_material"] == "Leather"
    assert MATERIAL_METADATA["Tweed"]["parent_material"] == "Wool"
    assert MATERIAL_METADATA["Cashmere"]["parent_material"] == "Wool"
    assert MATERIAL_METADATA["Chiffon"]["parent_material"] == "Silk"
    assert MATERIAL_METADATA["Canvas"]["parent_material"] == "Cotton"
    assert MATERIAL_METADATA["Denim"]["parent_material"] == "Cotton"


@pytest.mark.unit
def test_all_6_occasions_with_valid_formality_score() -> None:
    expected_occasions = {"Casual", "Office", "Date Night", "Weekend Brunch", "Active", "Formal"}
    assert set(OCCASION_METADATA.keys()) == expected_occasions
    for name, meta in OCCASION_METADATA.items():
        score = meta["formality_score"]
        assert 1 <= score <= 5, f"Occasion '{name}' has formality_score {score} outside 1-5"


@pytest.mark.unit
def test_all_7_palettes_with_hex_codes() -> None:
    expected_palettes = {
        "Classic Neutrals",
        "Earthy Neutrals",
        "Soft Whites",
        "Sport Monochromes",
        "Urban Brights",
        "Warm Nudes",
        "Monochromes",
    }
    assert set(COLOR_PALETTE_METADATA.keys()) == expected_palettes
    for name, meta in COLOR_PALETTE_METADATA.items():
        assert "hex_codes" in meta, f"Palette '{name}' missing 'hex_codes'"
        assert isinstance(meta["hex_codes"], list), f"Palette '{name}' hex_codes is not a list"
        assert len(meta["hex_codes"]) > 0, f"Palette '{name}' has empty hex_codes"


@pytest.mark.unit
def test_all_6_body_types_in_metadata() -> None:
    expected_body_types = {"All", "Hourglass", "Athletic", "Rectangle", "Pear", "Inverted Triangle"}
    assert set(BODY_TYPE_METADATA.keys()) == expected_body_types


@pytest.mark.unit
def test_budget_tier_ranges_has_4_tiers() -> None:
    expected_tiers = {"Budget", "Mid", "Premium", "Luxury"}
    assert set(BUDGET_TIER_RANGES.keys()) == expected_tiers
    for tier, meta in BUDGET_TIER_RANGES.items():
        assert "min_inr" in meta, f"Tier '{tier}' missing 'min_inr'"
        assert "max_inr" in meta, f"Tier '{tier}' missing 'max_inr'"
        assert meta["min_inr"] < meta["max_inr"], f"Tier '{tier}' min_inr >= max_inr"
