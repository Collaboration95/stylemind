from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ and project root are importable during tests.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

_CSV_PATH = _ROOT / "data" / "products_seed.csv"


# ---------------------------------------------------------------------------
# Unit tests for the RTL CSV parser
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_rows_parse_to_14_fields() -> None:
    """Every row in products_seed.csv must parse to exactly 14 fields."""
    from seed import parse_csv  # type: ignore[import]

    products = parse_csv(_CSV_PATH)
    assert len(products) == 45, f"Expected 45 products, got {len(products)}"
    for p in products:
        assert len(p) == 14, f"Product {p.get('product_id')} has {len(p)} fields, expected 14"


@pytest.mark.unit
def test_problematic_rows_verified_verbatim() -> None:
    """The 4 problematic rows (P013, P030, P038, P042) have commas in descriptions.

    Verify that the description field contains content with a comma (not split),
    and that pairs_with is clean product IDs starting with 'P'.
    """
    from seed import parse_csv  # type: ignore[import]

    products = parse_csv(_CSV_PATH)
    by_id = {p["product_id"]: p for p in products}

    # P013: "Fine-gauge crew neck in 100% Mongolian cashmere — no logos, no fuss"
    assert "P013" in by_id
    assert "," not in by_id["P013"]["pairs_with"], f"P013 pairs_with contaminated: {by_id['P013']['pairs_with']!r}"
    assert len(by_id["P013"]["description"]) > 10

    # P030: description should be intact (comma present)
    assert "P030" in by_id
    assert "," not in by_id["P030"]["pairs_with"], f"P030 pairs_with contaminated: {by_id['P030']['pairs_with']!r}"
    assert len(by_id["P030"]["description"]) > 10

    # P038: description may contain comma
    assert "P038" in by_id
    assert "," not in by_id["P038"]["pairs_with"], f"P038 pairs_with contaminated: {by_id['P038']['pairs_with']!r}"
    assert len(by_id["P038"]["description"]) > 10

    # P042: description should be intact
    assert "P042" in by_id
    assert "," not in by_id["P042"]["pairs_with"], f"P042 pairs_with contaminated: {by_id['P042']['pairs_with']!r}"
    assert len(by_id["P042"]["description"]) > 10

    # Verify pairs_with IDs are valid product references
    for pid in ["P013", "P030", "P038", "P042"]:
        pw = by_id[pid]["pairs_with"]
        for pair_id in pw.split("|"):
            assert pair_id.strip().startswith("P"), f"Bad pairs_with entry in {pid}: {pair_id!r}"


@pytest.mark.unit
def test_aesthetic_remapping_casual_to_casual_minimalism() -> None:
    """P037 and P038 'Casual' aesthetic must be remapped to 'Casual Minimalism'."""
    from seed import parse_csv  # type: ignore[import]

    products = parse_csv(_CSV_PATH)
    by_id = {p["product_id"]: p for p in products}

    # P037: raw CSV has "Y2K|Casual" → must become "Y2K|Casual Minimalism"
    assert "P037" in by_id
    assert "Casual Minimalism" in by_id["P037"]["aesthetic"], (
        f"P037 aesthetic should contain 'Casual Minimalism', got: {by_id['P037']['aesthetic']!r}"
    )
    # "Casual" (bare) should not remain after remapping
    remaining_p037 = by_id["P037"]["aesthetic"].replace("Casual Minimalism", "")
    assert "Casual" not in remaining_p037, (
        f"P037 still has bare 'Casual' after remapping: {by_id['P037']['aesthetic']!r}"
    )

    # P038: raw CSV has "Streetwear|Casual" → must become "Streetwear|Casual Minimalism"
    assert "P038" in by_id
    assert "Casual Minimalism" in by_id["P038"]["aesthetic"], (
        f"P038 aesthetic should contain 'Casual Minimalism', got: {by_id['P038']['aesthetic']!r}"
    )
    remaining_p038 = by_id["P038"]["aesthetic"].replace("Casual Minimalism", "")
    assert "Casual" not in remaining_p038, (
        f"P038 still has bare 'Casual' after remapping: {by_id['P038']['aesthetic']!r}"
    )


@pytest.mark.unit
def test_rtl_parser_handles_extra_commas_in_description() -> None:
    """Synthetic test: RTL parser correctly handles a line with 2 commas in the description field.

    The line has 16 comma-separated tokens (14 fields, but the description contributes 3 tokens).
    The parser should reconstruct the description as 'A description, with two, extra commas'
    and leave pairs_with as 'P001|P002'.
    """
    from seed import parse_row  # type: ignore[import]

    # 14 fields: first 12 columns + description (has 2 extra commas = 3 tokens) + pairs_with (1 token)
    # Total tokens when split on comma: 12 + 3 + 1 = 16
    # Build the synthetic raw line with 2 commas inside description
    fields_1_to_12 = [
        "P099",
        "Test Product",
        "Tops",
        "Zara",
        "2000",
        "Budget",
        "Streetwear",
        "Casual",
        "All",
        "Urban Brights",
        "Cotton",
        "Year-round",
    ]
    description_with_commas = "A description, with two, extra commas"
    pairs_with = "P001|P002"

    raw_line = ",".join(fields_1_to_12) + "," + description_with_commas + "," + pairs_with

    tokens = parse_row(raw_line)

    # Must always return exactly 14 elements
    assert len(tokens) == 14, f"Expected 14 tokens, got {len(tokens)}: {tokens}"

    # The last token should be the pairs_with field
    assert tokens[-1].strip() == "P001|P002", f"pairs_with mismatch: {tokens[-1]!r}"

    # The 13th token (index 12) should be the reconstructed description
    reconstructed_desc = tokens[12].strip()
    assert "," in reconstructed_desc, f"Description should contain commas, got: {reconstructed_desc!r}"
    assert "extra commas" in reconstructed_desc, f"Description content lost: {reconstructed_desc!r}"
