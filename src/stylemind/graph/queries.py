from __future__ import annotations

# ---------------------------------------------------------------------------
# Product nodes
# ---------------------------------------------------------------------------

MERGE_PRODUCT = """
MERGE (p:Product {product_id: $product_id})
SET p.name = $name,
    p.description = $description,
    p.price_inr = $price_inr,
    p.category = $category,
    p.color_palette = $color_palette,
    p.material = $material,
    p.season = $season
"""

# ---------------------------------------------------------------------------
# Taxonomy nodes
# ---------------------------------------------------------------------------

MERGE_BRAND = """
MERGE (b:Brand {name: $name})
SET b.tier = $tier,
    b.country_of_origin = $country_of_origin
"""

MERGE_AESTHETIC = """
MERGE (a:Aesthetic {name: $name})
SET a.description = $description,
    a.keywords = $keywords
"""

MERGE_OCCASION = """
MERGE (o:Occasion {name: $name})
SET o.formality_score = $formality_score
"""

MERGE_BODY_TYPE = """
MERGE (bt:BodyType {name: $name})
SET bt.description = $description
"""

MERGE_COLOR_PALETTE = """
MERGE (cp:ColorPalette {name: $name})
SET cp.hex_codes = $hex_codes
"""

MERGE_MATERIAL = """
MERGE (m:Material {name: $name})
SET m.parent_material = $parent_material,
    m.sustainability_score = $sustainability_score,
    m.feel_tag = $feel_tag
"""

MERGE_SEASON = """
MERGE (s:Season {name: $name})
"""

MERGE_BUDGET_TIER = """
MERGE (bt:BudgetTier {label: $label})
SET bt.min_inr = $min_inr,
    bt.max_inr = $max_inr
"""

# ---------------------------------------------------------------------------
# Product → taxonomy relationships
# ---------------------------------------------------------------------------

REL_PRODUCT_BELONGS_TO_BRAND = """
MATCH (p:Product {product_id: $product_id})
MATCH (b:Brand {name: $brand_name})
MERGE (p)-[:BELONGS_TO]->(b)
"""

REL_PRODUCT_FITS_OCCASION = """
MATCH (p:Product {product_id: $product_id})
MATCH (o:Occasion {name: $occasion_name})
MERGE (p)-[:FITS_OCCASION]->(o)
"""

REL_PRODUCT_EMBODIES_AESTHETIC = """
MATCH (p:Product {product_id: $product_id})
MATCH (a:Aesthetic {name: $aesthetic_name})
MERGE (p)-[:EMBODIES]->(a)
"""

REL_PRODUCT_SUITS_BODY = """
MATCH (p:Product {product_id: $product_id})
MATCH (bt:BodyType {name: $body_type_name})
MERGE (p)-[:SUITS_BODY]->(bt)
"""

REL_PRODUCT_MADE_FROM = """
MATCH (p:Product {product_id: $product_id})
MATCH (m:Material {name: $material_name})
MERGE (p)-[:MADE_FROM]->(m)
"""

REL_PRODUCT_BEST_IN_SEASON = """
MATCH (p:Product {product_id: $product_id})
MATCH (s:Season {name: $season_name})
MERGE (p)-[:BEST_IN_SEASON]->(s)
"""

REL_PRODUCT_IN_COLOR = """
MATCH (p:Product {product_id: $product_id})
MATCH (cp:ColorPalette {name: $palette_name})
MERGE (p)-[:IN_COLOR]->(cp)
"""

REL_PRODUCT_AT_TIER = """
MATCH (p:Product {product_id: $product_id})
MATCH (bt:BudgetTier {label: $tier_label})
MERGE (p)-[:AT_TIER]->(bt)
"""

# ---------------------------------------------------------------------------
# PAIRS_WITH: store ONE directed edge, query undirectionally.
# Use sorted product IDs as canonical direction to avoid duplicates.
# ---------------------------------------------------------------------------

MERGE_PAIRS_WITH = """
MATCH (a:Product {product_id: $product_id_a})
MATCH (b:Product {product_id: $product_id_b})
MERGE (a)-[:PAIRS_WITH]->(b)
"""

# ---------------------------------------------------------------------------
# Brand relationships
# ---------------------------------------------------------------------------

REL_BRAND_KNOWN_FOR_AESTHETIC = """
MATCH (b:Brand {name: $brand_name})
MATCH (a:Aesthetic {name: $aesthetic_name})
MERGE (b)-[:KNOWN_FOR]->(a)
"""

REL_BRAND_AT_TIER = """
MATCH (b:Brand {name: $brand_name})
MATCH (bt:BudgetTier {label: $tier_label})
MERGE (b)-[:AT_TIER]->(bt)
"""

# ---------------------------------------------------------------------------
# Aesthetic OVERLAPS_WITH (bidirectional; caller creates both directions)
# ---------------------------------------------------------------------------

MERGE_OVERLAPS_WITH = """
MATCH (a:Aesthetic {name: $aesthetic_a})
MATCH (b:Aesthetic {name: $aesthetic_b})
MERGE (a)-[:OVERLAPS_WITH]->(b)
"""

# ---------------------------------------------------------------------------
# Vector indexes (384-dim, cosine; embedding column populated by embed.py)
# ---------------------------------------------------------------------------

CREATE_PRODUCT_VECTOR_INDEX = """
CREATE VECTOR INDEX product_embeddings IF NOT EXISTS
FOR (p:Product) ON (p.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
"""

CREATE_AESTHETIC_VECTOR_INDEX = """
CREATE VECTOR INDEX aesthetic_embeddings IF NOT EXISTS
FOR (a:Aesthetic) ON (a.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
"""

# ---------------------------------------------------------------------------
# Count queries (used in tests and health checks)
# ---------------------------------------------------------------------------

COUNT_PRODUCTS = "MATCH (p:Product) RETURN count(p) AS count"
COUNT_BRANDS = "MATCH (b:Brand) RETURN count(b) AS count"
COUNT_AESTHETICS = "MATCH (a:Aesthetic) RETURN count(a) AS count"
COUNT_PAIRS_WITH = "MATCH ()-[:PAIRS_WITH]->() RETURN count(*) AS count"
