# StyleMind — RAG-based Styling Chatbot

## Candidate Task Brief

---

## 1. Overview & Context

You are building the backend intelligence layer for StyleMind, a personal styling assistant.
The system must maintain a knowledge graph of fashion items, brands, occasions, and
aesthetics, then use Retrieval-Augmented Generation (RAG) to power a conversational agent.

As the user chats, the agent silently infers a style persona — preferences, dislikes, body
notes, budget range — never by asking directly, always through natural conversation signals.

---

## 2. Graph Schema — Node Types

Candidates must model the following node types in a graph database of their choice (Neo4j,
ArangoDB, or a vector + graph hybrid):

| Node type     | Description & key properties                                                              |
|---------------|-------------------------------------------------------------------------------------------|
| Product       | Core item in the catalogue. Properties: name, description, price_inr, color_palette, material, season. |
| Brand         | Fashion brand. Properties: name, tier (Budget/Mid/Premium/Luxury), country_of_origin.     |
| Aesthetic     | Style aesthetic (e.g. Quiet Luxury, Streetwear). Properties: name, description, keywords. |
| Occasion      | Wear context (e.g. Office, Date Night). Properties: name, formality_score.                |
| BodyType      | Body silhouette the item flatters. Properties: name, description.                         |
| ColorPalette  | Named colour grouping (e.g. Earthy Neutrals). Properties: name, hex_codes.                |
| Material      | Fabric or finish type. Properties: name, sustainability_score, feel_tag.                  |
| Season        | Seasonal wear window. Properties: name (SS/AW/Year-round).                                |
| StylePersona  | Inferred user profile. Properties: user_id, confidence_score, last_updated.               |
| BudgetTier    | Price segment. Properties: label, min_inr, max_inr.                                       |

---

## 3. Graph Schema — Required Relationships

All 15 relationships below must be implemented. Relationships marked * are bidirectional.

| From node     | Relationship       | To node            |
|---------------|--------------------|--------------------|
| Product       | BELONGS_TO         | Brand              |
| Product       | FITS_OCCASION      | Occasion           |
| Product       | EMBODIES           | Aesthetic          |
| Product       | SUITS_BODY         | BodyType           |
| Product       | MADE_FROM          | Material           |
| Product       | BEST_IN_SEASON     | Season             |
| Product       | IN_COLOR           | ColorPalette       |
| Product       | AT_TIER            | BudgetTier         |
| Product       | PAIRS_WITH *       | Product            |
| Aesthetic     | OVERLAPS_WITH *    | Aesthetic          |
| Brand         | KNOWN_FOR          | Aesthetic          |
| Brand         | AT_TIER            | BudgetTier         |
| StylePersona  | PREFERS            | Aesthetic          |
| StylePersona  | DISLIKES           | Material / Product |
| StylePersona  | SHOPS_AT           | Brand              |

---

## 4. Mock Data Requirements

A seed CSV file (products_seed.csv) is provided separately. Candidates must load this data
into their graph database using a reproducible seed script. Additional data can be added but
the provided seed must be present in full.

| Entity       | Min count | Notes                                                          |
|--------------|-----------|----------------------------------------------------------------|
| Products     | 40+       | Minimum — use the provided CSV as the base; extend freely      |
| Brands       | 12        | Across 4 budget tiers: Budget / Mid / Premium / Luxury         |
| Aesthetics   | 8         | e.g. Quiet Luxury, Streetwear, Cottagecore, Y2K, Old Money...  |
| Occasions    | 6         | Casual, Office, Date Night, Weekend Brunch, Formal, Active     |
| PAIRS_WITH   | 30+       | Edges between products — critical for the outfit builder       |

> The seed script must be idempotent — running it twice must not create duplicate nodes.
> Use MERGE (Neo4j) or equivalent upsert semantics.

---

## 5. Functional Requirements

### R1 — RAG Retrieval
- Embed product and aesthetic descriptions into a vector store.
- On each user turn, retrieve top-k relevant nodes using semantic similarity + graph traversal.
- Synthesise a grounded, citation-aware response — no hallucinated product names.

### R2 — Implicit Persona Inference
- After each turn, update a hidden StylePersona node for the session user.
- Extract signals: sentiment toward items shown, occasion mentions, budget hints, colour references, brand name-drops.
- The persona must NEVER be updated via a direct question to the user — inference only.

### R3 — Persona-aware Ranking
- All product recommendations must be re-ranked based on the current persona state.
- Early conversation → broad results. Later conversation → tight, personalised shortlist.
- Demonstrate measurable shift in recommendations between turn 1 and turn 5.

### R4 — Outfit Builder
- When the user shows interest in a product, traverse PAIRS_WITH edges to suggest a complete outfit.
- Each item in the outfit must include a brief justification for its inclusion.
- Outfits must be coherent — no season clashes, no occasion mismatches.

### R5 — Persona Snapshot Endpoint
- Expose a /persona API endpoint (or CLI command) returning the inferred persona as structured JSON.
- The JSON must include: preferred_aesthetics, disliked_materials, budget_tier, top_occasions, confidence_score.
- This endpoint is for the evaluator only — it must not be surfaced in the chat UI.

---

## 6. Deliverables

- Graph DB seed script that loads products_seed.csv and all relationships
- Embedding pipeline — product + aesthetic nodes → vector store
- RAG retrieval layer with graph traversal augmentation
- Conversational agent with persona inference loop
- Chat interface — CLI is acceptable; web UI is a bonus
- /persona endpoint returning structured JSON
- README with setup instructions (≤ 5 commands to get running)
- Short screen recording (Loom or similar) of a 5-turn demo conversation

---

## 7. Technical Constraints & Stack Notes

- **Language:** Python 3.10+
- **Graph DB:** Neo4j (preferred), ArangoDB, or equivalent — must support property graphs
- **Vector store:** any (Qdrant, Weaviate, FAISS, Chroma, Pinecone)
- **LLM:** OpenAI GPT-4o, Anthropic Claude, or any open-source model — document your choice
- **No hardcoded API keys** — use .env or environment variables
- All dependencies must be listed in requirements.txt
- The seed script must be idempotent (safe to run multiple times)

> **Bonus points:** streaming chat responses, temporal persona decay (older signals weighted less),
> multi-user session isolation, or an explain-my-recommendation mode showing the graph path used.
