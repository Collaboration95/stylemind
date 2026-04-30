# Design Decisions

## Neo4j as unified graph + vector store
Products, relationships, and 384-dim embeddings live in one database. Vector search and graph traversal run in a single Cypher pipeline, eliminating sync complexity and the fan-out latency of a separate vector DB.


## Split-model architecture

Groq + Llama 3.3 70B handles both streaming chat (sub-200 ms TTFT) and structured persona extraction (JSON-schema conformance). A single provider simplifies setup while remaining swappable via environment variables.

## Provider-agnostic LLM clients
clients are `OpenAI(base_url=..., api_key=...)`. Swapping providers is two environment variable changes.

## Silent persona inference
The system never asks users what they like. Preferences are extracted from conversational signals (liked aesthetics, disliked materials, budget cues, sentiment on shown products) after every turn. 

## Outfit coherence via graph traversal
Outfit candidates are validated by requiring ≥1 season overlap AND ≥1 occasion overlap using `PAIRS_WITH` edges in Neo4j. This is deterministic and explainable. Letting the LLM guess outfit coherence would produce plausible-sounding but fashion-incoherent combinations.

---

# Environment Variables

All configuration is via `.env`. See `.env.example` for the full list with defaults.

**Required:**

| Variable | Description |
|----------|-------------|
| `CHAT_API_KEY` | API key for the chat LLM provider (Groq by default) |
| `EXTRACTION_API_KEY` | API key for the extraction LLM (Groq by default) |
| `NEO4J_PASSWORD` | Neo4j password (must match `NEO4J_AUTH` in docker-compose) |

All other variables have sensible defaults. The chat and extraction LLMs are swappable via `CHAT_BASE_URL` / `EXTRACTION_BASE_URL` — any OpenAI-compatible endpoint works.

---


---

# Troubleshooting

**`NEO4J_PASSWORD` not set** — Copy `.env.example` to `.env` and fill in the required values.

**Neo4j not ready** — The CLI polls `/health` for up to 30s. Check `docker-compose logs neo4j` and ensure the password matches `NEO4J_AUTH` in `docker-compose.yml`.

**Empty responses** — Run `uv run python scripts/embed.py`. Products need embeddings before vector search works.

**Persona not updating** — Persona updates are fire-and-forget. Check logs for `chat persona update failed`. Common causes: Neo4j blip or missing `EXTRACTION_API_KEY`.

**Langfuse traces not appearing** — Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. The app degrades gracefully if Langfuse is unavailable.
