# AGENTS.md — StyleMind

## What
RAG fashion chatbot. Neo4j graph (products/brands/aesthetics) + vector index. Persona inferred silently from chat. Outfit builder via PAIRS_WITH traversal.

## Stack
- Python 3.14, uv, hatchling
- Neo4j 5 Community (graph + native vector index, ONE DB for both)
- Chat LLM: Groq + Llama 3.3 70B (OpenAI-compatible SDK, swap via CHAT_BASE_URL env)
- Extraction LLM: OpenAI gpt-4.1-nano (structured output, swap via EXTRACTION_BASE_URL env)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, 384 dims, no API key)
- FastAPI + SSE streaming. Rich CLI embeds server in bg thread.
- Langfuse self-hosted for observability (localhost:3000)
- NO LangChain. NO LlamaIndex. Framework-free.

## Conventions
- `from __future__ import annotations` every file
- Pydantic BaseModel = API contracts. @dataclass(frozen=True) = internal state. StrEnum = enums.
- Config: frozen dataclasses with `@classmethod from_env()`, thread-safe `get_config()` singleton, `_reset_config()` for tests
- Ruff: line-length=120, select E/W/F/I/UP/B/SIM, ignore E501/B008/SIM108/UP006/UP035/E402
- Pyright: basic mode, exclude tests/scripts
- All commands via `uv run`. Makefile targets: qgate, lint, format, type-check, test, audit
- Logging: `logger = logging.getLogger(__name__)`, structured key=value
- Tests: pytest, asyncio_mode=auto, markers unit/integration/e2e/performance. DI via fixtures, mock Neo4j/LLM.
- Docker: two-stage build, non-root appuser, healthcheck
- Git: conventional commits (feat/fix/test/docs/chore)

## Project layout
```
src/stylemind/
  config.py          # AppConfig singleton
  main.py            # FastAPI app + lifespan
  __main__.py        # CLI entry (starts server in bg thread)
  models/            # enums.py, schemas.py (Pydantic), domain.py (dataclasses)
  graph/             # client.py, queries.py, repository.py
  rag/               # embedder.py, retriever.py, reranker.py, generator.py
  persona/           # inference.py, manager.py
  outfit/            # builder.py
  cli/               # chat.py (Rich + prompt-toolkit)
  api/               # chat.py, persona.py, health.py
  observability.py   # Langfuse wrappers
scripts/             # seed.py, embed.py
data/                # products_seed.csv, enrichment.py
tests/               # conftest.py, test_*.py
```

## CSV gotcha
4 rows have unquoted commas in description. csv.reader and pandas BOTH FAIL. Use RTL parser:
```python
tokens = line.split(",")
first_12 = tokens[:12]; pairs_with = tokens[-1]; desc = ",".join(tokens[12:-1])
```
Remap aesthetic "Casual" -> "Casual Minimalism" for P037, P038.

## Graph schema
10 node types: Product, Brand, Aesthetic, Occasion, BodyType, ColorPalette, Material, Season, BudgetTier, StylePersona
15 relationships. PAIRS_WITH: store ONE directed edge, query undirectionally. OVERLAPS_WITH: derive from co-occurrence.
51 products (45 CSV + 6 synthetic for FabIndia + Auralee). 12 brands.

## Key rules
- Persona NEVER updated by asking user directly. Inference only.
- Seed script MUST be idempotent (MERGE semantics).
- LLM provider-agnostic: both clients use `OpenAI(base_url=..., api_key=...)`. Swap = env var change.
- get_persona() returns empty default on first turn, NEVER None.
- Outfit coherence: at least 1 season overlap AND 1 occasion overlap. Max 1 item per category.
- Planning docs in `../planning/` (01-05) have full details.
