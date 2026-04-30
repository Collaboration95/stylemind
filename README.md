**StyleMind is a RAG-powered fashion chatbot that silently learns your style persona through conversation and recommends outfits via Neo4j graph traversal.**

## Architecture

```mermaid
flowchart TD
    User([User]) --> CLI["CLI\n(Rich + prompt-toolkit)"]
    CLI --> FastAPI["FastAPI\n/chat SSE · /persona · /health"]

    FastAPI --> RAG["RAG Layer\nEmbedder → Retriever → Reranker → Generator"]
    FastAPI --> Persona["Persona Layer\nInferenceEngine · PersonaManager"]
    FastAPI --> Outfit["OutfitBuilder\nPAIRS_WITH traversal"]

    RAG --> Neo4j[("Neo4j\ngraph + vector index")]
    Persona --> Neo4j
    Outfit --> Neo4j

    Generator["Generator"] --> ChatLLM["Chat LLM\nGroq · Llama 3.3 70B"]
    RAG --> Generator

    Persona --> InferenceEngine["InferenceEngine"]
    InferenceEngine --> ExtractionLLM["Extraction LLM\nGroq · Llama 3.3 70B"]

    Persona --> PersonaManager["PersonaManager"]
    PersonaManager --> Langfuse["Langfuse Cloud\nobservability · tracing"]

    FastAPI -- "fire-and-forget\npersona update loop" --> Persona
```

## Quick Start

```bash
cp .env.example .env        # set CHAT_API_KEY (Groq), EXTRACTION_API_KEY (Groq), NEO4J_PASSWORD
docker-compose up --build   # seed + embed run automatically on startup
```

The app is available at `http://localhost:8000`. Neo4j Browser at `http://localhost:7474`.

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.14 | `uv` + hatchling |
| Graph + Vector DB | Neo4j 5 Community | One DB: graph traversal + native vector index |
| Chat LLM | Groq · Llama 3.3 70B | OpenAI-compatible SDK, swap via `CHAT_BASE_URL` |
| Extraction LLM | Groq · Llama 3.3 70B | Structured output (JSON schema), swap via `EXTRACTION_BASE_URL` |
| Embeddings | all-MiniLM-L6-v2 | Local, 384 dims, no API key |
| API | FastAPI + SSE | Streaming tokens, async lifespan |
| CLI | Rich + prompt-toolkit | Embeds FastAPI server in background thread |
| Observability | Langfuse Cloud | `@observe` spans across full pipeline, token usage, persona confidence scores |
| Packaging | Docker (two-stage, non-root) | `docker-compose up --build` starts everything |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Streaming chat via SSE — persona-aware RAG pipeline |
| `GET` | `/persona/{user_id}` | Current inferred persona snapshot |
| `GET` | `/outfit/{product_id}` | Build a coherent outfit around an anchor product |
| `GET` | `/products/names` | Product catalog for autocomplete |
| `GET` | `/health` | Liveness check (Neo4j + embedder) |

## CLI

```bash
uv run python -m stylemind
```

| Command | Action |
|---------|--------|
| `/help` | Show all commands and conversation starters |
| `/persona` | Print inferred style persona |
| `/outfit <name>` | Build outfit around a product (fuzzy match + tab-complete) |
| `/debug-dev` | Per-turn persona signals extracted this session |
| `/clear` | Clear conversation history |
| `/exit` | End session (also: `/quit`, `quit`, `exit`) |
| `1` / `2` / `3` | Use a conversation starter from the welcome screen |

Product names support **tab-completion** anywhere in the input.

## How It Works

### RAG pipeline (per chat turn)

```mermaid
sequenceDiagram
    participant U as User
    participant API as /chat (SSE)
    participant R as ProductRetriever
    participant RR as ProductReranker
    participant G as StyleMindGenerator
    participant PM as PersonaManager

    U->>API: POST /chat {message, history}
    API->>PM: get_persona(user_id)
    PM-->>API: PersonaSnapshot
    API->>R: retrieve(message)
    R-->>API: list[RetrievedProduct]
    API->>RR: rerank(products, persona)
    RR-->>API: list[RerankResult]
    API->>G: stream_response(message, history, products)
    G-->>U: SSE token stream
    API-->>U: __JSON__ sources + signals
    API-->>U: [DONE]
    Note over API,PM: fire-and-forget persona persistence
    API->>PM: update_persona(user_id, signals)
```

### Persona inference & storage

```mermaid
flowchart LR
    Msg["User message"] --> IE["InferenceEngine\nLlama 3.3 70B"]
    IE -->|"PersonaSignals\n(aesthetics, materials,\noccasions, budget)"| PM["PersonaManager"]
    PM -->|"UNWIND batch writes\nin single transaction"| Neo4j[("Neo4j\nStylePersona node\n+ relationship edges")]
    Neo4j -->|"GET_PERSONA_ALL\n(single query)"| Snap["PersonaSnapshot\n(decay-weighted)"]
    Snap --> RAG["RAG reranking\n+ outfit building"]
```

### Outfit builder graph traversal

```mermaid
flowchart TD
    Anchor["Anchor product\n(user expressed interest)"] -->|"PAIRS_WITH"| Candidates["Paired candidates"]
    Candidates --> Filter["Coherence filter\n≥1 season overlap\n≥1 occasion overlap"]
    Filter --> Rank["Persona ranking\n(aesthetic + occasion match)"]
    Rank --> Dedup["Deduplicate by category\n(max 1 item per category)"]
    Dedup --> Outfit["OutfitSuggestion\n(anchor + ≤4 items)"]
```

## Observability

| Service | URL |
|---------|-----|
| Langfuse Cloud | [us.cloud.langfuse.com](https://us.cloud.langfuse.com) |
| Neo4j Browser | [localhost:7474](http://localhost:7474) |

Langfuse captures per-turn spans for retrieval, reranking, persona extraction, generation, and outfit building. Token usage (prompt/completion/total) is logged for both LLMs. `score_persona_confidence` is emitted each turn for drift detection.

The `/debug-dev` CLI command provides a local alternative — all persona signals extracted during the session as a Rich table, no network required.

## Docs

| Document | Description |
|----------|-------------|
| [Design Decisions & Dev Setup](design.md) | Architecture rationale, environment variables, local development, troubleshooting |
| [Gap Analysis](docs/gap_analysis.md) | Requirement compliance matrix |
| [Future Improvements](docs/planning_future_improvements.md) | P2/P3 design docs with effort estimates |
| [Demo Script](docs/demo_script.md) | 5-turn walkthrough for screen recording |
