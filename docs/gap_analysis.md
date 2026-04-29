# StyleMind — Gap Analysis & Improvement Plan

Requirement doc: `docs/requirements.md` (converted from StyleMind_Candidate_Task.pdf)
Audit date: 2026-04-30

---

## 1. Core Requirement Compliance Matrix

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| **2** | 10 node types with correct properties | DONE | All 10 present, all properties verified |
| **3** | 15 relationships | DONE | All 15 + bonus INTERESTED_IN |
| **4.1** | 40+ products | DONE | 51 products (45 CSV + 6 synthetic) |
| **4.2** | 12 brands across 4 tiers | DONE | 12 brands, 4 budget tiers |
| **4.3** | 8+ aesthetics | DONE | 9 defined + 1 remapped |
| **4.4** | 6+ occasions | DONE | 6 occasions |
| **4.5** | 30+ PAIRS_WITH edges | DONE | 127 edges |
| **4.6** | Idempotent seed script | DONE | MERGE semantics throughout |
| **R1** | RAG retrieval (vector + graph) | DONE | Single Cypher: vector search + 8 OPTIONAL MATCHes |
| **R1** | Citation-aware, no hallucination | DONE | System prompt + product context grounding |
| **R2** | Implicit persona inference | DONE | PersonaInferenceEngine extracts signals silently |
| **R2** | Never ask user directly | DONE | System prompt enforces this |
| **R3** | Persona-aware reranking | DONE | ProductReranker with aesthetic boost/penalty/budget |
| **R3** | Measurable shift turn 1→5 | DONE | Demo script shows confidence 0.05→0.73 |
| **R4** | Outfit builder via PAIRS_WITH | DONE | OutfitBuilder with coherence validation |
| **R4** | Item justification | DONE | Rule-based justification per item |
| **R4** | No season/occasion clashes | DONE | ≥1 season AND ≥1 occasion overlap required |
| **R5** | /persona endpoint | DONE | GET /persona/{user_id} returns PersonaSnapshot JSON |
| **R5** | Required fields in JSON | DONE | preferred_aesthetics, disliked_materials, budget_tier, top_occasions, confidence_score |
| **6.1** | Seed script | DONE | scripts/seed.py |
| **6.2** | Embedding pipeline | DONE | scripts/embed.py (products + aesthetics) |
| **6.3** | RAG retrieval layer | DONE | rag/retriever.py + reranker.py |
| **6.4** | Conversational agent + persona loop | DONE | api/chat.py SSE + persona fire-and-forget |
| **6.5** | CLI chat interface | DONE | cli/chat.py with Rich + prompt-toolkit |
| **6.6** | /persona endpoint | DONE | api/persona.py |
| **6.7** | README ≤5 commands | DONE | 2 commands (cp + docker-compose) |
| **6.8** | Screen recording | GAP | Demo script exists, recording not yet made |
| **7** | Python 3.10+ | DONE | Python 3.14 |
| **7** | Neo4j | DONE | Neo4j 5 Community |
| **7** | No hardcoded API keys | DONE | All via .env |
| **7** | requirements.txt | DONE | 854 lines |
| **7** | Idempotent seed | DONE | MERGE semantics |

### Bonus Features

| Bonus Feature | Status | Implementation |
|---------------|--------|----------------|
| Streaming chat responses | DONE | SSE via FastAPI StreamingResponse |
| Temporal persona decay | DONE | Exponential decay in PersonaManager._apply_decay() |
| Multi-user session isolation | DONE | user_id-scoped StylePersona nodes + CLI generates unique UUID |
| Explain-my-recommendation mode | DONE | `explain=True` flag, ScoreBreakdown per product |

---

## 2. Gaps & Staleness Issues

### GAP-1: README references local Langfuse (STALE)
- **What:** README line 37 says "Langfuse at http://localhost:3000" and the architecture diagram references localhost:3000
- **Impact:** Confusing for evaluator — Langfuse is now cloud-only
- **Fix:** Update README to reference Langfuse Cloud, update diagram

### GAP-2: README CLI section missing new commands
- **What:** README only documents `/persona` and `quit/exit`. Missing: `/help`, `/debug-dev`, `/clear`, `/exit`
- **Impact:** Evaluator won't discover the developer tooling we built
- **Fix:** Update CLI Usage table in README

### GAP-3: Demo script references wrong env vars
- **What:** `docs/demo_script.md` references `GROQ_API_KEY` and `OPENAI_API_KEY` instead of `CHAT_API_KEY` and `EXTRACTION_API_KEY`
- **Impact:** Following the demo script will fail
- **Fix:** Update demo script env var names

### GAP-4: No screen recording
- **What:** Deliverable requires "Short screen recording (Loom or similar) of a 5-turn demo conversation"
- **Impact:** Missing deliverable
- **Fix:** User needs to record this manually using docs/demo_script.md

### GAP-5: CLI welcome experience is generic
- **What:** Welcome message is functional but impersonal. No personality, no warmth.
- **Impact:** First impression matters — evaluator's first touch is the CLI
- **Fix:** Warm intro with personality, capability summary, /help nudge

### GAP-6: No session summary on exit
- **What:** Typing `quit` just says "Goodbye!" with no recap of what was learned
- **Impact:** Missed opportunity to demonstrate the persona inference value
- **Fix:** Show final persona snapshot on exit if signals were extracted

### GAP-7: No turn indicator in prompt
- **What:** Every prompt says "You: " — no sense of conversation progression
- **Impact:** Hard to correlate with /debug-dev turn numbers; no visual sense of persona building
- **Fix:** Show turn number in prompt: "You (turn 3): "

### GAP-8: Persona confidence not surfaced in responses
- **What:** The LLM doesn't know its own confidence level. Low confidence = broad results, but the user can't tell
- **Impact:** No transparency about personalization state
- **Fix:** Add a subtle confidence indicator to the CLI after responses

---

## 3. Nice-to-Have Improvements (Product Thinking)

### UX-1: Onboarding flow
Show a warm welcome that sets expectations:
```
Hi! I'm StyleMind, your personal fashion stylist.
I'll learn your style as we chat — no questionnaires, I promise.
Just tell me what you're looking for, and I'll do the rest.
```
This signals intelligence and builds trust.

### UX-2: Conversation starters
When the user types nothing or seems stuck, suggest conversation starters:
- "Looking for a date night outfit?"
- "What's your vibe for this summer?"
- "Need something for the office?"

### UX-3: /outfit command
Currently outfit building only triggers when the LLM detects product interest via keyword matching. An explicit `/outfit <product name>` command would let users request outfits directly.

### UX-4: Price-range aware responses
When persona has budget_tier, the response text should naturally reference budget awareness:
"Since you're looking for budget-friendly options..." — this is now partially done via persona injection but could be more explicit in the system prompt.

### UX-5: Session persistence warning
Currently each CLI session gets a new user_id. Users might expect continuity. Adding a note in the welcome message ("New session — your style starts fresh!") sets expectations.

### PERF-1: Parallel extraction + persistence
Currently persona extraction happens inline (blocking the SSE close). The DB write is fire-and-forget. Both extraction AND persistence could be fully fire-and-forget to reduce response latency.

### PERF-2: Embedding model pre-warm
The sentence-transformers model loads on first embed call. Pre-loading at import time (during startup) would eliminate cold-start latency on the first user query.

### OBS-1: Response latency tracking
Log and trace end-to-end response time (user message → last SSE chunk). Surface in Langfuse as a custom metric. Important for production monitoring.

### OBS-2: Retrieval quality scoring
Score each retrieved product set against the final user message. Log "retrieval relevance" in Langfuse. Enables systematic RAG quality tracking.

---

## 4. Priority Matrix

| Priority | Item | Effort | Impact | Category |
|----------|------|--------|--------|----------|
| P0 | GAP-1: Fix stale README | 10 min | High | Correctness |
| P0 | GAP-2: Document new CLI commands | 10 min | High | Correctness |
| P0 | GAP-3: Fix demo script env vars | 5 min | High | Correctness |
| P1 | GAP-5: Warm CLI welcome | 15 min | High | UX |
| P1 | GAP-6: Session summary on exit | 20 min | Medium | UX |
| P1 | GAP-7: Turn indicator in prompt | 5 min | Medium | UX |
| P1 | GAP-8: Confidence indicator | 15 min | Medium | UX |
| P2 | UX-1: Onboarding flow | 15 min | Medium | UX |
| P2 | UX-3: /outfit command | 30 min | Medium | Feature |
| P3 | PERF-1: Parallel extraction | 20 min | Low | Performance |
| P3 | OBS-1: Latency tracking | 15 min | Low | Observability |
