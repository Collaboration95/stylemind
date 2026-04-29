# StyleMind — Future Improvements Planning

Items organized by priority. Each includes motivation, design, implementation sketch,
and effort estimate. These are improvements beyond the core requirements — they
demonstrate product thinking, developer experience awareness, and performance consciousness.

---

## P2: Medium Priority — Ship Next

### P2-1: /outfit Command (Explicit Outfit Request)

**Problem:** Outfit building currently only triggers when the LLM detects product interest
via keyword matching in `generator.detect_product_interest()`. This is fragile — the user
must mention a product by name AND use an interest phrase ("I like", "show me more", etc.).

**Motivation:** An explicit `/outfit <product name>` command gives users direct control and
demonstrates the PAIRS_WITH graph traversal capability on demand — critical for the demo.

**Design:**
```
You (turn 3): /outfit Silk Slip Cami

StyleMind: Here's a complete outfit built around the Silk Slip Cami by COS...
┌──────────────────────────────────────────────────────┐
│ Outfit Suggestion — Date Night / SS                  │
├────────────────┬────────┬─────┬────────┬─────────────┤
│ Item           │ Cat.   │ Brand│ Price │ Justification│
├────────────────┼────────┼─────┼────────┼─────────────┤
│ Silk Slip Cami │ Top    │ COS │ ₹3,800│ anchor       │
│ Tailored Short │ Bottom │ COS │ ₹3,200│ PAIRS_WITH   │
│ ...            │ ...    │ ... │ ...   │ ...          │
└────────────────┴────────┴─────┴────────┴─────────────┘
```

**Implementation:**
1. In `cli/chat.py`, add `/outfit` command handler:
   - Parse product name from input: `/outfit <name>`
   - Fuzzy-match against a local cache of product names (fetched once at startup via a new `/products` API endpoint, or from the first SSE sources payload)
   - Send a crafted message like "I love the {name}, show me a complete outfit" to `/chat`
   - OR: add a dedicated `/outfit/{product_id}` API endpoint that runs only the outfit builder
2. Option B (simpler): new `GET /outfit/{product_id}?user_id=X` endpoint in `api/outfit.py`
   - Calls `outfit_builder.build_outfit(product_id, user_id, persona)`
   - Returns `OutfitSuggestion` JSON directly
   - CLI renders it via existing `_render_outfit()`

**Effort:** 30 min (Option B), 45 min (Option A with fuzzy match)

---

### P2-2: Conversation Starters on Empty Input

**Problem:** When a user opens the CLI and doesn't know what to type, there's no guidance
beyond the welcome message. The `/help` text now includes starters, but they aren't
surfaced proactively.

**Motivation:** Reduces time-to-first-interaction and guides users toward queries that
showcase the system's strengths (occasion-based, aesthetic-based, budget-aware).

**Design:** After the welcome message, show 3 randomly selected conversation starters:
```
Try one of these to get started:
  1. "I need an outfit for a wedding next month"
  2. "Show me streetwear under 5k"
  3. "What goes well with linen pants?"
```

**Implementation:**
1. Define a list of 8-10 curated starters in `cli/chat.py`
2. `random.sample(starters, 3)` in the welcome flow
3. Optionally: if user types just a number (1/2/3), send the corresponding starter

**Effort:** 10 min

---

### P2-3: Product Name Autocomplete in CLI

**Problem:** Users need to type exact product names for outfit matching or interest
detection. Typos or partial names fail silently.

**Motivation:** Shows polish and attention to UX. prompt-toolkit already supports
autocompletion.

**Design:** On CLI startup, fetch product names via a lightweight API call. Feed them
into a `WordCompleter` for prompt-toolkit. Tab-completion triggers on product names.

**Implementation:**
1. Add `GET /products/names` endpoint (returns `list[str]` of product names)
2. In CLI init, fetch the list and create `WordCompleter(names, ignore_case=True)`
3. Pass completer to `prompt()` call

**Effort:** 20 min

---

## P3: Low Priority — Nice to Have

### P3-1: Parallel Persona Extraction

**Problem:** In the current flow (after PR 1 changes), persona signal extraction happens
inline before the SSE `[DONE]` event so the CLI can capture the signals payload. The DB
persistence is fire-and-forget. But the extraction LLM call (~200-400ms) adds to the
total SSE response time.

**Motivation:** For production, every 100ms matters. Users perceive the response as "done"
once text streaming stops, but the SSE connection stays open for the signals payload.

**Design Options:**
- **Option A (keep inline):** Accept the latency. The signals payload arrives right after
  text and before `[DONE]`, so the user sees products → text → done in quick succession.
  The extraction delay is hidden by the streaming text duration. Keep this as-is.
- **Option B (parallel extraction):** Start extraction concurrently with the LLM streaming.
  Send signals via a separate SSE event after `[DONE]`. CLI handles post-DONE events.
  More complex but shaves ~200ms off perceived response time.

**Recommendation:** Option A is fine for a take-home. The extraction happens in ~200ms
which is invisible while text is still streaming. Optimize only if measured latency is an issue.

**Effort:** 30 min (Option B)

---

### P3-2: Response Latency Tracking in Langfuse

**Problem:** We log response time in the CLI (`{elapsed:.1f}s`) but don't send it to Langfuse.
Production monitoring needs this as a custom metric for alerting and SLA tracking.

**Motivation:** Shows observability maturity — not just tracing spans but tracking business
metrics (time-to-first-token, total response time).

**Design:**
1. In `api/chat.py`, wrap `_sse_stream` with timing:
   ```python
   start = time.monotonic()
   # ... yield all events ...
   elapsed = time.monotonic() - start
   langfuse_context.score_current_observation(name="response_latency_ms", value=elapsed * 1000)
   ```
2. Create a Langfuse dashboard widget for p50/p95 response latency

**Effort:** 15 min

---

### P3-3: Retrieval Quality Scoring

**Problem:** We don't know how well the retrieved products match the user's actual intent.
The similarity score is a vector distance, not a relevance judgment.

**Motivation:** Systematic RAG quality tracking enables data-driven improvements (e.g.,
"retrieval quality dropped after we changed embeddings").

**Design:**
1. After reranking, compute `mean_similarity` and `max_similarity` across top-k products
2. Log both as Langfuse scores on the retrieval span
3. Over time, correlate with user satisfaction signals (positive sentiment on shown products)

**Effort:** 15 min

---

### P3-4: Aesthetic-Based Vector Search Fallback

**Problem:** When the user's query doesn't match any product embeddings well (all scores
below threshold), we return empty results. The LLM then says "no products match."

**Motivation:** Aesthetic embeddings exist but are unused in retrieval. A fallback that
searches aesthetic embeddings could find the right vibe even when product descriptions
don't directly match.

**Design:**
1. If `ProductRetriever.retrieve()` returns fewer than 3 results, trigger aesthetic fallback
2. Embed the query, search the aesthetic vector index
3. Find top-matching aesthetic → retrieve products with that aesthetic via graph traversal
4. Merge with original results, deduplicate

**Effort:** 45 min

---

### P3-5: Session Continuity Across CLI Restarts

**Problem:** Each CLI session generates a new `user_id` (UUID). The user's persona is saved
in Neo4j but lost when they restart the CLI because they get a new ID.

**Motivation:** For demo purposes this is fine (fresh persona = cleaner demo). But for a
real product, session continuity matters.

**Design:**
1. On first run, save `user_id` to `~/.stylemind/session.json`
2. On subsequent runs, load and reuse the ID
3. Add `--new-session` flag to force a fresh ID
4. Add `--user-id <id>` flag for explicit control (useful for demos)

**Effort:** 15 min

---

## Summary Priority Matrix

| Item | Effort | Demo Impact | Product Thinking | Recommendation |
|------|--------|-------------|------------------|----------------|
| P2-1 /outfit command | 30min | High | Medium | Do it |
| P2-2 Conversation starters | 10min | Medium | High | Do it |
| P2-3 Autocomplete | 20min | Medium | High | Do it if time permits |
| P3-1 Parallel extraction | 30min | None | Medium | Skip for now |
| P3-2 Latency tracking | 15min | None | High | Do it (quick win) |
| P3-3 Retrieval scoring | 15min | None | High | Do it (quick win) |
| P3-4 Aesthetic fallback | 45min | Medium | High | Do if time permits |
| P3-5 Session continuity | 15min | Low | Medium | Skip for take-home |
