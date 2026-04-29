# StyleMind Demo Script — 5-Turn Persona Evolution Walkthrough

Loom recording target: under 5 minutes. Each turn approximately 30 seconds of narration.

---

## Prerequisites

### 1. Environment Variables

```bash
export GROQ_API_KEY="<your-groq-key>"
export OPENAI_API_KEY="<your-openai-key>"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<your-neo4j-password>"
# Verify keys are set without exposing values:
echo "GROQ_API_KEY is set: $([ -n "$GROQ_API_KEY" ] && echo yes || echo NO)"
echo "OPENAI_API_KEY is set: $([ -n "$OPENAI_API_KEY" ] && echo yes || echo NO)"
```

### 2. Start Infrastructure

```bash
docker-compose up --build -d
# Wait ~15 seconds for Neo4j to be healthy
docker-compose ps
```

### 3. Seed the Graph and Embed Products

```bash
uv run python scripts/seed.py    # idempotent — safe to re-run
uv run python scripts/embed.py   # generates 384-dim MiniLM vectors
```

These commands are idempotent; re-running them will not create duplicate nodes.

### 4. Verify Seed

```bash
# Optional quick sanity check via cypher-shell or Neo4j Browser
# MATCH (p:Product) RETURN count(p)  -- expect 51
# MATCH ()-[r:PAIRS_WITH]->() RETURN count(r)  -- expect 100+
```

---

## How to Run the Demo

### Terminal Layout (2-panel split recommended)

Open two terminal panes side-by-side. A tool like tmux or iTerm2 split pane works well.

**Left pane — main chat:**
```bash
uv run python -m stylemind
```

The CLI starts the FastAPI server in a background thread, then opens the Rich chat interface. You type messages at the `> ` prompt and see streaming LLM responses token-by-token.

**Right pane — persona inspection:**
After each chat turn, type `/persona` in the left pane (or run `curl http://localhost:8000/persona/<session-id>` in the right pane) to see the current `PersonaSnapshot` as pretty-printed JSON.

### Font Size

Increase terminal font to at least 16pt so text is readable in a Loom recording at 1080p.

### Session ID

The CLI prints a session ID on startup. Keep it visible or note it for curl commands in the right pane.

---

## Turn 1 — Broad Office Ask

**User types:**
> I need some office outfits for my new corporate job

**What happens internally:**

1. **Embed** — The query is encoded by `sentence-transformers/all-MiniLM-L6-v2` into a 384-dim vector.
2. **Retrieve** — Vector similarity search returns the top-10 most semantically relevant products across all aesthetics. No persona signal exists yet, so the retriever applies no filter.
3. **Rerank** — `PersonaAwareReranker` runs with an empty `PersonaSnapshot` (confidence_score=0.0). Persona boost factor is `confidence * weight = 0.0 * weight = 0`, so all items retain their raw similarity score.
4. **Generate** — Groq Llama 3.3-70B streams a response citing 5-8 broad products.
5. **Persona update** — `PersonaInferenceEngine` calls gpt-4.1-nano with the turn. It extracts `PersonaSignals`: `signal_strength` is low (no strong preferences stated), `mentioned_occasions=["Office"]`, no aesthetics, no budget. These signals are written as weighted edges to Neo4j.

**Expected response preview:**

- LLM surfaces a broad mix: Linen Wide-Leg Trouser (COS), Oversized Blazer (Arket), Silk Blouse (Massimo Dutti), Leather Loafer (COS), Structured Leather Tote (Massimo Dutti)
- Response tone is broad and educational: "For a corporate environment, you might consider..."
- Products span Corporate Minimalism, Old Money, and Quiet Luxury — all office-appropriate aesthetics

**Run `/persona` after this turn:**

```json
{
  "preferred_aesthetics": [],
  "disliked_materials": [],
  "budget_tier": null,
  "top_occasions": ["Office"],
  "confidence_score": 0.05
}
```

**Loom narrator notes:**

- "Notice that on Turn 1, StyleMind returns a broad cross-section. There is no persona yet — the system knows nothing about this user's taste."
- Point to the `confidence_score: 0.05` in the right pane. "Barely any signal — one mention of 'office' as an occasion."
- Highlight that results include 5+ products across multiple aesthetics — the system is casting a wide net.

---

## Turn 2 — Style Signal (Minimal / Understated)

**User types:**
> I prefer minimal and understated styles, nothing too flashy

**What happens internally:**

1. **Embed** — Query vector computed.
2. **Retrieve** — Vector search returns top-10; Quiet Luxury and Corporate Minimalism products naturally rank high due to semantic overlap with "minimal understated."
3. **Rerank** — Persona boost is still low but non-zero. The `liked_aesthetics=["Quiet Luxury", "Corporate Minimalism"]` edges written from this turn will increase boost weight on the next turn. On this turn the reranker already nudges Quiet Luxury products upward.
4. **Generate** — LLM stream highlights minimal, logo-free pieces.
5. **Persona update** — gpt-4.1-nano extracts `liked_aesthetics=["Quiet Luxury", "Corporate Minimalism"]`, `signal_strength=0.7`. Weighted PREFERS edges created in Neo4j.

**Expected response preview:**

- LLM narrows to: Crew Neck Cashmere Knit (Arket), Pleated Midi Skirt (COS), Tailored Wool Trouser (Massimo Dutti), Silk Blouse (Massimo Dutti)
- Response tone: "Since you prefer understated, these pieces rely on fabric quality rather than statement design..."
- Flashy or logo-heavy items (Streetwear, Y2K) are absent from results

**Run `/persona` after this turn:**

```json
{
  "preferred_aesthetics": ["Quiet Luxury"],
  "disliked_materials": [],
  "budget_tier": null,
  "top_occasions": ["Office"],
  "confidence_score": 0.22
}
```

**Loom narrator notes:**

- "Notice `preferred_aesthetics` now contains 'Quiet Luxury'. The persona picked that up purely from what the user said — no direct question was ever asked."
- "Confidence jumped from 0.05 to 0.22. The system is starting to model this person."
- Point to the absence of Streetwear/Y2K products: "Already the results are narrowing."

---

## Turn 3 — Budget + Material Dislike

**User types:**
> I'm on a mid-range budget, and I really hate synthetic fabrics like polyester

**What happens internally:**

1. **Embed** — Query vector computed.
2. **Retrieve** — Vector search; natural language around "mid-range budget" and "synthetic fabrics" pulls in products with relevant descriptions.
3. **Rerank** — `PersonaAwareReranker` now has two active signals: PREFERS Quiet Luxury/Corporate Minimalism (boost), DISLIKES Polyester (penalty). Products containing Polyester or Recycled Polyester in their material field are penalized. Budget tier filter de-weights Premium/Luxury items outside the 2000-8000 INR range.
4. **Generate** — LLM stream avoids recommending athletic/performance wear (Recycled Polyester) and very expensive pieces.
5. **Persona update** — gpt-4.1-nano extracts `budget_signal="Mid"`, `disliked_materials=["Polyester"]`, `signal_strength=0.8`. DISLIKES Material edge created in Neo4j.

**Expected response preview:**

- LLM surfaces: Linen Wide-Leg Trouser (COS, ₹4,200), Silk Slip Cami (COS, ₹3,800), Pleated Midi Skirt (COS, ₹4,800), Leather Loafer (COS, ₹6,200)
- Response tone: "Sticking to natural fabrics and mid-range pricing, here are pieces that fit your criteria..."
- New Balance athleisure (Recycled Polyester) and premium Toteme (₹18,500+) do not appear

**Run `/persona` after this turn:**

```json
{
  "preferred_aesthetics": ["Quiet Luxury", "Corporate Minimalism"],
  "disliked_materials": ["Polyester"],
  "budget_tier": "Mid",
  "top_occasions": ["Office"],
  "confidence_score": 0.38
}
```

**Loom narrator notes:**

- "Budget tier is now locked in: Mid (₹2,000–₹8,000). And 'Polyester' has been added to `disliked_materials`."
- "The reranker is actively penalizing any product with synthetic materials — you can see athletic wear has dropped out of results entirely."
- "Confidence is now 0.38 — we're getting a real picture of this user."

---

## Turn 4 — Product Interest Triggers Outfit Builder

**User types:**
> I love that silk slip cami, can you show me an outfit with it?

**What happens internally:**

1. **Keyword detection** — The API layer scans the message for outfit-trigger keywords: "love that", "outfit with", "show me". Matched. Product name "silk slip cami" is matched against known product names → resolves to P012 (Silk Slip Cami, COS, ₹3,800).
2. **Outfit builder** — `OutfitBuilder.build(anchor=P012)` runs a PAIRS_WITH graph traversal from P012. It traverses the undirected PAIRS_WITH edges to collect candidates: P001 (Linen Wide-Leg Trouser), P002 (Oversized Blazer), P015 (Slip Dress Midi), P019 (Tailored Blazer), P027 (Pleated Midi Skirt).
3. **Coherence filter** — Filters candidates to those sharing at least 1 season overlap (P012 is SS|AW) AND at least 1 occasion overlap (P012 is Date Night|Casual). Enforces max 1 item per category.
4. **Persona filter** — Further filters against active persona: prefers Quiet Luxury/Corporate Minimalism, dislikes Polyester, Mid budget.
5. **Outfit result** — Final outfit (3–4 coherent items): Silk Slip Cami (Tops, anchor) + Pleated Midi Skirt (Bottoms, P027) + Leather Loafer (Footwear, P010) + Structured Leather Tote (Bags, P021, shown as optional accessory layer).
6. **Generate** — LLM streams description of the complete outfit, explaining season/occasion coherence and why each piece works together.
7. **Persona update** — gpt-4.1-nano extracts `sentiment_on_shown={"P012": "positive"}`, `signal_strength=0.9`. Positive sentiment edge recorded.

**Expected response preview:**

- "Here's a complete outfit built around the Silk Slip Cami..."
  - Tops: Silk Slip Cami (COS, ₹3,800) — your anchor piece
  - Bottoms: Pleated Midi Skirt (COS, ₹4,800) — fluid silk blend, same Quiet Luxury aesthetic
  - Footwear: Leather Loafer (COS, ₹6,200) — polished and season-spanning
  - Bag: Structured Leather Tote (Massimo Dutti, ₹14,500) — optional upgrade layer
- "This outfit works for Date Night (silk + minimal) and the pieces share season overlap (SS/AW)"
- All items: Mid budget range, natural materials only (Silk, Leather), Quiet Luxury aesthetic

**Run `/persona` after this turn:**

```json
{
  "preferred_aesthetics": ["Quiet Luxury", "Corporate Minimalism"],
  "disliked_materials": ["Polyester"],
  "budget_tier": "Mid",
  "top_occasions": ["Office", "Date Night"],
  "confidence_score": 0.55
}
```

**Loom narrator notes:**

- "The system detected 'I love that' and triggered the outfit builder automatically — no special command needed."
- "Point to the outfit items: each one is a different category — Tops, Bottoms, Footwear, Bags. That's the max-1-per-category rule in action."
- "The outfit builder used the PAIRS_WITH graph edges — these are hand-curated relationships, not LLM guesses. Season and occasion were validated before any item was included."
- "Notice Date Night appeared in `top_occasions` — inferred from the Silk Slip Cami context, not from the user stating it."
- "Confidence is now 0.55 — we're past the halfway mark."

---

## Turn 5 — Tight Personalized Results

**User types:**
> What about something for a date night, still keeping it minimal?

**What happens internally:**

1. **Embed** — Query vector computed; "date night minimal" naturally clusters near Quiet Luxury, Old Money, Date Night products.
2. **Retrieve** — Vector search returns top-10. Because the query mentions "date night", `mentioned_occasions=["Date Night"]` products surface.
3. **Rerank** — `PersonaAwareReranker` runs with a fully populated snapshot (confidence_score=0.55+). Persona boost is now strong. The reranker applies:
   - +boost: Quiet Luxury, Corporate Minimalism, Old Money aesthetics
   - -penalty: Polyester materials
   - +boost: Date Night occasion match
   - Budget filter: Mid tier preferred (₹2,000–₹8,000)
4. **Generate** — LLM streams a tight, highly personalized set. Only 3 products rather than 8 — the filter funnel is tight.
5. **Persona update** — `mentioned_occasions` adds Date Night with high weight. `confidence_score` grows further.

**Expected response preview:**

- LLM surfaces only: Slip Dress Midi (Toteme, shown as aspirational), Pleated Midi Skirt + Silk Slip Cami combination, Pearl Drop Earring (Arket, ₹2,200) as accessory layer
- Response tone: "For date night while staying minimal, these three pieces fit your profile exactly..."
- Zero athletic wear. Zero synthetic fabrics. Zero Streetwear. Zero Budget-tier fast fashion. Tight curation.

**Run `/persona` after this turn:**

```json
{
  "preferred_aesthetics": ["Quiet Luxury", "Corporate Minimalism", "Old Money"],
  "disliked_materials": ["Polyester"],
  "budget_tier": "Mid",
  "top_occasions": ["Office", "Date Night"],
  "confidence_score": 0.73
}
```

**Loom narrator notes:**

- "Compare this to Turn 1. We went from 8 broad products spanning 5 aesthetics to 3 tightly curated items, all Quiet Luxury or Old Money, all natural materials, all Mid budget."
- "Confidence_score is now 0.73 — the persona is solid. Any further turns will produce even tighter results."
- "The system learned all of this from 5 natural conversation turns — no profile form, no preferences page, no direct questions. Pure inference."
- "This is the core value proposition: the persona evolves silently and the results get better without the user doing any extra work."

---

## Key Moments to Emphasize in the Recording

### Streaming Speed
- Tokens appear in the terminal in real-time via SSE. On Groq + Llama 3.3-70B, the first token typically arrives in under 500ms and full responses complete in 3–6 seconds.
- During the recording, do not cut away while the response is streaming — the live token-by-token appearance is a key demo point.

### Confidence Score Growing
- Show the `/persona` output after each turn. Call out the number explicitly in narration:
  - Turn 1: 0.05
  - Turn 2: 0.22
  - Turn 3: 0.38
  - Turn 4: 0.55
  - Turn 5: 0.73
- This arc — 0.0 → 0.73 — is the central technical story.

### Outfit Suggestion Coherence
- On Turn 4, zoom in on (or highlight with cursor) the outfit items. Call out:
  - One item per category (Tops, Bottoms, Footwear, Bags)
  - Season overlap validated (all items share SS and/or AW)
  - Occasion overlap validated (all items include Date Night or Casual)
  - All items match the active persona: Quiet Luxury, Mid budget, no Polyester

### Breadth Narrowing
- A useful visual at the end: scroll up in the terminal to show Turn 1 results (broad, many aesthetics) then scroll to Turn 5 results (tight, 3 items).
- Narrate: "Same system, same vector index, same graph — but five turns of implicit signal completely changed the output."

---

## Loom Recording Tips

- **Duration:** Target 4–5 minutes. Aim for ~30 seconds per turn plus ~30 seconds for intro and ~30 seconds for closing summary.
- **Font size:** Set terminal font to at least 16pt before recording. In iTerm2: Preferences > Profiles > Text > Font size.
- **No secrets visible:** Do not show API key values. Use `echo "GROQ_API_KEY is set: yes"` style checks rather than `echo $GROQ_API_KEY`.
- **Layout:** Use a 50/50 horizontal split. Left pane: chat CLI. Right pane: `/persona` output (re-run after each turn, clear between turns with `clear`).
- **Cursor guidance:** Move the mouse cursor to highlight key values in the persona JSON after each turn — viewers need to know where to look.
- **Do not rush Turn 4:** The outfit builder output takes a moment to appear (graph traversal + coherence filter + LLM stream). Let it render fully before narrating.
- **Suggested breakpoints (cumulative):**
  - 0:00–0:30 — Introduction: what StyleMind is, what the demo will show
  - 0:30–1:00 — Turn 1 (broad results, persona = empty)
  - 1:00–1:30 — Turn 2 (style signal, aesthetic appears)
  - 1:30–2:00 — Turn 3 (budget + material dislike, budget_tier + disliked_materials appear)
  - 2:00–3:00 — Turn 4 (outfit trigger — give this extra time, it is the most visual moment)
  - 3:00–3:30 — Turn 5 (tight personalized results, confidence_score 0.73)
  - 3:30–4:30 — Closing: scroll back to compare Turn 1 vs Turn 5, summarize the persona arc
  - 4:30–5:00 — Buffer / bloopers / outro
