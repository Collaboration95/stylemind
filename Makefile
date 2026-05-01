.PHONY: qgate lint format type-check test test-unit test-integration test-e2e \
       audit chat web-chat seed embed seed-and-embed setup dev health \
       db-up db-down up down clean pre-commit-install

# ── Quality Gate ────────────────────────────────────────────────────────────────

qgate: lint-check format-check type-check audit test-unit

lint-check:
	uv run ruff check src/ tests/ scripts/

lint:
	uv run ruff check --fix src/ tests/ scripts/

format-check:
	uv run ruff format --check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

type-check:
	uv run pyright

audit:
	uv run pip-audit

# ── Tests ───────────────────────────────────────────────────────────────────────

test:
	uv run pytest -v

test-unit:
	uv run pytest -m unit -v

test-integration:
	uv run pytest -m integration -v

test-e2e:
	uv run pytest -m e2e -v

test-perf:
	uv run pytest -m performance -v

# ── Data Pipeline ───────────────────────────────────────────────────────────────

seed:
	uv run python scripts/seed.py

embed:
	uv run python scripts/embed.py

seed-and-embed: seed embed

# ── Run ─────────────────────────────────────────────────────────────────────────

chat:
	uv run python -m stylemind

web-chat:
	uv run streamlit run src/stylemind/ui/app.py --server.port 8000 --server.headless true --server.fileWatcherType none

dev:
	uv run uvicorn stylemind.main:app --reload --host 127.0.0.1 --port 8001

health:
	@curl -sf http://localhost:8001/health && echo " OK" || echo " FAIL (is the server running?)"

# ── Docker ──────────────────────────────────────────────────────────────────────

db-up:
	docker compose up -d neo4j
	@echo "Waiting for Neo4j healthcheck..."
	@while ! docker compose ps neo4j 2>/dev/null | grep -q "healthy"; do sleep 2; done
	@echo "Neo4j ready at bolt://localhost:7687 — browser at http://localhost:7474"

db-down:
	docker compose down neo4j

up:
	BUILDX_BUILDER=desktop-linux docker compose up --build -d
	@echo "API: http://localhost:8001  Neo4j: http://localhost:7474  Web UI: make web-chat (localhost:8000)"

down:
	docker compose down

# ── Setup ───────────────────────────────────────────────────────────────────────

setup: pre-commit-install
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example — edit API keys before running")
	uv sync
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. Edit .env with your API keys (CHAT_API_KEY, EXTRACTION_API_KEY)"
	@echo "  2. make db-up              # start Neo4j"
	@echo "  3. make seed-and-embed     # load data + embeddings"
	@echo "  4. make web-chat           # launch web UI at http://localhost:8000"
	@echo "     make chat               # or: terminal CLI alternative"

pre-commit-install:
	uv run pre-commit install

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned caches and Docker volumes"
