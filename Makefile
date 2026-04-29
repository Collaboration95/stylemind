.PHONY: qgate lint format type-check test audit chat seed

qgate:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/
	uv run pyright
	uv run pip-audit

lint:
	uv run ruff check --fix src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

type-check:
	uv run pyright

test:
	uv run pytest -v

audit:
	uv run pip-audit

chat:
	uv run python -m stylemind

seed:
	uv run python scripts/seed.py

embed:
	uv run python scripts/embed.py
