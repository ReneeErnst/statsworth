.PHONY: install lint typecheck test

install:
	uv sync

lint:
	uv run ruff check stats_core/ tests/
	uv run ruff format --check stats_core/ tests/

typecheck:
	uv run mypy stats_core/

test:
	uv run pytest tests/ -v --tb=short
