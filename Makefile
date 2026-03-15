.PHONY: install lint typecheck test coverage pre-commit check

install:
	uv sync --all-extras

lint:
	uv run ruff check statsworth/ tests/
	uv run ruff format --check statsworth/ tests/

typecheck:
	uv run mypy statsworth/

test:
	uv run pytest tests/ -v --tb=short

coverage:
	uv run pytest tests/ --cov=statsworth --cov-report=term-missing

pre-commit:
	uv run pre-commit run --all-files

check: lint typecheck test
