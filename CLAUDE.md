# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install    # uv sync --all-extras
make lint       # ruff check + format check
make typecheck  # mypy statsworth/
make test       # pytest tests/ -v --tb=short
make coverage    # pytest with coverage report
make pre-commit  # run all pre-commit hooks against all files
```

Run a single test file:
```bash
uv run pytest tests/test_efa.py -v --tb=short
```

Run a specific test:
```bash
uv run pytest tests/test_efa.py::test_function_name -v
```

## Architecture

A personal reusable Python library for statistical analysis. Uses `uv` for package management and `hatchling` as build backend.

### Modules

Read the source before making changes. Module-level orientation:

- **`statsworth/preprocessing.py`** — Data cleaning and scale utilities
- **`statsworth/factor_analysis/efa.py`** — EFA via `factor-analyzer`
- **`statsworth/anova/one_way.py`** — One-way ANOVA and Welch/Games-Howell variants
- **`statsworth/anova/manova.py`** — One-way MANOVA with post-hoc follow-ups
- **`statsworth/sem.py`** — SEM fit statistics (RMSEA) via semopy
- **`statsworth/visualization.py`** — Plotting functions for EFA, correlations, and normality checks

### Testing conventions

- `tests/conftest.py` sets matplotlib backend to `"Agg"` (non-interactive)
- Fixtures use seeded RNG for reproducibility
- Visualization tests are smoke tests (verify no exception, check return type/shape)
- Prefer `@pytest.mark.parametrize` over multiple separate test functions when testing the same behavior across different inputs
