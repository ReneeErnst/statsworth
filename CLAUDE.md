# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install    # uv sync
make lint       # ruff check + format check
make typecheck  # mypy stats_core/
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

**`stats_core/preprocessing.py`** — Data cleaning utilities:
- `clean_columns` — normalize column names, drop all-NaN rows, strip survey prefixes
- `corrected_item_total_correlations` — item-rest correlations for scale reliability
- `vif` — Variance Inflation Factors for multicollinearity detection
- `scale_totals(df, subscales: dict[str, list[str]])` — subscale totals; subscales are parameterized by dict (not hardcoded)

**`stats_core/factor_analysis/efa.py`** — EFA via `factor-analyzer`:
- `efa`, `parallel_analysis`, `factor_loadings_table`, `get_items_with_low_loadings`, `no_low_loadings_solution`, `strongest_loadings`, `cronbach_alpha`
- `LOW_LOADING_THRESHOLD = 0.4`

**`stats_core/anova/`** — ANOVA/MANOVA:
- `one_way.py`: `one_way_anova` (+ Tukey HSD), `welch_anova_and_games_howell`, `games_howell`
- `manova.py`: `one_way_manova` (+ Tukey), `one_way_manova_games_howell`
- Games-Howell post-hoc is conditional on significance; `DEFAULT_ALPHA = 0.05`

**`stats_core/sem.py`** — `rmsea_95ci(model)`: RMSEA + 95% CI from a semopy model using noncentral chi-square

**`stats_core/visualization.py`** — Plotting functions; most generate matplotlib figures as side effects while returning data:
- `efa_item_corr_matrix`, `scree_plot`, `scree_parallel_analysis`, `plot_loadings_heatmap`, `check_normality`, `corr_heatmap`, `highlight_corr`

### Testing conventions

- `tests/conftest.py` sets matplotlib backend to `"Agg"` (non-interactive)
- Fixtures use seeded RNG for reproducibility
- Visualization tests are smoke tests (verify no exception, check return type/shape)
- Prefer `@pytest.mark.parametrize` over multiple separate test functions when testing the same behavior across different inputs
