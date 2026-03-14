# statsworth

[![CI](https://github.com/ReneeErnst/statsworth/actions/workflows/ci.yml/badge.svg)](https://github.com/ReneeErnst/statsworth/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ReneeErnst/statsworth/branch/main/graph/badge.svg)](https://codecov.io/gh/ReneeErnst/statsworth)

A reusable Python library for statistical analysis, built to support research workflows.

## Installation

```bash
uv add "statsworth @ git+https://github.com/ReneeErnst/statsworth.git"
# or
pip install "statsworth @ git+https://github.com/ReneeErnst/statsworth.git"
```

Pin to a specific release:
```bash
uv add "statsworth @ git+https://github.com/ReneeErnst/statsworth.git@v0.1.0"
```

## Modules

### Preprocessing (`statsworth.preprocessing`)
- `clean_columns(df)` — normalize column names, drop all-NaN rows, strip survey prefixes
- `corrected_item_total_correlations(df)` — item-rest correlations for scale reliability
- `vif(df)` — Variance Inflation Factors for multicollinearity detection
- `scale_totals(df, subscales)` — compute subscale totals from a `dict[str, list[str]]` mapping

### Factor Analysis (`statsworth.factor_analysis`)
- `efa(df, n_factors, ...)` — EFA via `factor-analyzer`
- `parallel_analysis(df, ...)` — determine optimal factor count
- `factor_loadings_table(loadings, item_names, factor_names)` — format loadings as a DataFrame
- `get_items_with_low_loadings(loadings, item_names)` — items below `LOW_LOADING_THRESHOLD` (0.4)
- `no_low_loadings_solution(df, low_loadings, n_factors)` — iterative item removal and re-EFA
- `strongest_loadings(loadings, item_names)` — map each item to its primary factor
- `cronbach_alpha(df)` — Cronbach's alpha for scale reliability

### ANOVA (`statsworth.anova`)
- `one_way_anova(df, group_col, dv_col)` — one-way ANOVA with Tukey HSD post-hoc
- `welch_anova_and_games_howell(df, group_col, dv_col)` — Welch ANOVA + Games-Howell post-hoc
- `games_howell(df, group_col, dv_col)` — standalone Games-Howell post-hoc
- `one_way_manova(df, group_col, dv_cols)` — MANOVA with Tukey follow-up
- `one_way_manova_games_howell(df, group_col, dv_cols)` — MANOVA with Games-Howell follow-up

### SEM (`statsworth.sem`)
- `rmsea_95ci(model)` — RMSEA point estimate + 95% CI from a `semopy` model

### Visualization (`statsworth.visualization`)
- `efa_item_corr_matrix(df)` — item correlation matrix for EFA
- `scree_plot(eigenvalues)` — scree plot from eigenvalues
- `scree_parallel_analysis(...)` — scree plot overlaid with parallel analysis results
- `plot_loadings_heatmap(loadings, ...)` — factor loadings heatmap
- `check_normality(df, ...)` — multi-panel normality diagnostics (returns dict, plots as side effect)
- `corr_heatmap(df)` — masked lower-triangle correlation heatmap
- `highlight_corr(val)` — CSS styling helper for correlation tables in notebooks

## Development

```bash
uv sync          # install dependencies
make lint        # ruff check + format check
make typecheck   # mypy
make test        # pytest
make coverage    # pytest with coverage report
make pre-commit  # run all pre-commit hooks
```
