# statsworth

[![CI](https://github.com/ReneeErnst/statsworth/actions/workflows/ci.yml/badge.svg)](https://github.com/ReneeErnst/statsworth/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ReneeErnst/statsworth/branch/main/graph/badge.svg)](https://codecov.io/gh/ReneeErnst/statsworth)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)

A reusable Python library for statistical analysis, built to support research workflows.

## Installation

```bash
uv add "statsworth @ git+https://github.com/ReneeErnst/statsworth.git"
# or
pip install "statsworth @ git+https://github.com/ReneeErnst/statsworth.git"
```

To use `statsworth.sem`, include the `sem` extra (installs `semopy`):
```bash
uv add "statsworth[sem] @ git+https://github.com/ReneeErnst/statsworth.git"
# or
pip install "statsworth[sem] @ git+https://github.com/ReneeErnst/statsworth.git"
```

Pin to a specific release:
```bash
uv add "statsworth @ git+https://github.com/ReneeErnst/statsworth.git@v0.1.0"
```

## Modules

| Module | Description | Extra required |
|---|---|---|
| `statsworth.preprocessing` | Column normalization, item-total correlations, VIF, subscale totals | — |
| `statsworth.factor_analysis` | EFA, parallel analysis, loadings utilities, Cronbach's alpha | — |
| `statsworth.anova` | One-way ANOVA, Welch ANOVA, MANOVA, Games-Howell post-hoc | — |
| `statsworth.sem` | SEM/CFA fit indices and RMSEA 95% CI | `sem` |
| `statsworth.visualization` | Scree plots, loadings heatmaps, normality diagnostics, correlation heatmaps | — |

## Development

```bash
uv sync          # install dependencies
make lint        # ruff check + format check
make typecheck   # mypy
make test        # pytest
make coverage    # pytest with coverage report
make pre-commit  # run all pre-commit hooks
```
