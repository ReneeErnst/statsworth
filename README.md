# statsworth

[![CI](https://github.com/ReneeErnst/statsworth/actions/workflows/ci.yml/badge.svg)](https://github.com/ReneeErnst/statsworth/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)

A reusable Python library for statistical analysis, built to support research workflows.

## Installation

The base package includes preprocessing and ANOVA modules:

```bash
uv add "statsworth @ git+https://github.com/ReneeErnst/statsworth.git"
# or
pip install "statsworth @ git+https://github.com/ReneeErnst/statsworth.git"
```

Install optional extras as needed:

```bash
# EFA (factor-analyzer, scikit-learn)
uv add "statsworth[efa] @ git+https://github.com/ReneeErnst/statsworth.git"

# Visualization (matplotlib)
uv add "statsworth[viz] @ git+https://github.com/ReneeErnst/statsworth.git"

# SEM (semopy)
uv add "statsworth[sem] @ git+https://github.com/ReneeErnst/statsworth.git"

# Everything
uv add "statsworth[all] @ git+https://github.com/ReneeErnst/statsworth.git"
```

Pin to a specific release:
```bash
uv add "statsworth @ git+https://github.com/ReneeErnst/statsworth.git@v0.1.0"
```

## Modules

| Module | Description | Extra |
|---|---|---|
| `statsworth.preprocessing` | Column normalization, item-total correlations, VIF, subscale totals | — |
| `statsworth.anova` | One-way ANOVA, Welch ANOVA, MANOVA, Games-Howell post-hoc | — |
| `statsworth.factor_analysis` | EFA, parallel analysis, loadings utilities, Cronbach's alpha | `efa` |
| `statsworth.visualization` | Scree plots, loadings heatmaps, normality diagnostics, correlation heatmaps | `viz` |
| `statsworth.sem` | SEM/CFA fit indices and RMSEA 95% CI | `sem` |

## Development

```bash
make install     # uv sync --all-extras
make lint        # ruff check + format check
make typecheck   # mypy
make test        # pytest
make coverage    # pytest with coverage report
make pre-commit  # run all pre-commit hooks
```
