# Migration Research: scaledev → statsworth

## 1. Core Logic — Migration Candidates

### `scaledev/preprocessor.py`
| Function | Purpose | Migration Priority |
|---|---|---|
| `clean_columns(df)` | Normalizes column names, drops all-NaN rows | High |
| `corrected_item_total_correlations(df)` | Item-rest correlations (scale reliability) | High |
| `vif(df)` | Variance Inflation Factors for multicollinearity | High |
| `scale_totals(df, scale_items)` | Subscale totals — **hardcoded column names** | Medium (needs refactor first) |

### `scaledev/modeler.py`
| Function | Purpose | Migration Priority |
|---|---|---|
| `efa(df, n_factors, method, rotation)` | EFA wrapper via factor-analyzer | High |
| `parallel_analysis(df, K, ...)` | Determines optimal factor count | High |
| `factor_loadings_table(loadings, item_names, factor_names)` | Formats loadings array as DataFrame | High |
| `get_items_with_low_loadings(loadings, item_names, threshold)` | Identifies weak items | High |
| `no_low_loadings_solution(df, low_loadings, n_factors)` | Iterative item removal + re-EFA | High |
| `strongest_loadings(loadings, item_names)` | Maps items to their primary factor | High |
| `one_way_anova(df, group_col, dv_col)` | ANOVA + Tukey HSD | High |
| `welch_anova_and_games_howell(df, group_col, dv_col)` | Welch ANOVA + Games-Howell | High |
| `games_howell(df, group_col, dv_col)` | Standalone Games-Howell post-hoc | High |
| `one_way_manova(df, group_col, dv_cols)` | MANOVA + Tukey follow-up | High |
| `one_way_manova_games_howell(df, group_col, dv_cols, alpha)` | MANOVA + Games-Howell follow-up | High |
| `rmsea_95ci(model)` | RMSEA point estimate + 95% CI from semopy model | High |

### `scaledev/vizer.py`
| Function | Purpose | Migration Priority |
|---|---|---|
| `highlight_corr(val)` | CSS styling helper for correlation tables | Medium |
| `corr_matrix(df, cols)` | Styled pandas correlation matrix | Medium |
| `scree_plot(common_factors_ev, max_viz)` | Eigenvalue scree plot | High |
| `scree_parallel_analysis(max_scree_factors, avg_factor_eigens, data_ev)` | Parallel analysis scree plot | High |
| `plot_loadings_heatmap(loadings, item_names, factor_names)` | Factor loadings heatmap | High |
| `check_normality(df, dist_plot, qq_plot)` | Multi-panel normality diagnostics | High |
| `corr_matrix_v2(df, title)` | Masked lower-triangle correlation heatmap | Medium |

### `scaledev/__init__.py`
| Function | Purpose | Migration Priority |
|---|---|---|
| `get_data_dir()` | Returns path to data directory | Low (repo-specific) |
| `set_pd_display()` | Configures pandas display options | Low (utility/convenience) |

### Notebooks — Extractable Logic
The following patterns used in notebooks should inform the library's API design:
- Reverse scoring items before analysis (currently done inline in notebooks, not in library)
- Cronbach's alpha calculation (not yet in the library, done ad-hoc with `pingouin`)
- Data loading from Excel with column cleanup pipeline (`openpyxl` + `clean_columns`)
- Iterative EFA refinement workflow (good candidate for a higher-level workflow function)

---

## 2. Structural Gaps — Directory Reorganization

### Current statsworth structure
```
statsworth/
├── .git/
├── .gitignore
├── LICENSE
├── README.md
└── refactor_plan.md
```

### Proposed statsworth structure (after migration)
```
statsworth/
├── statsworth/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── columns.py          # clean_columns, corrected_item_total_correlations, vif
│   ├── factor_analysis/
│   │   ├── __init__.py
│   │   └── efa.py              # efa, parallel_analysis, factor_loadings_table, etc.
│   ├── anova/
│   │   ├── __init__.py
│   │   └── tests.py            # one_way_anova, welch_anova_and_games_howell, games_howell
│   ├── manova/
│   │   ├── __init__.py
│   │   └── tests.py            # one_way_manova, one_way_manova_games_howell
│   ├── sem/
│   │   ├── __init__.py
│   │   └── fit_indices.py      # rmsea_95ci
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # All vizer.py functions
├── tests/
│   ├── test_preprocessing.py
│   ├── test_efa.py
│   ├── test_anova.py
│   ├── test_manova.py
│   ├── test_sem.py
│   └── test_visualization.py
├── pyproject.toml
├── Makefile
├── LICENSE
├── README.md
└── refactor_plan.md
```

**Key reorganization decisions:**
- Split `modeler.py` (483 lines, 4 domains) into `factor_analysis/`, `anova/`, `manova/`, `sem/`
- Rename `vizer.py` → `visualization/plots.py` for clarity
- Use `preprocessing/` instead of `preprocessor` for consistency with sklearn conventions
- Do NOT migrate `get_data_dir()` or `set_pd_display()` — these are scaledev-specific utilities

---

## 3. Dependencies

The following dependencies from scaledev must be added to statsworth's `pyproject.toml`:

| Package | Used In | Notes |
|---|---|---|
| `pandas>=2.2.3` | All modules | Core data structure |
| `numpy>=2.2.1` | modeler, vizer | Numerical operations |
| `scipy>=1.14.1` | modeler (rmsea_95ci, games_howell) | Statistical distributions |
| `statsmodels>=0.14.4` | preprocessor (vif), modeler (MANOVA) | Statistical models |
| `factor-analyzer>=0.5.1` | modeler (efa, parallel_analysis) | EFA implementation |
| `pingouin>=0.5.5` | modeler (Welch ANOVA) | Statistical tests |
| `semopy>=2.3.11` | modeler (rmsea_95ci) | Structural equation modeling |
| `matplotlib>=3.10.0` | vizer | Plotting |
| `seaborn` | vizer (plot_loadings_heatmap, corr_matrix_v2) | Statistical plots |
| `openpyxl>=3.1.5` | notebooks (Excel I/O) | Excel file support |

**Dev/CI dependencies:**
| Package | Purpose |
|---|---|
| `pytest>=9.0.2` | Test runner |
| `mypy>=1.14.1` | Type checking |
| `ruff>=0.9.3` | Linting + formatting |

**Build system:** Use `hatchling` + `uv` (same as scaledev — no reason to change).

**Note:** `graphviz` and `jupyter` from scaledev are NOT needed in statsworth library code (they are runtime/notebook tools, not library dependencies).

---

## 4. Test Coverage

### Existing tests (from scaledev)
- `tests/test_modeler.py` — covers only `rmsea_95ci()` (9 test methods, well-written)
  - These tests can be migrated directly to `statsworth/tests/test_sem.py`
  - `_fit_model()` helper and `_expected_95ci()` reference implementation should be preserved

### Tests that need to be written for statsworth

**`test_preprocessing.py`**
- `clean_columns`: column name normalization, all-NaN row removal
- `corrected_item_total_correlations`: known-answer test with synthetic data
- `vif`: known multicollinearity case (highly correlated features → high VIF)
- `scale_totals`: once refactored to accept parameterized subscale definitions

**`test_efa.py`**
- `efa`: smoke test with synthetic factor structure data
- `parallel_analysis`: returns integer, correct range
- `factor_loadings_table`: shape and column name validation
- `get_items_with_low_loadings`: correct items returned below threshold
- `no_low_loadings_solution`: convergence with synthetic data
- `strongest_loadings`: maps each item to highest absolute loading factor

**`test_anova.py`**
- `one_way_anova`: known-answer test (groups with different means)
- `welch_anova_and_games_howell`: significant and non-significant cases
- `games_howell`: pairwise comparison structure validation

**`test_manova.py`**
- `one_way_manova`: known-answer multivariate test
- `one_way_manova_games_howell`: structural validation

**`test_visualization.py`**
- Smoke tests only (functions execute without error)
- Use `matplotlib.testing` or `pytest-mpl` for output validation if desired

**Recommended test data strategy:**
- Generate synthetic DataFrames with known statistical properties using `numpy.random` with fixed seeds
- Avoid loading from Excel files in tests (no `data/` directory in statsworth)

---

## 5. Refactor Opportunities

### Critical (fix before migration)
1. **Syntax error in `preprocessor.py` line 22**
   ```python
   # BUG: double assignment
   df.columns = df.columns.str.lower().columns = df.columns.str.lower()
   # FIX:
   df.columns = df.columns.str.lower()
   ```

2. **Hardcoded column names in `scale_totals()`**
   - Currently references `inclusion1`–`inclusion5`, `presence1`–`presence6`, etc. by string
   - Refactor to accept a `dict[str, list[str]]` parameter mapping subscale names to item lists
   - Example: `scale_totals(df, subscales={"inclusion": ["inc1", "inc2", ...]})`

### High Priority
3. **Extract shared MANOVA logic**
   - `one_way_manova()` and `one_way_manova_games_howell()` share ~90% of code
   - Extract common body into `_run_manova(df, group_col, dv_cols, post_hoc_fn)` internal helper

4. **Add type hints throughout**
   - `preprocessor.py`: all functions lack type annotations
   - `vizer.py`: all functions lack type annotations
   - `modeler.py`: `efa()`, `parallel_analysis()`, `factor_loadings_table()`, `games_howell()` are missing annotations
   - Use `pd.DataFrame`, `np.ndarray`, `list[str]`, `dict[str, list[str]]` etc.

5. **Replace print statements with structured returns or logging**
   - `check_normality()` prints directly to stdout — inappropriate for a library
   - Consider returning a dict of results and letting callers decide display

### Medium Priority
6. **Add missing `plt.show()` in `corr_matrix_v2()`**

7. **Vectorize `corrected_item_total_correlations()`**
   - Replace column loop with pandas vectorized operations

8. **Extract magic numbers as module-level constants**
   - `LOW_LOADING_THRESHOLD = 0.4` in `factor_analysis/efa.py`
   - `DEFAULT_ALPHA = 0.05` in `anova/tests.py`

### Low Priority
9. **Add module-level docstrings** to all modules

10. **Add `__all__` exports** to each `__init__.py`
