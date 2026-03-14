# Migration Plan: scaledev → stats-core

> **Status:** Phase 1 complete. Awaiting signal to begin Phase 2.

---

## Phase 1: Environment Setup ✓

### 1.1 Directory structure
```
stats_core/
├── __init__.py
├── preprocessing.py
├── visualization.py
├── sem.py
├── anova/
│   ├── __init__.py
│   ├── one_way.py
│   └── manova.py
└── factor_analysis/
    ├── __init__.py
    └── efa.py
tests/
└── __init__.py
```

### 1.2 `pyproject.toml`
`hatchling` build backend, `uv` for dependency management. All runtime and dev dependencies installed.

### 1.3 `Makefile`
Targets: `install`, `lint`, `typecheck`, `test`.

### 1.4 Gate — PASSED
```bash
uv run python -c "import stats_core"  # exits 0
```

---

## Phase 2: Foundation Migration

### 2.1 Migrate `clean_columns`
Copy from `scaledev/preprocessor.py` into `stats_core/preprocessing.py`.
Apply the **critical syntax fix** (line 22 double assignment):
```python
# Before (bug):
df.columns = df.columns.str.lower().columns = df.columns.str.lower()
# After (fix):
df.columns = df.columns.str.lower()
```
Add type hints: `def clean_columns(df: pd.DataFrame) -> pd.DataFrame`.

### 2.2 Migrate `corrected_item_total_correlations` and `vif`
Copy into `stats_core/preprocessing.py`.
Add full type hints to both functions.
Vectorize `corrected_item_total_correlations` (replace column loop with `.corrwith()` or equivalent pandas vectorized operation).

### 2.3 Refactor and migrate `scale_totals`
Replace hardcoded column names with a `subscales: dict[str, list[str]]` parameter:
```python
def scale_totals(df: pd.DataFrame, subscales: dict[str, list[str]]) -> pd.DataFrame:
    ...
```
Do **not** migrate the old signature.

### 2.4 Write `tests/test_preprocessing.py`
- `test_clean_columns_normalizes_names`: uppercase + spaces → lowercase + underscores
- `test_clean_columns_drops_all_nan_rows`: row where every value is NaN is removed
- `test_corrected_item_total_correlations_known_answer`: synthetic 5-item scale, verify correlations are in [-1, 1] and expected high/low items match
- `test_vif_high_multicollinearity`: two near-identical columns → VIF > 10
- `test_scale_totals_parameterized`: dict-based subscale definition produces correct sums

**Verification:**
```bash
uv run pytest tests/test_preprocessing.py -v
```
All tests must pass before Phase 3.

---

## Phase 3: Domain Migration

### 3.1 Migrate `factor_analysis` module
Copy from `scaledev/modeler.py` into `stats_core/factor_analysis/efa.py`:
- `efa`, `parallel_analysis`, `factor_loadings_table`, `get_items_with_low_loadings`, `no_low_loadings_solution`, `strongest_loadings`

Add missing type hints to `efa()`, `parallel_analysis()`, `factor_loadings_table()`.
Extract magic number: `LOW_LOADING_THRESHOLD = 0.4` as module-level constant.

### 3.2 Write `tests/test_efa.py`
- `test_efa_smoke`: synthetic 3-factor data (fixed seed), verify loadings shape = `(n_items, n_factors)`
- `test_parallel_analysis_returns_int`: result is a positive integer ≤ n_items
- `test_factor_loadings_table_shape`: correct shape and column names
- `test_get_items_with_low_loadings`: items below threshold are returned, items above are not
- `test_strongest_loadings_maps_correctly`: each item maps to its highest absolute loading factor

**Verification:**
```bash
uv run pytest tests/test_efa.py -v
```

### 3.3 Migrate `anova` module
Copy from `scaledev/modeler.py` into `stats_core/anova/one_way.py`:
- `one_way_anova`, `welch_anova_and_games_howell`, `games_howell`

Extract magic number: `DEFAULT_ALPHA = 0.05` as module-level constant.

### 3.4 Write `tests/test_anova.py`
- `test_one_way_anova_significant`: three groups with large mean difference → p < 0.05
- `test_one_way_anova_nonsignificant`: three groups with identical means → p > 0.05
- `test_welch_anova_significant`: unequal-variance groups with large difference → significant
- `test_games_howell_pairwise_structure`: result contains all pairwise group combinations

**Verification:**
```bash
uv run pytest tests/test_anova.py -v
```

### 3.5 Migrate `manova` module with shared-logic extraction
Copy from `scaledev/modeler.py` into `stats_core/anova/manova.py`:
- `one_way_manova`, `one_way_manova_games_howell`

**Extract shared logic** into internal helper:
```python
def _run_manova(
    df: pd.DataFrame,
    group_col: str,
    dv_cols: list[str],
    post_hoc_fn: Callable,
) -> ...:
    ...
```
Both public functions delegate to `_run_manova`.

### 3.6 Write `tests/test_manova.py`
- `test_one_way_manova_significant`: multivariate groups with clear separation → significant
- `test_one_way_manova_games_howell_structure`: result contains expected pairwise comparison keys

**Verification:**
```bash
uv run pytest tests/test_manova.py -v
```

**Phase 3 full gate:**
```bash
uv run pytest tests/test_efa.py tests/test_anova.py tests/test_manova.py -v
```
All must pass before Phase 4.

---

## Phase 4: Visualization & SEM

### 4.1 Migrate `visualization` module
Copy from `scaledev/vizer.py` into `stats_core/visualization.py`:
- `highlight_corr`, `corr_matrix`, `scree_plot`, `scree_parallel_analysis`, `plot_loadings_heatmap`, `check_normality`, `corr_matrix_v2`

Apply fixes:
- **`check_normality`**: replace `print()` calls with a returned `dict` of results; keep plot generation as a side effect
- **`corr_matrix_v2`**: add missing `plt.show()` call

Add type hints to all functions.
Do **not** migrate `get_data_dir()` or `set_pd_display()`.

### 4.2 Migrate `sem` module
Copy `rmsea_95ci` from `scaledev/modeler.py` into `stats_core/sem.py`.
Confirm type hints are present (already annotated in source).

### 4.3 Migrate existing `rmsea_95ci` tests
Copy `scaledev/tests/test_modeler.py` → `stats-core/tests/test_sem.py`.
Preserve `_fit_model()` helper and `_expected_95ci()` reference implementation unchanged.
Update import: `from scaledev.modeler import rmsea_95ci` → `from stats_core.sem import rmsea_95ci`.

### 4.4 Write `tests/test_visualization.py`
Smoke tests only — verify functions execute without raising:
- `test_scree_plot_runs`: passes synthetic eigenvalue list
- `test_scree_parallel_analysis_runs`: passes synthetic parallel analysis data
- `test_plot_loadings_heatmap_runs`: passes synthetic loadings array
- `test_check_normality_runs`: passes synthetic DataFrame, verify return type is `dict`
- `test_corr_matrix_v2_runs`: passes synthetic DataFrame

**Verification:**
```bash
uv run pytest tests/test_sem.py tests/test_visualization.py -v
```
All must pass before Phase 5.

---

## Phase 5: Cleanup & Final Verification

### 5.1 Type hints pass
```bash
uv run mypy stats_core/
```
Resolve all errors. Target: zero mypy errors with `strict = false` (gradual typing).

### 5.2 Lint and format pass
```bash
uv run ruff check stats_core/ tests/
uv run ruff format stats_core/ tests/
```

### 5.3 Add `__all__` to all `__init__.py` files
`anova/__init__.py` and `factor_analysis/__init__.py` re-export their public APIs.
Top-level `stats_core/__init__.py` re-exports everything.

### 5.4 Add module-level docstrings
One-line docstrings to all modules:
`preprocessing.py`, `visualization.py`, `sem.py`, `anova/one_way.py`, `anova/manova.py`, `factor_analysis/efa.py`

### 5.5 Full test suite
```bash
uv run pytest tests/ -v --tb=short
```
**Gate:** 100% pass rate. All tests green before declaring migration complete.

### 5.6 Final lint + type gate
```bash
uv run ruff check stats_core/ tests/ && uv run mypy stats_core/
```

---

## Summary Table

| Phase | Key Deliverable | Gate Command |
|---|---|---|
| 1 ✓ | `pyproject.toml`, directory scaffold | `uv run python -c "import stats_core"` |
| 2 | `preprocessing.py` + tests | `uv run pytest tests/test_preprocessing.py -v` |
| 3 | `factor_analysis/efa.py`, `anova/one_way.py`, `anova/manova.py` + tests | `uv run pytest tests/test_efa.py tests/test_anova.py tests/test_manova.py -v` |
| 4 | `visualization.py`, `sem.py` + tests | `uv run pytest tests/test_sem.py tests/test_visualization.py -v` |
| 5 | Type hints, lint, `__all__`, full suite | `uv run pytest tests/ -v && uv run ruff check stats_core/ tests/` |
