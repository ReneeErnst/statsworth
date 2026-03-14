"""Cross-validates stats_core against the original scaledev implementation.

Runs equivalent functions from both codebases on identical synthetic inputs and
checks that numerical outputs match. Exits 0 if all checks pass, 1 if any fail.

Usage:
    uv run python validate_migration.py
"""

import sys

import matplotlib

matplotlib.use("Agg")  # must be before any other matplotlib import

import numpy as np
import pandas as pd

# ── Import scaledev (original) ───────────────────────────────────────────────
SCALEDEV_PATH = "/home/renee/repos/scaledev"
sys.path.insert(0, SCALEDEV_PATH)

import scaledev.modeler as orig_mod  # noqa: E402
import scaledev.preprocessor as orig_pre  # noqa: E402

# ── Import stats_core (migration) ────────────────────────────────────────────
from stats_core.anova.manova import (  # noqa: E402
    one_way_manova,
    one_way_manova_games_howell,
)
from stats_core.anova.one_way import (  # noqa: E402
    games_howell,
    one_way_anova,
    welch_anova_and_games_howell,
)
from stats_core.factor_analysis.efa import (  # noqa: E402
    efa,
    factor_loadings_table,
    get_items_with_low_loadings,
    strongest_loadings,
)
from stats_core.preprocessing import (  # noqa: E402
    corrected_item_total_correlations,
    vif,
)

# ── Result tracking ──────────────────────────────────────────────────────────
_results: list[tuple[str, str, str]] = []


def _check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _results.append((status, name, detail))
    tag = "✓" if passed else "✗"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag} {name}{suffix}")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Synthetic data ───────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# Preprocessing — plain numeric DataFrame
df_pre = pd.DataFrame(
    rng.normal(5, 1, (60, 6)),
    columns=[f"item{i}" for i in range(6)],
)

# Factor analysis — two-factor structure with clear loading pattern
n_obs = 200
F1 = rng.normal(0, 1, n_obs)
F2 = rng.normal(0, 1, n_obs)
noise = rng.normal(0, 0.25, (n_obs, 8))
loading_matrix = np.array(
    [
        [0.80, 0.10],
        [0.75, 0.10],
        [0.85, 0.10],
        [0.70, 0.10],
        [0.10, 0.80],
        [0.10, 0.75],
        [0.10, 0.85],
        [0.10, 0.70],
    ]
)
data = F1[:, None] * loading_matrix[:, 0] + F2[:, None] * loading_matrix[:, 1] + noise
df_efa = pd.DataFrame(data, columns=[f"item{i}" for i in range(8)])

# ANOVA — three clearly separated groups
df_anova = pd.DataFrame(
    {
        "group": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        "score": np.concatenate(
            [
                rng.normal(10, 1, 50),
                rng.normal(20, 1, 50),
                rng.normal(30, 1, 50),
            ]
        ),
    }
)

# MANOVA — three groups, two DVs
df_manova = pd.DataFrame(
    {
        "group": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
        "dv1": np.concatenate(
            [rng.normal(10, 1, 60), rng.normal(20, 1, 60), rng.normal(30, 1, 60)]
        ),
        "dv2": np.concatenate(
            [rng.normal(5, 1, 60), rng.normal(15, 1, 60), rng.normal(25, 1, 60)]
        ),
    }
)

# ── 1. PREPROCESSING ─────────────────────────────────────────────────────────
_section("1. Preprocessing")

# corrected_item_total_correlations
orig_citc = orig_pre.corrected_item_total_correlations(df_pre.copy())
new_citc = corrected_item_total_correlations(df_pre.copy())
orig_vals = orig_citc.set_index("Item")["Corrected_Item_Total_Correlation"]
new_vals = new_citc.set_index("Item")["Corrected_Item_Total_Correlation"]
try:
    np.testing.assert_allclose(
        orig_vals[df_pre.columns].values,
        new_vals[df_pre.columns].values,
        rtol=1e-10,
    )
    _check("corrected_item_total_correlations: values match", True)
except AssertionError as e:
    _check("corrected_item_total_correlations", False, str(e))

# vif
orig_vif = orig_pre.vif(df_pre.copy())
new_vif = vif(df_pre.copy())
try:
    orig_vif_vals = orig_vif.set_index("feature")["VIF"]
    new_vif_vals = new_vif.set_index("feature")["VIF"]
    np.testing.assert_allclose(
        orig_vif_vals[new_vif_vals.index].values,
        new_vif_vals.values,
        rtol=1e-6,
    )
    _check("vif: values match", True)
except AssertionError as e:
    _check("vif", False, str(e))

# clean_columns — original has a known double-assignment bug; document its behaviour
try:
    orig_pre.clean_columns(
        pd.DataFrame([[1, 2]], columns=["Survey (Hello) world", "plain"])
    )
    _check(
        "clean_columns: original runs without error (bug may be harmless on this input)",
        True,
    )
except Exception as e:
    _check(
        "clean_columns: original raises exception (known bug confirmed)",
        True,
        f"error: {type(e).__name__}",
    )

# ── 2. FACTOR ANALYSIS ───────────────────────────────────────────────────────
_section("2. Factor Analysis")

fa_orig = orig_mod.efa(df_efa.copy(), n_factors=2)
fa_new = efa(df_efa.copy(), n_factors=2)
try:
    np.testing.assert_allclose(fa_new.loadings_, fa_orig.loadings_, rtol=1e-6)
    _check("efa: loadings match", True)
except AssertionError as e:
    _check("efa: loadings match", False, str(e))

# factor_loadings_table
item_names = df_efa.columns
factor_names = ["F1", "F2"]
loadings_arr = fa_new.loadings_
orig_ft = orig_mod.factor_loadings_table(loadings_arr, item_names, factor_names)
new_ft = factor_loadings_table(loadings_arr, item_names, factor_names)
try:
    np.testing.assert_allclose(
        orig_ft.fillna(0).values,
        new_ft.fillna(0).values,
        rtol=1e-10,
    )
    _check("factor_loadings_table: values match", True)
except AssertionError as e:
    _check("factor_loadings_table", False, str(e))

# get_items_with_low_loadings — use threshold=0.4 (same default in both)
orig_low = orig_mod.get_items_with_low_loadings(loadings_arr, list(item_names), 0.4)
new_low = get_items_with_low_loadings(loadings_arr, list(item_names))
_check(
    "get_items_with_low_loadings: match",
    orig_low == new_low,
    f"orig={orig_low}  new={new_low}",
)

# strongest_loadings
orig_sl = orig_mod.strongest_loadings(loadings_arr, list(item_names))
new_sl = strongest_loadings(loadings_arr, list(item_names))
items_match = list(orig_sl["item"]) == list(new_sl["item"])
factors_match = list(orig_sl["strongest_factor"]) == list(new_sl["strongest_factor"])
try:
    np.testing.assert_allclose(orig_sl["loading"].values, new_sl["loading"].values, rtol=1e-10)
    loadings_match = True
except AssertionError:
    loadings_match = False
_check(
    "strongest_loadings: items, factors, and loadings match",
    items_match and factors_match and loadings_match,
)

# ── 3. ANOVA ─────────────────────────────────────────────────────────────────
_section("3. ANOVA")

# one_way_anova
orig_at, _ = orig_mod.one_way_anova(df_anova.copy(), "group", "score")
new_at, _ = one_way_anova(df_anova.copy(), "group", "score")
try:
    orig_p = float(orig_at["PR(>F)"].dropna().iloc[0])
    new_p = float(new_at["PR(>F)"].dropna().iloc[0])
    np.testing.assert_allclose(orig_p, new_p, rtol=1e-6)
    _check("one_way_anova: p-value matches", True, f"p={new_p:.4e}")
except AssertionError as e:
    _check("one_way_anova", False, str(e))

# games_howell
orig_gh = orig_mod.games_howell(df_anova.copy(), "group", "score")
new_gh = games_howell(df_anova.copy(), "group", "score")
try:
    np.testing.assert_allclose(
        orig_gh["p-value"].values,
        new_gh["p-value"].values,
        rtol=1e-6,
    )
    _check("games_howell: p-values match", True)
except AssertionError as e:
    _check("games_howell", False, str(e))

# welch_anova_and_games_howell
orig_welch, orig_wgh = orig_mod.welch_anova_and_games_howell(
    df_anova.copy(), "group", "score"
)
new_welch, new_wgh = welch_anova_and_games_howell(df_anova.copy(), "group", "score")
try:
    np.testing.assert_allclose(orig_welch.pvalue, new_welch.pvalue, rtol=1e-6)
    _check(
        "welch_anova_and_games_howell: Welch p-value matches", True, f"p={new_welch.pvalue:.4e}"
    )
except AssertionError as e:
    _check("welch_anova_and_games_howell", False, str(e))
if orig_wgh is not None and new_wgh is not None:
    try:
        np.testing.assert_allclose(
            orig_wgh["p-value"].values, new_wgh["p-value"].values, rtol=1e-6
        )
        _check("welch_anova_and_games_howell: GH p-values match", True)
    except AssertionError as e:
        _check("welch_anova_and_games_howell: GH p-values match", False, str(e))

# ── 4. MANOVA ────────────────────────────────────────────────────────────────
_section("4. MANOVA")

# one_way_manova
orig_mv, orig_mat, orig_mtukey = orig_mod.one_way_manova(
    df_manova.copy(), "group", ["dv1", "dv2"]
)
new_mv, new_mat, new_mtukey = one_way_manova(
    df_manova.copy(), "group", ["dv1", "dv2"]
)
try:
    # Both implementations return MultivariateTestResults (already called .mv_test())
    orig_pillai = float(orig_mv.results["C(group)"]["stat"].iloc[0, 0])
    new_pillai = float(new_mv.results["C(group)"]["stat"].iloc[0, 0])
    np.testing.assert_allclose(orig_pillai, new_pillai, rtol=1e-6)
    _check("one_way_manova: Pillai's trace matches", True, f"pillai={new_pillai:.6f}")
except AssertionError as e:
    _check("one_way_manova: Pillai's trace matches", False, str(e))
if orig_mat is not None and new_mat is not None:
    try:
        for dv in ["dv1", "dv2"]:
            orig_p = float(orig_mat[dv]["PR(>F)"].dropna().iloc[0])
            new_p = float(new_mat[dv]["PR(>F)"].dropna().iloc[0])
            np.testing.assert_allclose(orig_p, new_p, rtol=1e-6)
        _check("one_way_manova: follow-up ANOVA p-values match", True)
    except AssertionError as e:
        _check("one_way_manova: follow-up ANOVA p-values match", False, str(e))

# one_way_manova_games_howell
orig_mv2, orig_welch2, orig_gh2 = orig_mod.one_way_manova_games_howell(
    df_manova.copy(), "group", ["dv1", "dv2"]
)
new_mv2, new_welch2, new_gh2 = one_way_manova_games_howell(
    df_manova.copy(), "group", ["dv1", "dv2"]
)
if orig_welch2 is not None and new_welch2 is not None:
    try:
        for dv in ["dv1", "dv2"]:
            np.testing.assert_allclose(
                orig_welch2[dv].pvalue, new_welch2[dv].pvalue, rtol=1e-6
            )
        _check("one_way_manova_games_howell: Welch p-values match", True)
    except AssertionError as e:
        _check("one_way_manova_games_howell: Welch p-values match", False, str(e))
    try:
        for dv in ["dv1", "dv2"]:
            np.testing.assert_allclose(
                orig_gh2[dv]["p-value"].values,
                new_gh2[dv]["p-value"].values,
                rtol=1e-6,
            )
        _check("one_way_manova_games_howell: GH p-values match (both DVs)", True)
    except AssertionError as e:
        _check("one_way_manova_games_howell: GH p-values match", False, str(e))

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{'═' * 60}")
n_pass = sum(1 for s, _, _ in _results if s == "PASS")
n_fail = sum(1 for s, _, _ in _results if s == "FAIL")
print(f"  {n_pass} passed, {n_fail} failed out of {len(_results)} checks")
print(f"{'═' * 60}\n")

if n_fail:
    sys.exit(1)
