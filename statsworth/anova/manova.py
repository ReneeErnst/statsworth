"""One-way MANOVA with follow-up analyses."""

from collections.abc import Callable

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statsworth.anova.one_way import DEFAULT_ALPHA, games_howell


def _run_manova(
    df: pd.DataFrame,
    group_col: str,
    dv_cols: list[str],
    post_hoc_fn: Callable,
    alpha: float = DEFAULT_ALPHA,
) -> tuple:
    """Fit a one-way MANOVA and conditionally run follow-up analyses.

    Validates group sizes, fits the MANOVA, checks significance via Pillai's
    Trace, and delegates to ``post_hoc_fn`` when significant.

    Args:
        df: DataFrame containing the data.
        group_col: Column name for the grouping variable.
        dv_cols: List of dependent variable column names.
        post_hoc_fn: Callable ``(df, group_col, dv_cols, alpha) -> (dict, dict)``
            that performs follow-up tests and returns two result dicts.
        alpha: Significance level for Pillai's Trace check (default
            ``DEFAULT_ALPHA``).

    Returns:
        Tuple of (MANOVA results, follow-up dict A or None, follow-up dict B or None).

    Raises:
        ValueError: If the smallest group is not larger than the number of DVs.
    """
    min_group_size = df.groupby(group_col).size().min()
    if min_group_size <= len(dv_cols):
        raise ValueError(
            f"MANOVA not suitable. The smallest group size ({min_group_size}) "
            f"must be greater than the number of dependent variables ({len(dv_cols)})."
        )

    formula = f"{' + '.join(dv_cols)} ~ C({group_col})"
    manova_results = MANOVA.from_formula(formula, data=df).mv_test()

    pillai_p = manova_results.results[f"C({group_col})"]["stat"].iloc[0, 4]
    if pillai_p < alpha:
        follow_up_a, follow_up_b = post_hoc_fn(df, group_col, dv_cols, alpha)
        return manova_results, follow_up_a, follow_up_b

    return manova_results, None, None


def _tukey_follow_up(df: pd.DataFrame, group_col: str, dv_cols: list[str], alpha: float) -> tuple[dict, dict]:
    anova_tables: dict = {}
    tukey_results: dict = {}
    for dv in dv_cols:
        model = smf.ols(f"{dv} ~ C({group_col})", data=df).fit()
        anova_tables[dv] = sm.stats.anova_lm(model, typ=1)
        tukey_results[dv] = pairwise_tukeyhsd(df[dv], df[group_col])
    return anova_tables, tukey_results


def _games_howell_follow_up(df: pd.DataFrame, group_col: str, dv_cols: list[str], alpha: float) -> tuple[dict, dict]:
    welch_tables: dict = {}
    gh_results: dict = {}
    for dv in dv_cols:
        welch = sms.anova_oneway(df[dv], df[group_col], use_var="unequal")
        welch_tables[dv] = welch
        gh_results[dv] = games_howell(df, group_col, dv) if welch.pvalue < alpha else None
    return welch_tables, gh_results


def one_way_manova(df: pd.DataFrame, group_col: str, dv_cols: list[str]) -> tuple:
    """Perform a one-way MANOVA with Tukey's HSD follow-up.

    Follow-up ANOVAs and Tukey tests are only run when the overall MANOVA
    is significant (Pillai's Trace p < 0.05).

    Args:
        df: DataFrame containing the data.
        group_col: Column name for the grouping variable.
        dv_cols: List of dependent variable column names.

    Returns:
        Tuple of (MANOVA results, dict of ANOVA tables, dict of Tukey results).
        The second and third elements are ``None`` when MANOVA is not significant.

    Raises:
        ValueError: If the smallest group is not larger than the number of DVs.
    """
    return _run_manova(df, group_col, dv_cols, _tukey_follow_up)


def one_way_manova_games_howell(
    df: pd.DataFrame,
    group_col: str,
    dv_cols: list[str],
    alpha: float = DEFAULT_ALPHA,
) -> tuple:
    """Perform a one-way MANOVA with Welch ANOVA and Games-Howell follow-up.

    Follow-up tests are only run when the overall MANOVA is significant
    (Pillai's Trace p < ``alpha``). Within follow-ups, Games-Howell is only
    run per DV when Welch's ANOVA is also significant.

    Args:
        df: DataFrame containing the data.
        group_col: Column name for the grouping variable.
        dv_cols: List of dependent variable column names.
        alpha: Significance level (default ``DEFAULT_ALPHA``).

    Returns:
        Tuple of (MANOVA results, dict of Welch ANOVA results, dict of
        Games-Howell results). The second and third elements are ``None``
        when MANOVA is not significant.

    Raises:
        ValueError: If the smallest group is not larger than the number of DVs.
    """
    return _run_manova(df, group_col, dv_cols, _games_howell_follow_up, alpha)
