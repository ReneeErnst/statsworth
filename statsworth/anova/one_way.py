"""One-way ANOVA and post-hoc tests."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DEFAULT_ALPHA = 0.05


def one_way_anova(df: pd.DataFrame, group_col: str, dv_col: str) -> tuple:
    """Perform a one-way ANOVA with Tukey's HSD post-hoc test.

    Args:
        df: DataFrame containing the data.
        group_col: Column name for the grouping variable.
        dv_col: Column name for the dependent variable.

    Returns:
        Tuple of (ANOVA table DataFrame, TukeyHSDResults object).
    """
    model = ols(f"{dv_col} ~ C({group_col})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)
    tukey_results = pairwise_tukeyhsd(df[dv_col], df[group_col])
    return anova_table, tukey_results


def welch_anova_and_games_howell(
    df: pd.DataFrame, group_col: str, dv_col: str
) -> tuple:
    """Perform Welch's ANOVA with Games-Howell post-hoc test.

    Games-Howell is only computed when Welch's ANOVA is significant
    (p < ``DEFAULT_ALPHA``).

    Args:
        df: DataFrame containing the data.
        group_col: Column name for the grouping variable.
        dv_col: Column name for the dependent variable.

    Returns:
        Tuple of (Welch ANOVA result, Games-Howell DataFrame or None).
    """
    welch_result = sms.anova_oneway(df[dv_col], df[group_col], use_var="unequal")

    if welch_result.pvalue < DEFAULT_ALPHA:
        gh_results = games_howell(df, group_col, dv_col)
        return welch_result, gh_results
    else:
        return welch_result, None


def games_howell(df: pd.DataFrame, group_col: str, dv_col: str) -> pd.DataFrame:
    """Perform the Games-Howell post-hoc test for unequal variances.

    Computes all pairwise group comparisons using Welch-style degrees of
    freedom and the studentized range distribution for p-value calculation.

    Args:
        df: DataFrame containing the data.
        group_col: Column name for the grouping variable.
        dv_col: Column name for the dependent variable.

    Returns:
        DataFrame with columns ``Group1``, ``Group2``, ``Mean Diff``, ``t``,
        ``df``, and ``p-value``, one row per pair.
    """
    groups = df[group_col].unique()
    results = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = df.loc[df[group_col] == groups[i], dv_col].to_numpy()
            group2 = df.loc[df[group_col] == groups[j], dv_col].to_numpy()

            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1 = np.var(group1, ddof=1)
            var2 = np.var(group2, ddof=1)

            t_stat = (mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))
            df_num = (var1 / n1 + var2 / n2) ** 2
            df_denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            df_val = df_num / df_denom

            p_val = psturng(np.abs(t_stat) * np.sqrt(2), len(groups), df_val)
            results.append([groups[i], groups[j], mean1 - mean2, t_stat, df_val, p_val])

    columns = ["Group1", "Group2", "Mean Diff", "t", "df", "p-value"]
    return pd.DataFrame(results, columns=columns)
