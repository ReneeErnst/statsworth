"""Exploratory Factor Analysis (EFA) utilities."""

from typing import Tuple

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import FactorAnalyzer

LOW_LOADING_THRESHOLD = 0.4


def efa(
    df: pd.DataFrame | np.ndarray,
    n_factors: int | None = None,
    method: str = "ml",
    rotation: str = "oblimin",
) -> FactorAnalyzer:
    """Perform Exploratory Factor Analysis.

    Uses maximum likelihood fitting and oblimin rotation by default,
    which is the preferred method for correlated factors.

    Args:
        df: DataFrame or array containing the scale items/variables.
        n_factors: Number of factors to extract. If None, the number is not
            pre-specified and the analyzer uses its default behaviour.
        method: Fitting method passed to FactorAnalyzer (default ``"ml"``).
        rotation: Rotation method passed to FactorAnalyzer (default ``"oblimin"``).

    Returns:
        Fitted FactorAnalyzer object.
    """
    if n_factors:
        fa = FactorAnalyzer(method=method, rotation=rotation, n_factors=n_factors)
    else:
        fa = FactorAnalyzer(method=method, rotation=rotation)

    fa.fit(df)
    return fa


def parallel_analysis(
    df: pd.DataFrame,
    K: int = 10,
    print_eigenvalues: bool = False,
    show_scree_plot: bool = False,
    max_scree_factors: int = 20,
) -> int:
    """Determine the number of factors to retain via parallel analysis.

    Generates K random datasets of the same shape as ``df``, computes their
    average eigenvalues, and compares them to the real data eigenvalues.
    Factors are retained where the real eigenvalue exceeds the random average
    (and both are positive).

    Args:
        df: Input data for factor analysis.
        K: Number of random-data iterations (default 10).
        print_eigenvalues: Print eigenvalues to stdout if True.
        show_scree_plot: Display a parallel-analysis scree plot if True.
        max_scree_factors: Maximum number of factors shown in the scree plot.

    Returns:
        Suggested number of factors to retain.
    """
    n_rows, n_features = df.shape

    sum_factor_eigens = np.empty(n_features)
    for _ in range(K):
        fa = efa(np.random.normal(size=(n_rows, n_features)))
        f_ev = fa.get_eigenvalues()[1]
        sum_factor_eigens = sum_factor_eigens + f_ev

    avg_factor_eigens = sum_factor_eigens / K

    fa.fit(df)
    data_ev = fa.get_eigenvalues()[1]

    diff = data_ev - avg_factor_eigens
    masked_diff = diff[(data_ev > 0) & (avg_factor_eigens > 0)]
    suggested_factors = int(sum(masked_diff > 0))

    if print_eigenvalues:
        print("Factor eigenvalues for random data:\n", avg_factor_eigens)
        print("Factor eigenvalues for real data:\n", data_ev)

    print(
        f"Parallel analysis suggests that the number of factors = {suggested_factors}"
    )  # noqa: E501

    if show_scree_plot:
        from stats_core.visualization import scree_parallel_analysis

        scree_parallel_analysis(max_scree_factors, avg_factor_eigens, data_ev)

    return suggested_factors


def factor_loadings_table(
    loadings: np.ndarray,
    item_names: pd.Index | list,
    factor_names: list[str],
) -> pd.DataFrame:
    """Format a loadings array as a labelled DataFrame.

    Args:
        loadings: Array of shape ``(n_items, n_factors)`` from a fitted
            FactorAnalyzer.
        item_names: Item names corresponding to the rows of ``loadings``.
            Pass a ``pd.Index`` for a single shared index, or a list of
            per-factor name lists.
        factor_names: Column names for each factor.

    Returns:
        DataFrame of factor loadings with items as the index.
    """
    if loadings.shape[1] == 1 and not isinstance(item_names[0], list):
        item_names = [item_names]

    loadings_dict = {}
    for factor_idx, factor_name in enumerate(factor_names):
        if isinstance(item_names, pd.Index):
            valid_index = item_names
        else:
            valid_index = item_names[factor_idx]
        loadings_dict[factor_name] = pd.Series(
            loadings[:, factor_idx], index=valid_index
        )

    return pd.DataFrame(loadings_dict)


def get_items_with_low_loadings(
    loadings: np.ndarray,
    item_names: list[str],
    threshold: float = LOW_LOADING_THRESHOLD,
) -> list[str]:
    """Return items whose absolute loading is below ``threshold`` on every factor.

    Args:
        loadings: Array of shape ``(n_items, n_factors)``.
        item_names: Item names corresponding to the rows of ``loadings``.
        threshold: Loading magnitude cutoff (default ``LOW_LOADING_THRESHOLD``).

    Returns:
        List of item names with no loading above the threshold.
    """
    return [
        item_names[i]
        for i in range(len(item_names))
        if all(abs(loading) < threshold for loading in loadings[i, :])
    ]


def no_low_loadings_solution(
    df: pd.DataFrame,
    low_loadings: list[str],
    n_factors: int,
) -> Tuple[pd.DataFrame, FactorAnalyzer]:
    """Iteratively remove low-loading items and re-run EFA until none remain.

    Args:
        df: DataFrame for EFA.
        low_loadings: Items identified as low-loading in the initial EFA.
        n_factors: Number of factors to extract at each iteration.

    Returns:
        Tuple of (cleaned DataFrame with low-loading items removed,
        final fitted FactorAnalyzer).
    """
    efa_model = efa(df=df, n_factors=n_factors)

    while len(low_loadings) > 0:
        df = df.drop(columns=low_loadings)
        efa_model = efa(df=df, n_factors=n_factors)
        low_loadings = get_items_with_low_loadings(
            efa_model.loadings_, list(df.columns), threshold=LOW_LOADING_THRESHOLD
        )
        print("Items with low loadings: ", low_loadings)

    return df.reset_index(drop=True), efa_model


def strongest_loadings(loadings: np.ndarray, item_names: list[str]) -> pd.DataFrame:
    """Map each item to the factor it loads most strongly on.

    Args:
        loadings: Array of shape ``(n_items, n_factors)``.
        item_names: Item names corresponding to the rows of ``loadings``.

    Returns:
        DataFrame with columns ``item``, ``strongest_factor``, and ``loading``,
        sorted by factor then loading magnitude descending.
    """
    df_loadings = pd.DataFrame(loadings, index=item_names)
    df_loadings.columns = [i + 1 for i in range(df_loadings.shape[1])]

    strongest_factors = df_loadings.abs().idxmax(axis=1)

    df_result = pd.DataFrame(
        {"item": item_names, "strongest_factor": strongest_factors}
    )
    df_result["loading"] = df_result.apply(
        lambda row: df_loadings.loc[row["item"], row["strongest_factor"]], axis=1
    )

    return df_result.sort_values(
        by=["strongest_factor", "loading"], ascending=[True, False]
    )
