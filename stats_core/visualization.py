"""Visualization utilities for factor analysis, normality checks, and correlations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def highlight_corr(val: float) -> str:
    """Return a CSS background-color style for a correlation value.

    Highlights strong correlations (|r| > 0.6) in green and weak
    correlations (|r| < 0.3) in blue. Used with ``DataFrame.style.map``.

    Args:
        val: A correlation coefficient.

    Returns:
        CSS background-color string.
    """
    if abs(val) > 0.6 and val != 1.0:
        color = "green"
    elif abs(val) < 0.3:
        color = "blue"
    else:
        color = ""
    return f"background-color: {color}"


def efa_item_corr_matrix(
    df: pd.DataFrame,
    cols: list[str],
    title: str | None = None,
) -> "pd.io.formats.style.Styler":
    """Return a styled correlation matrix for EFA item screening.

    Highlights strong correlations (|r| > 0.6) in green and weak correlations
    (|r| < 0.3) in blue to flag candidates for removal during item reduction.
    Only the lower triangle is shown to reduce visual noise.

    Args:
        df: DataFrame containing the data.
        cols: Column names to include in the correlation matrix.
        title: Optional caption displayed above the table.

    Returns:
        Styled pandas Styler object.
    """
    corr = df[cols].corr()
    n = len(corr)

    def _style(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for i in range(n):
            for j in range(n):
                if j > i:
                    styles.iloc[i, j] = "visibility: hidden"
                else:
                    styles.iloc[i, j] = highlight_corr(data.iloc[i, j])
        return styles

    styled = corr.style.apply(_style, axis=None)
    if title:
        styled = styled.set_caption(title)
    return styled


def scree_plot(common_factors_ev: np.ndarray, max_viz: int = 20) -> None:
    """Plot a scree plot of common factor eigenvalues.

    Args:
        common_factors_ev: Array of common factor eigenvalues.
        max_viz: Maximum number of factors to display (default 20).
    """
    plt.scatter(range(1, max_viz + 1), common_factors_ev[:max_viz])
    plt.plot(range(1, max_viz + 1), common_factors_ev[:max_viz])
    plt.title("Scree Plot - Common Factors Eigen values")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.xticks(range(1, max_viz + 1))
    plt.show()


def scree_parallel_analysis(
    max_scree_factors: int,
    avg_factor_eigens: np.ndarray,
    data_ev: np.ndarray,
) -> None:
    """Plot a parallel analysis scree plot overlaying random and real eigenvalues.

    Args:
        max_scree_factors: Number of factors to show on the x-axis.
        avg_factor_eigens: Average eigenvalues from random data iterations.
        data_ev: Eigenvalues from the real data.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, max_scree_factors + 1], [1, 1], "k--", alpha=0.3)
    plt.plot(
        range(1, max_scree_factors + 1),
        avg_factor_eigens[:max_scree_factors],
        "g",
        label="FA - random",
        alpha=0.4,
    )
    plt.scatter(
        range(1, max_scree_factors + 1), data_ev[:max_scree_factors], c="g", marker="o"
    )
    plt.plot(
        range(1, max_scree_factors + 1),
        data_ev[:max_scree_factors],
        "g",
        label="FA - data",
    )
    plt.title("Parallel Analysis Scree Plots", {"fontsize": 20})
    plt.xlabel("Factors", {"fontsize": 15})
    plt.xticks(
        ticks=range(1, max_scree_factors + 1),
        labels=[str(i) for i in range(1, max_scree_factors + 1)],
    )
    plt.ylabel("Eigenvalue", {"fontsize": 15})
    plt.legend()
    plt.show()


def plot_loadings_heatmap(
    loadings: np.ndarray,
    item_names: list,
    factor_names: list[str],
) -> None:
    """Plot a heatmap of factor loadings.

    Args:
        loadings: Array of shape ``(n_items, n_factors)``.
        item_names: Item labels for the y-axis (rows).
        factor_names: Factor labels for the x-axis (columns).
    """
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        loadings,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 8},
        xticklabels=factor_names,
        yticklabels=item_names,
    )
    plt.title("Factor Loadings Heatmap")
    plt.xlabel("Factors")
    plt.ylabel("Items")
    plt.show()


def check_normality(
    df: pd.DataFrame,
    dist_plot: bool = True,
    qq_plot: bool = True,
) -> dict:
    """Check normality assumptions for each column in a DataFrame.

    Computes descriptive statistics, skewness, kurtosis, and a
    Kolmogorov-Smirnov test for each column. Optionally generates
    distribution and Q-Q plots as side effects.

    Args:
        df: DataFrame whose columns are the variables to check.
        dist_plot: Whether to display histogram/KDE distribution plots.
        qq_plot: Whether to display Q-Q plots.

    Returns:
        Dict keyed by column name, each value a dict with keys:
        ``describe``, ``skewness``, ``kurtosis``,
        ``ks_statistic``, ``ks_pvalue``.
    """
    results: dict = {}

    for col in df.columns:
        ks_test = stats.kstest((df[col] - df[col].mean()) / df[col].std(), "norm")

        results[col] = {
            "describe": df[col].describe(),
            "skewness": df[col].skew(),
            "kurtosis": df[col].kurtosis(),
            "ks_statistic": ks_test.statistic,
            "ks_pvalue": ks_test.pvalue,
        }

        if dist_plot or qq_plot:
            fig, ax = plt.subplots(
                1, 2 if dist_plot and qq_plot else 1, figsize=(12, 4)
            )

            if dist_plot:
                sns.histplot(df[col], kde=True, ax=ax[0] if qq_plot else ax)
                (ax[0] if qq_plot else ax).set_title(f"Distribution Plot of {col}")

            if qq_plot:
                stats.probplot(df[col], dist="norm", plot=ax[1] if dist_plot else ax)
                (ax[1] if dist_plot else ax).set_title(f"Q-Q Plot of {col}")

            plt.tight_layout()
            plt.show()

    return results


def scatter_with_regression(df: pd.DataFrame, x: str, y: str) -> float:
    """Plot a scatter plot with a regression line and return the Pearson correlation.

    Displays a seaborn lmplot for the given variable pair.  The plot is shown
    as a side effect; the correlation coefficient is returned so callers can
    inspect or print it.

    Args:
        df: DataFrame containing the variables.
        x: Column name for the x-axis variable.
        y: Column name for the y-axis variable.

    Returns:
        Pearson correlation coefficient between ``x`` and ``y``.
    """
    sns.lmplot(x=x, y=y, data=df)
    plt.show()
    return float(df[x].corr(df[y]))


def corr_heatmap(df: pd.DataFrame, title: str) -> None:
    """Plot a masked lower-triangle correlation heatmap.

    Args:
        df: DataFrame whose columns are correlated.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    corr_df = df.corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, annot=True, cmap="Blues", ax=ax, center=0, mask=mask)
    ax.set_title(title)
    plt.show()
