"""Data preprocessing utilities for scale development and analysis."""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names in a DataFrame and drop all-NaN rows.

    Strips content outside parentheses (e.g. survey export prefixes),
    lowercases all column names, and removes rows where every value is NaN.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame with cleaned column names and all-NaN rows removed.
    """
    df = df.copy()

    # Remove all characters before "("
    df.columns = df.columns.map(lambda x: x.split("(", 1)[1] if "(" in x else x)

    # Remove all characters after ")"
    df.columns = df.columns.map(lambda x: x.split(")", 1)[0] if ")" in x else x)

    # Rename columns to lowercase
    df.columns = df.columns.str.lower()

    # Remove any rows where all data is missing
    df = df.dropna(how="all")

    return df


def corrected_item_total_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate corrected item-total correlations (item-rest correlations).

    For each item, correlates it with the sum of all *other* items,
    excluding the item itself from the total (hence "corrected").

    Args:
        df: DataFrame where rows are respondents and columns are items.

    Returns:
        DataFrame with columns ``Item`` and ``Corrected_Item_Total_Correlation``,
        sorted by correlation descending.
    """
    total = df.sum(axis=1)
    rest = df.apply(lambda col: total - col)
    correlations = df.corrwith(rest)

    result = (
        correlations.rename_axis("Item")
        .reset_index(name="Corrected_Item_Total_Correlation")
        .sort_values("Corrected_Item_Total_Correlation", ascending=False)
    )
    return result


def vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factors for multicollinearity diagnostics.

    Args:
        df: DataFrame of predictor variables (no constant column needed).

    Returns:
        DataFrame with columns ``feature`` and ``VIF``.
    """
    X = add_constant(df)
    result = pd.DataFrame(
        {
            "feature": X.columns,
            "VIF": [
                variance_inflation_factor(X.values, i) for i in range(len(X.columns))
            ],
        }
    )
    return result


def scale_totals(df: pd.DataFrame, subscales: dict[str, list[str]]) -> pd.DataFrame:
    """Add subscale total columns to a DataFrame.

    Args:
        df: DataFrame containing item columns.
        subscales: Mapping of subscale name to list of item column names.
            A column ``<name>_total`` is added for each entry.

    Returns:
        DataFrame with a ``<name>_total`` column appended for each subscale.

    Example:
        >>> scale_totals(df, {"inclusion": ["inc1", "inc2", "inc3"]})
    """
    df = df.copy()
    for name, items in subscales.items():
        df[f"{name}_total"] = df[items].sum(axis=1)
    return df
