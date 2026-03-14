import numpy as np
import pandas as pd
import pytest

from stats_core.preprocessing import (
    clean_columns,
    corrected_item_total_correlations,
    scale_totals,
    vif,
)


class TestCleanColumns:
    def test_normalizes_to_lowercase(self):
        df = pd.DataFrame({"Item_A": [1], "ITEM_B": [2]})
        result = clean_columns(df)
        assert list(result.columns) == ["item_a", "item_b"]

    def test_strips_prefix_before_parenthesis(self):
        df = pd.DataFrame({"Q1 (item_a)": [1], "Q2 (item_b)": [2]})
        result = clean_columns(df)
        assert list(result.columns) == ["item_a", "item_b"]

    def test_drops_all_nan_rows(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, np.nan, 6.0]})
        result = clean_columns(df)
        assert len(result) == 2
        assert result["a"].tolist() == [1.0, 3.0]

    def test_retains_partial_nan_rows(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [4.0, 5.0]})
        result = clean_columns(df)
        assert len(result) == 2

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"ITEM_A": [1]})
        clean_columns(df)
        assert list(df.columns) == ["ITEM_A"]


class TestCorrectedItemTotalCorrelations:
    def test_returns_correct_columns(self):
        rng = np.random.default_rng(42)
        cols = [f"item{i}" for i in range(5)]
        df = pd.DataFrame(rng.integers(1, 6, size=(100, 5)), columns=cols)
        result = corrected_item_total_correlations(df)
        assert list(result.columns) == ["Item", "Corrected_Item_Total_Correlation"]

    def test_all_items_present(self):
        rng = np.random.default_rng(42)
        cols = [f"item{i}" for i in range(5)]
        df = pd.DataFrame(rng.integers(1, 6, size=(100, 5)), columns=cols)
        result = corrected_item_total_correlations(df)
        assert set(result["Item"]) == set(df.columns)

    def test_correlations_in_valid_range(self):
        rng = np.random.default_rng(42)
        cols = [f"item{i}" for i in range(5)]
        df = pd.DataFrame(rng.integers(1, 6, size=(100, 5)), columns=cols)
        result = corrected_item_total_correlations(df)
        assert result["Corrected_Item_Total_Correlation"].between(-1, 1).all()

    def test_sorted_descending(self):
        rng = np.random.default_rng(42)
        cols = [f"item{i}" for i in range(5)]
        df = pd.DataFrame(rng.integers(1, 6, size=(100, 5)), columns=cols)
        result = corrected_item_total_correlations(df)
        corrs = result["Corrected_Item_Total_Correlation"].tolist()
        assert corrs == sorted(corrs, reverse=True)

    def test_high_correlation_item_ranks_first(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 100)
        # item0 strongly correlated with rest, item4 is noise
        df = pd.DataFrame(
            {
                "item0": base + rng.normal(0, 0.1, 100),
                "item1": base + rng.normal(0, 0.1, 100),
                "item2": base + rng.normal(0, 0.1, 100),
                "item3": base + rng.normal(0, 0.1, 100),
                "item4": rng.normal(0, 1, 100),
            }
        )
        result = corrected_item_total_correlations(df)
        top_items = result.head(4)["Item"].tolist()
        assert "item4" not in top_items


class TestVif:
    def test_returns_correct_columns(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
        result = vif(df)
        assert "feature" in result.columns
        assert "VIF" in result.columns

    def test_high_vif_for_collinear_features(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        df = pd.DataFrame(
            {"a": x, "b": x + rng.normal(0, 0.01, 100), "c": rng.normal(size=100)}
        )
        result = vif(df)
        feature_vifs = result[result["feature"] != "const"].set_index("feature")["VIF"]
        assert feature_vifs["a"] > 10
        assert feature_vifs["b"] > 10

    def test_low_vif_for_independent_features(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
        result = vif(df)
        feature_vifs = result[result["feature"] != "const"].set_index("feature")["VIF"]
        assert (feature_vifs < 5).all()


class TestScaleTotals:
    def test_adds_total_columns(self):
        df = pd.DataFrame({"a1": [1, 2], "a2": [3, 4], "b1": [5, 6], "b2": [7, 8]})
        result = scale_totals(df, {"sub_a": ["a1", "a2"], "sub_b": ["b1", "b2"]})
        assert "sub_a_total" in result.columns
        assert "sub_b_total" in result.columns

    def test_correct_sums(self):
        df = pd.DataFrame({"a1": [1, 2], "a2": [3, 4]})
        result = scale_totals(df, {"sub_a": ["a1", "a2"]})
        assert result["sub_a_total"].tolist() == [4, 6]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a1": [1], "a2": [2]})
        scale_totals(df, {"sub_a": ["a1", "a2"]})
        assert "sub_a_total" not in df.columns

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a1": [1]})
        with pytest.raises(KeyError):
            scale_totals(df, {"sub_a": ["a1", "nonexistent"]})
