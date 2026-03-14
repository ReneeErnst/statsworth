import numpy as np
import pandas as pd
import pytest

from stats_core.factor_analysis.efa import (
    LOW_LOADING_THRESHOLD,
    cronbach_alpha,
    efa,
    factor_loadings_table,
    get_items_with_low_loadings,
    no_low_loadings_solution,
    strongest_loadings,
)


def _make_factor_data(seed: int = 42) -> pd.DataFrame:
    """Synthetic 3-factor dataset with 9 items (3 per factor)."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=200)
    f2 = rng.normal(size=200)
    f3 = rng.normal(size=200)
    noise = lambda: rng.normal(0, 0.3, 200)  # noqa: E731
    return pd.DataFrame(
        {
            "i1": f1 + noise(),
            "i2": f1 + noise(),
            "i3": f1 + noise(),
            "i4": f2 + noise(),
            "i5": f2 + noise(),
            "i6": f2 + noise(),
            "i7": f3 + noise(),
            "i8": f3 + noise(),
            "i9": f3 + noise(),
        }
    )


class TestEfa:
    def test_returns_factor_analyzer(self):
        from factor_analyzer.factor_analyzer import FactorAnalyzer

        df = _make_factor_data()
        result = efa(df, n_factors=3)
        assert isinstance(result, FactorAnalyzer)

    def test_loadings_shape(self):
        df = _make_factor_data()
        result = efa(df, n_factors=3)
        assert result.loadings_.shape == (9, 3)

    def test_default_method_and_rotation(self):
        df = _make_factor_data()
        result = efa(df, n_factors=2)
        assert result.method == "ml"
        assert result.rotation == "oblimin"

    def test_accepts_numpy_array(self):
        df = _make_factor_data()
        result = efa(df.values, n_factors=3)
        assert result.loadings_.shape == (9, 3)

    def test_n_factors_none_uses_default(self):
        df = _make_factor_data()
        result = efa(df)
        assert result.loadings_ is not None


class TestParallelAnalysis:
    def test_returns_int(self):
        from stats_core.factor_analysis.efa import parallel_analysis

        df = _make_factor_data()
        result = parallel_analysis(df, K=5)
        assert isinstance(result, int)

    def test_result_in_valid_range(self):
        from stats_core.factor_analysis.efa import parallel_analysis

        df = _make_factor_data()
        result = parallel_analysis(df, K=5)
        assert 1 <= result <= len(df.columns)

    def test_suggests_three_factors_for_three_factor_data(self):
        from stats_core.factor_analysis.efa import parallel_analysis

        df = _make_factor_data(seed=0)
        result = parallel_analysis(df, K=10)
        # Strong 3-factor structure should suggest 3 factors
        assert result == 3

    def test_print_eigenvalues(self, capsys):
        from stats_core.factor_analysis.efa import parallel_analysis

        df = _make_factor_data()
        parallel_analysis(df, K=5, print_eigenvalues=True)
        captured = capsys.readouterr()
        assert "Factor eigenvalues" in captured.out

    def test_show_scree_plot(self):
        from stats_core.factor_analysis.efa import parallel_analysis

        df = _make_factor_data()
        result = parallel_analysis(df, K=5, show_scree_plot=True, max_scree_factors=9)
        assert isinstance(result, int)


class TestFactorLoadingsTable:
    def test_shape(self):
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        factor_names = ["F1", "F2", "F3"]
        result = factor_loadings_table(fa.loadings_, df.columns, factor_names)
        assert result.shape == (9, 3)

    def test_column_names(self):
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        factor_names = ["F1", "F2", "F3"]
        result = factor_loadings_table(fa.loadings_, df.columns, factor_names)
        assert list(result.columns) == factor_names

    def test_index_is_item_names(self):
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        result = factor_loadings_table(fa.loadings_, df.columns, ["F1", "F2", "F3"])
        assert list(result.index) == list(df.columns)

    def test_single_factor(self):
        loadings = np.array([[0.8], [0.7], [0.6]])
        item_names = pd.Index(["i1", "i2", "i3"])
        result = factor_loadings_table(loadings, item_names, ["F1"])
        assert result.shape == (3, 1)
        assert list(result.columns) == ["F1"]
        assert list(result.index) == ["i1", "i2", "i3"]

    def test_per_factor_item_names(self):
        # Pass a list-of-lists: one name list per factor (exercises the else branch)
        loadings = np.array([[0.8, 0.1], [0.7, 0.2], [0.1, 0.9]])
        item_names = [["i1", "i2", "i3"], ["i1", "i2", "i3"]]
        result = factor_loadings_table(loadings, item_names, ["F1", "F2"])
        assert result.shape == (3, 2)
        assert list(result.columns) == ["F1", "F2"]


class TestGetItemsWithLowLoadings:
    def test_returns_low_loading_items(self):
        # Manually construct loadings: item0 has low loadings, item1 has high
        loadings = np.array([[0.1, 0.2], [0.8, 0.1]])
        items = ["low_item", "high_item"]
        result = get_items_with_low_loadings(loadings, items, threshold=0.4)
        assert result == ["low_item"]

    def test_high_loading_items_excluded(self):
        loadings = np.array([[0.9, 0.1], [0.8, 0.2]])
        items = ["item_a", "item_b"]
        result = get_items_with_low_loadings(loadings, items, threshold=0.4)
        assert result == []

    def test_uses_absolute_value(self):
        # Negative loading of -0.7 should be treated as high (abs = 0.7)
        loadings = np.array([[-0.7, 0.1]])
        items = ["item_a"]
        result = get_items_with_low_loadings(loadings, items, threshold=0.4)
        assert result == []

    def test_default_threshold_is_constant(self):
        loadings = np.array([[LOW_LOADING_THRESHOLD - 0.01, 0.0]])
        items = ["borderline"]
        result = get_items_with_low_loadings(loadings, items)
        assert result == ["borderline"]


class TestNoLowLoadingsSolution:
    def test_returns_dataframe_and_model(self):
        from factor_analyzer.factor_analyzer import FactorAnalyzer

        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        low = get_items_with_low_loadings(fa.loadings_, list(df.columns))
        result_df, result_model = no_low_loadings_solution(df, low, n_factors=3)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(result_model, FactorAnalyzer)

    def test_removes_low_loading_items(self):
        """Add a pure-noise item; it should be dropped after iteration."""
        rng = np.random.default_rng(7)
        df = _make_factor_data()
        df["noise"] = rng.normal(size=200)
        fa = efa(df, n_factors=3)
        low = get_items_with_low_loadings(fa.loadings_, list(df.columns))
        result_df, _ = no_low_loadings_solution(df, low, n_factors=3)
        assert "noise" not in result_df.columns

    def test_no_low_loadings_remain(self):
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        low = get_items_with_low_loadings(fa.loadings_, list(df.columns))
        result_df, result_model = no_low_loadings_solution(df, low, n_factors=3)
        remaining_low = get_items_with_low_loadings(
            result_model.loadings_, list(result_df.columns)
        )
        assert remaining_low == []


class TestCronbachAlpha:
    def test_returns_float_and_array(self):
        import numpy as np

        df = _make_factor_data()
        alpha, ci = cronbach_alpha(df)
        assert isinstance(float(alpha), float)
        assert len(ci) == 2

    def test_high_internal_consistency(self):
        """Strongly correlated items should yield alpha > 0.9."""
        df = _make_factor_data()
        # Use only the first 3 items (all loading on the same factor)
        alpha, _ = cronbach_alpha(df[["i1", "i2", "i3"]])
        assert float(alpha) > 0.9

    def test_alpha_in_valid_range(self):
        df = _make_factor_data()
        alpha, _ = cronbach_alpha(df)
        assert 0.0 <= float(alpha) <= 1.0

    def test_ci_lower_less_than_upper(self):
        df = _make_factor_data()
        _, ci = cronbach_alpha(df)
        assert ci[0] < ci[1]


class TestStrongestLoadings:
    def test_returns_all_items(self):
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        result = strongest_loadings(fa.loadings_, list(df.columns))
        assert set(result["item"]) == set(df.columns)

    def test_columns_present(self):
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        result = strongest_loadings(fa.loadings_, list(df.columns))
        assert list(result.columns) == ["item", "strongest_factor", "loading"]

    def test_factor_labels_are_integers(self):
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        result = strongest_loadings(fa.loadings_, list(df.columns))
        assert result["strongest_factor"].isin([1, 2, 3]).all()

    def test_items_map_to_correct_factor(self):
        """Items i1-i3 (factor 1), i4-i6 (factor 2), i7-i9 (factor 3)."""
        df = _make_factor_data()
        fa = efa(df, n_factors=3)
        result = strongest_loadings(fa.loadings_, list(df.columns))
        item_to_factor = result.set_index("item")["strongest_factor"].to_dict()
        # All items sharing the same true factor should map to the same model factor
        group1 = {item_to_factor["i1"], item_to_factor["i2"], item_to_factor["i3"]}
        group2 = {item_to_factor["i4"], item_to_factor["i5"], item_to_factor["i6"]}
        group3 = {item_to_factor["i7"], item_to_factor["i8"], item_to_factor["i9"]}
        assert len(group1) == 1
        assert len(group2) == 1
        assert len(group3) == 1
        assert len({group1.pop(), group2.pop(), group3.pop()}) == 3
