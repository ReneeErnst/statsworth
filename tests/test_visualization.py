import numpy as np
import pandas as pd
import pytest

from statsworth.visualization import (
    check_normality,
    corr_heatmap,
    efa_item_corr_matrix,
    highlight_corr,
    plot_loadings_heatmap,
    scatter_with_regression,
    scree_parallel_analysis,
    scree_plot,
)


def _make_df(seed: int = 0, n: int = 100, cols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n, cols)),
        columns=[f"var{i}" for i in range(cols)],
    )


class TestHighlightCorr:
    @pytest.mark.parametrize(
        "val,expected_color",
        [
            (0.8, "green"),
            (-0.7, "green"),
            (1.0, ""),  # exactly 1.0 is excluded from green
            (0.5, ""),  # between 0.3 and 0.6 → no highlight
            (0.2, "blue"),
            (-0.2, "blue"),
        ],
    )
    def test_returns_correct_color(self, val, expected_color):
        assert highlight_corr(val) == f"background-color: {expected_color}"


class TestScreePlot:
    @pytest.mark.parametrize(
        "ev,max_viz",
        [
            (np.array([3.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2]), 5),
            (np.ones(25), 20),
        ],
    )
    def test_runs_without_error(self, ev, max_viz):
        scree_plot(ev, max_viz=max_viz)


class TestScreeParallelAnalysis:
    def test_runs_without_error(self):
        rng = np.random.default_rng(0)
        avg_eigens = rng.uniform(0.5, 2.0, 20)
        data_ev = rng.uniform(0.5, 3.0, 20)
        scree_parallel_analysis(10, avg_eigens, data_ev)


class TestPlotLoadingsHeatmap:
    def test_runs_without_error(self):
        rng = np.random.default_rng(0)
        loadings = rng.uniform(-1, 1, size=(6, 2))
        item_names = [f"item{i}" for i in range(6)]
        factor_names = ["F1", "F2"]
        plot_loadings_heatmap(loadings, item_names, factor_names)


class TestCheckNormality:
    def test_runs_without_plots(self):
        df = _make_df()
        result = check_normality(df, dist_plot=False, qq_plot=False)
        assert isinstance(result, dict)

    def test_keys_match_columns(self):
        df = _make_df()
        result = check_normality(df, dist_plot=False, qq_plot=False)
        assert set(result.keys()) == set(df.columns)

    @pytest.mark.parametrize(
        "key", ["describe", "skewness", "kurtosis", "ks_statistic", "ks_pvalue"]
    )
    def test_result_has_expected_keys(self, key):
        df = _make_df(cols=1)
        result = check_normality(df, dist_plot=False, qq_plot=False)
        assert key in result["var0"]

    def test_ks_pvalue_in_valid_range(self):
        df = _make_df()
        result = check_normality(df, dist_plot=False, qq_plot=False)
        for col_result in result.values():
            assert 0.0 <= col_result["ks_pvalue"] <= 1.0

    def test_normal_data_has_high_ks_pvalue(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(size=500)})
        result = check_normality(df, dist_plot=False, qq_plot=False)
        assert result["x"]["ks_pvalue"] > 0.05

    @pytest.mark.parametrize(
        "dist_plot,qq_plot", [(True, True), (True, False), (False, True)]
    )
    def test_runs_with_plots(self, dist_plot, qq_plot):
        df = _make_df(cols=1)
        result = check_normality(df, dist_plot=dist_plot, qq_plot=qq_plot)
        assert isinstance(result, dict)


class TestEfaItemCorrMatrix:
    def test_returns_styler(self):
        import pandas.io.formats.style

        df = _make_df()
        result = efa_item_corr_matrix(df, list(df.columns))
        assert isinstance(result, pandas.io.formats.style.Styler)

    @pytest.mark.parametrize(
        "cols,expected_shape",
        [
            (["var0", "var1"], (2, 2)),
            (["var0", "var1", "var2"], (3, 3)),
        ],
    )
    def test_shape_matches_cols(self, cols, expected_shape):
        df = _make_df()
        result = efa_item_corr_matrix(df, cols)
        assert result.data.shape == expected_shape

    def test_title_sets_caption(self):
        df = _make_df()
        result = efa_item_corr_matrix(df, list(df.columns), title="My Matrix")
        assert result.caption == "My Matrix"


class TestCorrHeatmap:
    def test_runs_without_error(self):
        df = _make_df()
        corr_heatmap(df, title="Test Correlation Matrix")


class TestScatterWithRegression:
    def test_returns_float(self):
        df = _make_df()
        result = scatter_with_regression(df, "var0", "var1")
        assert isinstance(result, float)

    def test_correlation_in_valid_range(self):
        df = _make_df()
        result = scatter_with_regression(df, "var0", "var1")
        assert -1.0 <= result <= 1.0

    def test_known_correlation(self):
        """Perfectly correlated columns should return r=1.0."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=100)
        df = pd.DataFrame({"x": x, "y": x})
        result = scatter_with_regression(df, "x", "y")
        assert abs(result - 1.0) < 1e-10

    def test_matches_pandas_corr(self):
        df = _make_df(seed=7)
        result = scatter_with_regression(df, "var0", "var2")
        expected = df["var0"].corr(df["var2"])
        assert abs(result - expected) < 1e-10
