import numpy as np
import pandas as pd

from stats_core.anova.one_way import (
    games_howell,
    one_way_anova,
    welch_anova_and_games_howell,
)


def _make_significant_groups(seed: int = 0) -> pd.DataFrame:
    """Three groups with clearly different means."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
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


def _make_nonsignificant_groups(seed: int = 0) -> pd.DataFrame:
    """Three groups with identical means."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "group": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "score": rng.normal(15, 1, 150),
        }
    )


def _make_unequal_variance_groups(seed: int = 0) -> pd.DataFrame:
    """Three groups with different means and very different variances."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "group": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
            "score": np.concatenate(
                [
                    rng.normal(10, 0.5, 60),
                    rng.normal(20, 5.0, 60),
                    rng.normal(30, 0.5, 60),
                ]
            ),
        }
    )


class TestOneWayAnova:
    def test_significant_result(self):
        df = _make_significant_groups()
        anova_table, tukey = one_way_anova(df, "group", "score")
        p_value = anova_table["PR(>F)"].dropna().iloc[0]
        assert p_value < 0.05

    def test_nonsignificant_result(self):
        df = _make_nonsignificant_groups()
        anova_table, tukey = one_way_anova(df, "group", "score")
        p_value = anova_table["PR(>F)"].dropna().iloc[0]
        assert p_value > 0.05

    def test_returns_anova_table_and_tukey(self):
        df = _make_significant_groups()
        anova_table, tukey = one_way_anova(df, "group", "score")
        assert "PR(>F)" in anova_table.columns
        # TukeyHSDResults has a summary method
        assert hasattr(tukey, "summary")

    def test_tukey_contains_all_pairs(self):
        df = _make_significant_groups()
        _, tukey = one_way_anova(df, "group", "score")
        summary = tukey.summary().data[1:]  # skip header row
        pair_strings = {(row[0], row[1]) for row in summary}
        assert len(pair_strings) == 3  # A-B, A-C, B-C


class TestWelchAnovaAndGamesHowell:
    def test_significant_returns_gh_results(self):
        df = _make_unequal_variance_groups()
        welch, gh = welch_anova_and_games_howell(df, "group", "score")
        assert welch.pvalue < 0.05
        assert gh is not None
        assert isinstance(gh, type(pd.DataFrame()))

    def test_nonsignificant_returns_none_for_gh(self):
        df = _make_nonsignificant_groups()
        welch, gh = welch_anova_and_games_howell(df, "group", "score")
        assert welch.pvalue > 0.05
        assert gh is None

    def test_welch_result_has_pvalue(self):
        df = _make_significant_groups()
        welch, _ = welch_anova_and_games_howell(df, "group", "score")
        assert hasattr(welch, "pvalue")
        assert 0.0 <= welch.pvalue <= 1.0


class TestGamesHowell:
    def test_returns_dataframe(self):
        df = _make_significant_groups()
        result = games_howell(df, "group", "score")
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        df = _make_significant_groups()
        result = games_howell(df, "group", "score")
        expected = ["Group1", "Group2", "Mean Diff", "t", "df", "p-value"]
        assert list(result.columns) == expected

    def test_all_pairs_present(self):
        df = _make_significant_groups()
        result = games_howell(df, "group", "score")
        assert len(result) == 3  # C(3, 2) = 3 pairs

    def test_p_values_in_valid_range(self):
        df = _make_significant_groups()
        result = games_howell(df, "group", "score")
        assert result["p-value"].between(0, 1).all()

    def test_significant_differences_detected(self):
        df = _make_significant_groups()
        result = games_howell(df, "group", "score")
        assert (result["p-value"] < 0.05).all()
