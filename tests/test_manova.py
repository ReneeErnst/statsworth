import numpy as np
import pandas as pd
import pytest

from stats_core.anova.manova import one_way_manova, one_way_manova_games_howell


def _make_significant_manova_data(seed: int = 0) -> pd.DataFrame:
    """Three groups with clearly different multivariate means."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
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


def _make_nonsignificant_manova_data(seed: int = 0) -> pd.DataFrame:
    """Three groups with identical multivariate means."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "group": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
            "dv1": rng.normal(15, 1, 180),
            "dv2": rng.normal(10, 1, 180),
        }
    )


class TestOneWayManova:
    def test_significant_returns_follow_up_dicts(self):
        df = _make_significant_manova_data()
        manova, anova_tables, tukey = one_way_manova(df, "group", ["dv1", "dv2"])
        assert anova_tables is not None
        assert tukey is not None
        assert set(anova_tables.keys()) == {"dv1", "dv2"}
        assert set(tukey.keys()) == {"dv1", "dv2"}

    def test_nonsignificant_returns_none_follow_ups(self):
        df = _make_nonsignificant_manova_data()
        manova, anova_tables, tukey = one_way_manova(df, "group", ["dv1", "dv2"])
        assert anova_tables is None
        assert tukey is None

    def test_raises_on_insufficient_group_size(self):
        # 2 DVs, groups of size 2 — min_group_size (2) must be > n_dvs (2)
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "dv1": [1.0, 2.0, 3.0, 4.0],
                "dv2": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="MANOVA not suitable"):
            one_way_manova(df, "group", ["dv1", "dv2"])

    def test_manova_result_has_expected_structure(self):
        df = _make_significant_manova_data()
        manova, _, _ = one_way_manova(df, "group", ["dv1", "dv2"])
        assert "C(group)" in manova.results


class TestOneWayManovaGamesHowell:
    def test_significant_returns_follow_up_dicts(self):
        df = _make_significant_manova_data()
        manova, welch, gh = one_way_manova_games_howell(df, "group", ["dv1", "dv2"])
        assert welch is not None
        assert gh is not None
        assert set(welch.keys()) == {"dv1", "dv2"}
        assert set(gh.keys()) == {"dv1", "dv2"}

    def test_nonsignificant_returns_none_follow_ups(self):
        df = _make_nonsignificant_manova_data()
        manova, welch, gh = one_way_manova_games_howell(df, "group", ["dv1", "dv2"])
        assert welch is None
        assert gh is None

    def test_raises_on_insufficient_group_size(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "dv1": [1.0, 2.0, 3.0, 4.0],
                "dv2": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="MANOVA not suitable"):
            one_way_manova_games_howell(df, "group", ["dv1", "dv2"])

    @pytest.mark.parametrize("dv", ["dv1", "dv2"])
    def test_significant_dv_has_gh_results(self, dv):
        df = _make_significant_manova_data()
        _, _, gh = one_way_manova_games_howell(df, "group", ["dv1", "dv2"])
        # Both DVs are strongly significant, so both should have GH results
        assert gh[dv] is not None

    def test_custom_alpha(self):
        df = _make_significant_manova_data()
        # Should still be significant at stricter alpha
        manova, welch, gh = one_way_manova_games_howell(
            df, "group", ["dv1", "dv2"], alpha=0.001
        )
        assert welch is not None
