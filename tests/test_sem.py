import numpy as np
import pandas as pd
import pytest
import semopy
from scipy.stats import chi2 as chi2_dist
from scipy.stats import ncx2

from statsworth.sem import fit_indices, rmsea_95ci

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_model(n: int = 300, seed: int = 42) -> semopy.Model:
    """Fit a simple one-factor CFA on simulated data."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n)
    noise = rng.standard_normal((n, 4)) * 0.5
    loadings = np.array([1.0, 0.9, 0.8, 0.7])
    data = pd.DataFrame(
        factor[:, None] * loadings + noise,
        columns=["x1", "x2", "x3", "x4"],
    )
    model = semopy.Model("f =~ x1 + x2 + x3 + x4")
    model.fit(data)
    return model


def _expected_95ci(chi2_obs: float, dof: float, n: int):
    """Reference implementation using scipy directly."""
    from scipy.optimize import brentq

    scale = dof * (n - 1)

    if chi2_dist.cdf(chi2_obs, dof) < 0.975:
        lower = 0.0
    else:
        ncp_lower = brentq(lambda ncp: ncx2.cdf(chi2_obs, dof, ncp) - 0.975, 0, chi2_obs * 4)
        lower = np.sqrt(ncp_lower / scale)

    ncp_upper = brentq(lambda ncp: ncx2.cdf(chi2_obs, dof, ncp) - 0.025, 0, chi2_obs * 4)
    upper = np.sqrt(ncp_upper / scale)

    return lower, upper


# ---------------------------------------------------------------------------
# rmsea_95ci
# ---------------------------------------------------------------------------


def test_rmsea_95ci_returns_tuple_of_three_floats():
    result = rmsea_95ci(_fit_model())
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_rmsea_95ci_lower_le_point_le_upper():
    point, lower, upper = rmsea_95ci(_fit_model())
    assert lower <= point <= upper


def test_rmsea_95ci_point_matches_semopy():
    model = _fit_model()
    expected = float(semopy.calc_stats(model)["RMSEA"].iloc[0])
    point, _, _ = rmsea_95ci(model)
    assert point == pytest.approx(expected, rel=1e-6)


def test_rmsea_95ci_bounds_are_non_negative():
    _, lower, upper = rmsea_95ci(_fit_model())
    assert lower >= 0.0
    assert upper >= 0.0


def test_rmsea_95ci_lower_lt_upper():
    _, lower, upper = rmsea_95ci(_fit_model())
    assert lower < upper


def test_rmsea_95ci_matches_reference_implementation():
    model = _fit_model()
    fit = semopy.calc_stats(model)
    chi2_obs = float(fit["chi2"].iloc[0])
    dof = float(fit["DoF"].iloc[0])
    n = model.n_samples

    expected_lower, expected_upper = _expected_95ci(chi2_obs, dof, n)
    _, lower, upper = rmsea_95ci(model)

    assert lower == pytest.approx(expected_lower, rel=1e-6)
    assert upper == pytest.approx(expected_upper, rel=1e-6)


def test_rmsea_95ci_lower_is_zero_when_chi2_le_dof():
    """When chi2 <= dof, RMSEA point estimate is 0 and lower CI bound must be 0."""
    rng = np.random.default_rng(0)
    n = 2000
    factor = rng.standard_normal(n)
    noise = rng.standard_normal((n, 4)) * 0.1
    data = pd.DataFrame(
        factor[:, None] * np.array([1.0, 1.0, 1.0, 1.0]) + noise,
        columns=["x1", "x2", "x3", "x4"],
    )
    model = semopy.Model("f =~ x1 + x2 + x3 + x4")
    model.fit(data)

    fit = semopy.calc_stats(model)
    chi2_obs = float(fit["chi2"].iloc[0])
    dof = float(fit["DoF"].iloc[0])

    if chi2_obs <= dof:
        _, lower, _ = rmsea_95ci(model)
        assert lower == 0.0


def test_rmsea_95ci_width_decreases_with_larger_n():
    """Larger samples should produce narrower CIs."""
    _, lo_small, hi_small = rmsea_95ci(_fit_model(n=100, seed=1))
    _, lo_large, hi_large = rmsea_95ci(_fit_model(n=2000, seed=1))
    assert (hi_large - lo_large) < (hi_small - lo_small)


# ---------------------------------------------------------------------------
# fit_indices
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "chi2",
    "dof",
    "p_value",
    "cfi",
    "tli",
    "gfi",
    "agfi",
    "nfi",
    "rmsea",
    "rmsea_lower",
    "rmsea_upper",
    "aic",
    "bic",
    "log_lik",
}


def test_fit_indices_returns_dict_with_all_keys():
    assert set(fit_indices(_fit_model()).keys()) == _EXPECTED_KEYS


def test_fit_indices_all_values_are_floats():
    assert all(isinstance(v, float) for v in fit_indices(_fit_model()).values())


def test_fit_indices_rmsea_bounds_bracket_point():
    stats = fit_indices(_fit_model())
    assert stats["rmsea_lower"] <= stats["rmsea"] <= stats["rmsea_upper"]


def test_fit_indices_rmsea_consistent_with_rmsea_95ci():
    model = _fit_model()
    stats = fit_indices(model)
    point, lower, upper = rmsea_95ci(model)
    assert stats["rmsea"] == pytest.approx(point)
    assert stats["rmsea_lower"] == pytest.approx(lower)
    assert stats["rmsea_upper"] == pytest.approx(upper)


def test_fit_indices_dof_is_positive():
    assert fit_indices(_fit_model())["dof"] > 0


def test_fit_indices_p_value_in_unit_interval():
    assert 0.0 <= fit_indices(_fit_model())["p_value"] <= 1.0


@pytest.mark.parametrize("key", ["cfi", "tli", "gfi", "agfi", "nfi"])
def test_fit_indices_in_unit_interval(key):
    assert 0.0 <= fit_indices(_fit_model())[key] <= 1.0
