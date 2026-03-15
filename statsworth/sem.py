"""Structural equation model fit indices."""

import numpy as np
import semopy
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2_dist
from scipy.stats import ncx2


def rmsea_95ci(model: semopy.Model) -> tuple[float, float, float]:
    """Compute the RMSEA point estimate and 95% confidence interval.

    Uses the noncentral chi-square method (MacCallum et al., 1996): finds the
    noncentrality parameters whose distributions place the observed chi-square
    at their 2.5th and 97.5th percentiles respectively.

    Args:
        model: A fitted ``semopy.Model`` instance.

    Returns:
        Tuple ``(point, lower, upper)`` where ``point`` is the RMSEA point
        estimate and ``lower``/``upper`` are the 95% CI bounds.
    """
    fit = semopy.calc_stats(model)
    chi2_obs = float(fit["chi2"].iloc[0])
    dof = float(fit["DoF"].iloc[0])
    n = model.n_samples
    scale = dof * (n - 1)

    point = float(fit["RMSEA"].iloc[0])

    if chi2_dist.cdf(chi2_obs, dof) < 0.975:
        lower = 0.0
    else:
        ncp_lower = brentq(lambda ncp: ncx2.cdf(chi2_obs, dof, ncp) - 0.975, 0, chi2_obs * 4)
        lower = np.sqrt(ncp_lower / scale)

    if chi2_dist.cdf(chi2_obs, dof) < 0.025:
        upper = 0.0
    else:
        # Expand the upper bracket until the CDF drops below 0.025
        upper_b = max(chi2_obs * 4, 1.0)
        while ncx2.cdf(chi2_obs, dof, upper_b) > 0.025:
            upper_b *= 2
        ncp_upper = brentq(lambda ncp: ncx2.cdf(chi2_obs, dof, ncp) - 0.025, 0, upper_b)
        upper = np.sqrt(ncp_upper / scale)

    return point, lower, upper
