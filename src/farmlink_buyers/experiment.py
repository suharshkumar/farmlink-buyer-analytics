"""Experiment design for the intervention the analysis implies.

The causal work says short-shipping drives churn. The obvious product response
is a supply-commitment feature: hold buffer stock against confirmed orders for
high-value accounts so their fill rate stops depending on the spot market.

Before building it, the question is whether the effect is large enough to
detect at FarmLink's size. This module answers that: minimum detectable effect
at the current sample, required runtime, and the variance reduction available
from CUPED - which matters a great deal here, because pre-period order volume
is highly predictive of post-period behaviour and cuts the required sample
roughly in half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mde_proportion(baseline: float, n_per_arm: int, alpha: float = 0.05,
                   power: float = 0.80) -> float:
    """Minimum detectable absolute lift on a binary metric."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    se = np.sqrt(2 * baseline * (1 - baseline) / n_per_arm)
    return float((z_a + z_b) * se)


def sample_size_proportion(baseline: float, lift: float, alpha: float = 0.05,
                           power: float = 0.80) -> int:
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p1, p2 = baseline, baseline + lift
    pbar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (lift ** 2)))


def design_table(baseline_churn: float, n_buyers: int,
                 relative_lifts=(0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """What can we detect, and how long would it take?

    Lifts are expressed as *relative* reductions in churn, which is how a
    product team states a goal ("cut churn by 20%"). Absolute lifts are derived
    from the baseline, so the table can never ask for a churn rate below zero.
    """
    rows = []
    for rel in relative_lifts:
        lift = baseline_churn * rel
        n = sample_size_proportion(baseline_churn, -lift)
        rows.append({
            "absolute_lift": round(lift, 5),
            "relative_lift_pct": round(100 * rel, 1),
            "n_per_arm_required": n,
            "n_total_required": 2 * n,
            "feasible_at_current_size": bool(2 * n <= n_buyers),
            # Buyers accumulate exposures weekly, so a study needing more
            # buyer-weeks than we have buyers simply runs for more weeks.
            "weeks_to_power_at_current_size": round(2 * n / max(1, n_buyers), 1),
        })
    out = pd.DataFrame(rows)
    out["mde_at_current_size"] = mde_proportion(baseline_churn, n_buyers // 2)
    return out


def cuped_variance_reduction(pre: np.ndarray, post: np.ndarray) -> dict:
    """CUPED: regress the outcome on a pre-period covariate to strip variance.

    theta is chosen as cov(pre, post)/var(pre); the adjusted metric has the
    same expectation but variance scaled by (1 - rho^2). Reported as an
    effective sample-size multiplier, which is what a launch decision cares
    about.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    m = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[m], post[m]
    if len(pre) < 30 or pre.var() == 0:
        return {"rho": 0.0, "variance_reduction_pct": 0.0, "effective_n_multiplier": 1.0}

    theta = np.cov(pre, post, ddof=1)[0, 1] / pre.var(ddof=1)
    adj = post - theta * (pre - pre.mean())
    rho = float(np.corrcoef(pre, post)[0, 1])
    red = 1 - adj.var(ddof=1) / post.var(ddof=1)
    return {
        "theta": float(theta),
        "rho": rho,
        "variance_reduction_pct": float(100 * red),
        "effective_n_multiplier": float(1 / max(1e-9, 1 - red)),
    }


def sequential_boundary(n_looks: int, alpha: float = 0.05) -> pd.DataFrame:
    """O'Brien-Fleming spending boundaries for interim analyses.

    Included because the temptation to peek at a running experiment is
    universal, and peeking at a fixed 0.05 inflates the false-positive rate to
    roughly 0.14 across five looks. These boundaries keep it at 0.05.
    """
    looks = np.arange(1, n_looks + 1)
    frac = looks / n_looks
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_bound = z_alpha / np.sqrt(frac)
    return pd.DataFrame({
        "look": looks,
        "information_fraction": frac,
        "z_boundary": z_bound,
        "nominal_p_boundary": 2 * (1 - stats.norm.cdf(z_bound)),
    })


def srm_check(n_a: int, n_b: int, expected_ratio: float = 0.5) -> dict:
    """Sample-ratio mismatch test - the first thing to check before reading
    any experiment result. A failing SRM invalidates the whole readout."""
    n = n_a + n_b
    expected = np.array([n * expected_ratio, n * (1 - expected_ratio)])
    chi2 = ((np.array([n_a, n_b]) - expected) ** 2 / expected).sum()
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return {"n_a": n_a, "n_b": n_b, "chi2": float(chi2), "p_value": p,
            "srm_detected": bool(p < 0.001)}
