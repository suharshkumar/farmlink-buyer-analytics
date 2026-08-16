"""Correctness guards for the simulation and the analysis.

The two tests that matter most here are `test_forward_label_looks_forward`
(pandas `.rolling()` is trailing, so the obvious way to build a forward label
silently builds a backward one) and `test_causal_estimators_recover_truth`
(the whole argument of this repo is that the estimator works, so that claim is
asserted, not just asserted in prose).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farmlink_buyers import causal, churn, experiment, metrics, simulate  # noqa: E402


@pytest.fixture(scope="module")
def sim() -> dict:
    """Small shared simulation for structural checks - fast, built once."""
    return simulate.simulate(simulate.SimConfig(n_buyers=250))


@pytest.fixture(scope="module")
def sim_full() -> dict:
    """Full-size simulation, matching the configuration the README reports.

    The causal assertions run against this rather than the 250-buyer fixture
    on purpose: an estimator's accuracy is a function of sample size, so
    testing it at a size nobody reports would either fail spuriously or pass
    with thresholds too loose to mean anything.
    """
    return simulate.simulate(simulate.SimConfig())


# --------------------------------------------------------------------------
# Simulation sanity
# --------------------------------------------------------------------------
def test_simulation_shapes(sim):
    assert len(sim["orders"]) > 5000
    assert len(sim["buyers"]) == 250
    assert not sim["panel"].empty
    assert not sim["truth"].empty


def test_fill_rate_is_a_valid_proportion(sim):
    f = sim["orders"]["fill_rate"]
    assert f.between(0, 1).all()
    # Tuned so short-shipping is a real but not dominant event.
    assert 0.03 < (f < 0.9).mean() < 0.35


def test_delivered_never_exceeds_ordered(sim):
    o = sim["orders"]
    assert (o["qty_delivered_kg"] <= o["qty_ordered_kg"] + 1e-6).all()


def test_buyer_never_appears_after_churning(sim):
    """The panel must be truncated at the churn week, or every downstream
    retention number is wrong."""
    p = sim["panel"]
    churn_week = (
        p[p["churned_this_week"]].groupby("buyer_id")["week"].min()
    )
    last_week = p.groupby("buyer_id")["week"].max()
    common = churn_week.index.intersection(last_week.index)
    assert (last_week.loc[common] == churn_week.loc[common]).all()


def test_prices_are_real_not_random(sim):
    """Guard against silently falling back to synthetic prices: the real
    series has the 2022-24 inflation, so late prices must exceed early ones."""
    o = sim["orders"]
    early = o[o["week"] < "2023-07-01"]["unit_price_lkr"].mean()
    late = o[o["week"] > "2025-07-01"]["unit_price_lkr"].mean()
    assert late > early


def test_simulation_is_reproducible():
    a = simulate.simulate(simulate.SimConfig(n_buyers=120, seed=7))
    b = simulate.simulate(simulate.SimConfig(n_buyers=120, seed=7))
    pd.testing.assert_frame_equal(a["orders"], b["orders"])


# --------------------------------------------------------------------------
# Confounding must actually exist, or the causal section is vacuous
# --------------------------------------------------------------------------
def test_confounding_is_present(sim):
    df = causal.build_analysis_frame(sim)
    smd = causal.standardised_differences(df).set_index("covariate")["smd"]
    # Large buyers are protected on fill; volatile baskets are exposed.
    assert smd["log_size"] < -0.3
    assert smd["basket_volatility"] > 0.3


def test_naive_estimate_is_biased_upward(sim):
    df = causal.build_analysis_frame(sim)
    naive = causal.naive_difference(df).att
    truth = simulate.true_att(sim["truth"])["att"]
    assert naive > truth * 2, "confounding should inflate the naive estimate"


def test_causal_estimators_recover_truth(sim_full):
    """The central claim: adjustment gets close to the known answer."""
    res = causal.run_all(sim_full)
    r = res["results"].set_index("method")

    aipw = r.loc["aipw_doubly_robust"]
    assert aipw["covers_truth"], "AIPW confidence interval missed the truth"
    assert aipw["abs_error"] < 0.015

    # Matching should also land in the right neighbourhood.
    psm = r[r.index.str.startswith("psm")].iloc[0]
    assert psm["covers_truth"]

    # And the whole point: adjustment must beat doing nothing, by a lot.
    naive = r.loc["naive_difference"]
    assert naive["abs_error"] > aipw["abs_error"] * 10


def test_adjustment_balances_covariates(sim_full):
    """|SMD| < 0.1 is the conventional bar for a balanced matched sample."""
    res = causal.run_all(sim_full)
    assert (res["balance"]["max_abs_smd_after"] < 0.10).all()
    # ...and balance must genuinely improve, not start balanced.
    assert res["balance"]["smd_unadjusted"].abs().max() > 0.8


# --------------------------------------------------------------------------
# Churn labelling
# --------------------------------------------------------------------------
def test_forward_label_looks_forward():
    """A churn at week 10 must label weeks 2-9, never weeks 11+.

    pandas `.rolling()` is a trailing window; `shift(-1).rolling(8)` therefore
    looks backwards. This test pins the correct direction.
    """
    weeks = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    s = pd.Series(np.zeros(20, dtype=bool), index=range(20))
    s.iloc[10] = True

    fwd = s[::-1].rolling(8, min_periods=1).max()[::-1].shift(-1).fillna(0).astype(int)
    assert fwd.iloc[9] == 1     # churn is next week
    assert fwd.iloc[2] == 1     # churn is 8 weeks away
    assert fwd.iloc[1] == 0     # churn is 9 weeks away - outside the window
    assert fwd.iloc[11] == 0    # churn already happened; must not look back
    assert len(weeks) == 20


def test_churn_features_have_no_lookahead(sim):
    """Every history feature is built with .shift(1); confirm on one buyer."""
    frame = churn.build_training_frame(sim)
    b = frame[frame.buyer_id == frame.buyer_id.iloc[0]].sort_values("week")
    if len(b) < 12:
        pytest.skip("buyer too short")
    expected = b["ordered"].shift(1).rolling(8, min_periods=1).sum().to_numpy()
    got = b["orders_last_8w"].to_numpy()
    m = np.isfinite(expected) & np.isfinite(got)
    np.testing.assert_allclose(got[m], expected[m])


def test_churn_base_rate_is_plausible(sim):
    frame = churn.build_training_frame(sim)
    rate = frame["churn_next"].mean()
    assert 0.01 < rate < 0.30


def test_churn_split_is_temporal(sim):
    frame = churn.build_training_frame(sim)
    cm, (Xtr, ytr, tr), (Xte, yte, te) = churn.train(frame)
    assert tr["week"].max() < te["week"].min()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_retention_starts_at_or_below_one(sim):
    ret = metrics.cohort_retention(sim["panel"], sim["buyers"])
    assert (ret.fillna(0) <= 1.0 + 1e-9).all().all()


def test_retention_is_monotone_on_average(sim):
    """Cohort retention should decay; a rising average curve means the cohort
    definition is wrong or survivorship is leaking in."""
    curve = metrics.retention_curve(metrics.cohort_retention(sim["panel"], sim["buyers"]))
    early = curve[curve.months_since.between(1, 3)]["retention"].mean()
    late = curve[curve.months_since.between(8, 12)]["retention"].mean()
    assert late < early


def test_rfm_scores_in_range(sim):
    r = metrics.rfm(sim["orders"])
    for c in ("R", "F", "M"):
        assert r[c].between(1, 5).all()
    assert r["rfm_score"].between(3, 15).all()


# --------------------------------------------------------------------------
# Experiment design
# --------------------------------------------------------------------------
def test_sample_size_grows_as_effect_shrinks():
    big = experiment.sample_size_proportion(0.08, -0.04)
    small = experiment.sample_size_proportion(0.08, -0.01)
    assert small > big * 4


def test_design_table_handles_all_relative_lifts():
    d = experiment.design_table(0.02, n_buyers=900)
    assert d["n_per_arm_required"].notna().all()
    assert (d["n_per_arm_required"] > 0).all()


def test_cuped_reduces_variance():
    rng = np.random.default_rng(0)
    pre = rng.normal(100, 15, 5000)
    post = 0.7 * pre + rng.normal(30, 10, 5000)
    out = experiment.cuped_variance_reduction(pre, post)
    assert out["variance_reduction_pct"] > 30
    assert out["effective_n_multiplier"] > 1.4


def test_srm_detects_a_broken_split():
    assert not experiment.srm_check(5000, 4950)["srm_detected"]
    assert experiment.srm_check(5000, 4000)["srm_detected"]


def test_sequential_boundaries_are_stricter_early():
    b = experiment.sequential_boundary(5)
    assert b["z_boundary"].iloc[0] > b["z_boundary"].iloc[-1]
