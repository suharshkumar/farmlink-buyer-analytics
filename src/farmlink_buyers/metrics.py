"""Product and commercial metrics: cohorts, retention, RFM, LTV.

These are the numbers a weekly business review runs on. The emphasis
throughout is on *cohort* views rather than aggregate ones, because a
marketplace that is growing fast will show a flattering blended retention
number even while every individual cohort is getting worse - new buyers keep
refilling the top of the funnel and hide the leak underneath.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Cohort retention
# --------------------------------------------------------------------------
def cohort_retention(panel: pd.DataFrame, buyers: pd.DataFrame,
                     max_months: int = 24) -> pd.DataFrame:
    """Share of each signup cohort still ordering, by months since signup.

    "Retained" means placed at least one order in that month, which is the
    honest definition for a business with a weekly-to-monthly purchase cycle.
    A last-order-based definition would count a dormant account as retained
    until the day it is written off.
    """
    p = panel.merge(buyers[["buyer_id", "signup_week", "cohort", "segment"]],
                    on="buyer_id", how="left")
    p["month"] = p["week"].dt.to_period("M")
    p["signup_month"] = p["signup_week"].dt.to_period("M")
    p["months_since"] = (p["month"] - p["signup_month"]).apply(lambda x: x.n)
    p = p[(p["months_since"] >= 0) & (p["months_since"] <= max_months)]

    monthly = (
        p.groupby(["cohort", "buyer_id", "months_since"], as_index=False)["ordered"]
        .max()
    )
    size = monthly[monthly["months_since"] == 0].groupby("cohort")["buyer_id"].nunique()
    active = (
        monthly[monthly["ordered"]]
        .groupby(["cohort", "months_since"])["buyer_id"].nunique()
        .unstack(fill_value=0)
    )
    retention = active.div(size, axis=0)
    retention.index.name = "cohort"
    return retention


def retention_curve(retention: pd.DataFrame, min_cohort_size: int = 10) -> pd.DataFrame:
    """Average retention curve across cohorts, with an observed-cohort count.

    Later months are averaged over fewer cohorts (only the oldest ones have
    lived that long), so `n_cohorts` is reported alongside - a retention curve
    that ticks up at month 20 is nearly always survivorship, not loyalty.
    """
    out = pd.DataFrame({
        "months_since": retention.columns.astype(int),
        "retention": retention.mean(axis=0, skipna=True).to_numpy(),
        "n_cohorts": retention.notna().sum(axis=0).to_numpy(),
    })
    return out[out["n_cohorts"] >= 1].reset_index(drop=True)


# --------------------------------------------------------------------------
# RFM
# --------------------------------------------------------------------------
def rfm(orders: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency / Frequency / Monetary table with 1-5 quintile scores."""
    asof = asof or orders["week"].max()
    g = orders.groupby("buyer_id").agg(
        last_order=("week", "max"),
        frequency=("week", "nunique"),
        monetary=("line_gmv_lkr", "sum"),
        revenue=("revenue_lkr", "sum"),
        avg_fill=("fill_rate", "mean"),
        n_lines=("item", "size"),
    ).reset_index()
    g["recency_weeks"] = (asof - g["last_order"]).dt.days / 7.0

    # Recency is scored in reverse: recent = good. Ranked before binning
    # because many buyers share a last-order week, and tied edges would
    # otherwise collapse the quintiles.
    g["R"] = pd.qcut(
        g["recency_weeks"].rank(method="first", ascending=False), 5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    g["F"] = pd.qcut(g["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["M"] = pd.qcut(g["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["rfm_score"] = g["R"] + g["F"] + g["M"]

    def label(r):
        if r.R >= 4 and r.F >= 4 and r.M >= 4:
            return "champion"
        if r.R >= 4 and r.F >= 3:
            return "loyal"
        if r.R >= 4:
            return "promising"
        if r.R <= 2 and r.F >= 4:
            return "at_risk_high_value"
        if r.R <= 2 and r.M >= 4:
            return "cant_lose"
        if r.R <= 2:
            return "hibernating"
        return "needs_attention"

    g["rfm_segment"] = g.apply(label, axis=1)
    return g


def segment_personas(rfm_table: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """K-means personas over standardised RFM, as a data-driven cross-check on
    the rule-based segments above."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feats = ["recency_weeks", "frequency", "monetary", "avg_fill"]
    X = StandardScaler().fit_transform(np.log1p(rfm_table[feats].clip(lower=0)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    out = rfm_table.copy()
    out["persona"] = km.labels_
    profile = out.groupby("persona").agg(
        n=("buyer_id", "size"),
        recency_weeks=("recency_weeks", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
        avg_fill=("avg_fill", "mean"),
        revenue=("revenue", "sum"),
    ).reset_index()
    profile["revenue_share_pct"] = 100 * profile["revenue"] / profile["revenue"].sum()
    return profile.sort_values("monetary", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------
# Lifetime value
# --------------------------------------------------------------------------
def cohort_ltv(orders: pd.DataFrame, buyers: pd.DataFrame,
               max_months: int = 24) -> pd.DataFrame:
    """Cumulative net revenue per acquired buyer, by cohort and month age."""
    o = orders.merge(buyers[["buyer_id", "signup_week", "cohort"]], on="buyer_id", how="left")
    o["month"] = o["week"].dt.to_period("M")
    o["signup_month"] = o["signup_week"].dt.to_period("M")
    o["months_since"] = (o["month"] - o["signup_month"]).apply(lambda x: x.n)
    o = o[(o["months_since"] >= 0) & (o["months_since"] <= max_months)]

    rev = o.groupby(["cohort", "months_since"], as_index=False)["revenue_lkr"].sum()
    size = buyers.groupby("cohort")["buyer_id"].nunique().rename("cohort_size")
    rev = rev.merge(size, on="cohort", how="left")
    rev["rev_per_buyer"] = rev["revenue_lkr"] / rev["cohort_size"]
    rev["cum_ltv"] = rev.groupby("cohort")["rev_per_buyer"].cumsum()
    return rev


def ltv_summary(ltv: pd.DataFrame, horizons=(3, 6, 12)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        sub = ltv[ltv["months_since"] == h]
        if sub.empty:
            continue
        rows.append({"months": h,
                     "mean_cum_ltv_lkr": float(sub["cum_ltv"].mean()),
                     "median_cum_ltv_lkr": float(sub["cum_ltv"].median()),
                     "n_cohorts": int(sub["cohort"].nunique())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fill rate vs retention (descriptive - the causal version is in causal.py)
# --------------------------------------------------------------------------
def fill_rate_vs_retention(panel: pd.DataFrame, bins=(0, 0.7, 0.8, 0.9, 0.95, 1.01)) -> pd.DataFrame:
    """Churn rate by fill-rate band.

    Descriptive only. The gradient here is real but heavily confounded, which
    is precisely the motivation for `causal.py` - it is reported alongside the
    adjusted estimate so the size of the naive bias is visible.
    """
    p = panel.dropna(subset=["fill_rate"]).copy()
    p["fill_band"] = pd.cut(p["fill_rate"], bins=list(bins), right=False)
    g = p.groupby("fill_band", observed=True).agg(
        n_orders=("fill_rate", "size"),
        churn_rate=("churned_this_week", "mean"),
        mean_gmv=("gmv_lkr", "mean"),
    ).reset_index()
    g["churn_rate_pct"] = 100 * g["churn_rate"]
    return g


def north_star_table(orders: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Weekly metric tree: the handful of numbers a review actually needs."""
    w = orders.groupby("week", as_index=False).agg(
        gmv_lkr=("line_gmv_lkr", "sum"),
        revenue_lkr=("revenue_lkr", "sum"),
        orders=("buyer_id", "size"),
        active_buyers=("buyer_id", "nunique"),
        fill_rate=("fill_rate", "mean"),
    )
    w["aov_lkr"] = w["gmv_lkr"] / w["active_buyers"]
    churn = panel.groupby("week", as_index=False).agg(
        churned=("churned_this_week", "sum"),
        alive=("buyer_id", "nunique"),
    )
    w = w.merge(churn, on="week", how="left")
    w["weekly_churn_pct"] = 100 * w["churned"] / w["alive"]
    w["short_ship_pct"] = 100 * (
        orders.assign(s=orders.fill_rate < 0.9).groupby("week")["s"].mean().to_numpy()
    )
    return w
