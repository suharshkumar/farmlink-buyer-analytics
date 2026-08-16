"""Causal effect of a short-shipped order on buyer churn.

The question
-----------
"A buyer got 78% of what they ordered this week. How much more likely are they
to leave in the next two months?"

The naive answer - compare churn among short-shipped buyers to everyone else -
is badly wrong here, and wrong in a direction that matters. Short-shipping
happens during supply shortages, and shortages coincide with price spikes that
push buyers away on their own. Small buyers are de-prioritised on allocation
*and* churn more for unrelated commercial reasons. Both channels inflate the
raw gap.

What this module does
---------------------
Estimates the ATT five ways - naive, regression-adjusted, propensity-score
matched, IPW, and doubly-robust AIPW - and reports balance diagnostics for each.

Why it can be trusted
---------------------
Because the order book comes from a written-down data-generating process, the
true ATT is known. Every estimator below is scored against a number it never
sees. That converts "here is a causal estimate" into "here is a causal estimate
from a method demonstrated to recover the right answer under this confounding
structure" - which is the only version worth acting on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COVARIATES = [
    "log_size", "basket_volatility", "tenure_years", "weeks_inactive",
    "price_shock", "orders_last_8w", "gmv_last_8w_log", "prior_fill_mean",
    "prior_short_rate", "quality_pref",
]


def build_analysis_frame(sim: dict) -> pd.DataFrame:
    """Join the exposure table to pre-treatment covariates only.

    Every covariate is measured strictly *before* the treatment week. Including
    anything contemporaneous would condition on a collider and bias the
    estimate in an unpredictable direction.
    """
    truth, panel, buyers = sim["truth"], sim["panel"], sim["buyers"]

    p = panel.sort_values(["buyer_id", "week"]).copy()
    g = p.groupby("buyer_id", sort=False)

    # Trailing 8-week history, shifted so the current week never leaks in.
    p["orders_last_8w"] = (
        g["ordered"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).sum())
    )
    p["gmv_last_8w"] = (
        g["gmv_lkr"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).sum())
    )
    p["gmv_last_8w_log"] = np.log1p(p["gmv_last_8w"])
    p["prior_fill_mean"] = (
        g["fill_rate"].transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    )
    p["prior_short_rate"] = (
        g["short_shipped"].transform(
            lambda s: s.shift(1).rolling(12, min_periods=1).mean()
        )
    )

    keep = ["buyer_id", "week", "tenure_years", "weeks_inactive", "price_shock",
            "orders_last_8w", "gmv_last_8w_log", "prior_fill_mean", "prior_short_rate"]
    df = truth.merge(p[keep], on=["buyer_id", "week"], how="inner")
    df = df.merge(
        buyers[["buyer_id", "segment", "log_size", "basket_volatility", "quality_pref"]],
        on="buyer_id", how="left",
    )

    df["prior_fill_mean"] = df["prior_fill_mean"].fillna(df["prior_fill_mean"].mean())
    df["prior_short_rate"] = df["prior_short_rate"].fillna(0.0)
    return df.dropna(subset=COVARIATES).reset_index(drop=True)


def _design(df: pd.DataFrame) -> np.ndarray:
    X = df[COVARIATES].to_numpy(dtype=float)
    seg = pd.get_dummies(df["segment"], prefix="seg", drop_first=True).to_numpy(dtype=float)
    return np.hstack([X, seg])


def propensity_scores(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """P(short-shipped | pre-treatment covariates), from regularised logistic
    regression. Cross-fitted so the scores used for weighting are out-of-fold,
    which stops an over-fitted model from manufacturing spurious balance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, t = _design(df), df["treated"].to_numpy(int)
    ps = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, t):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        ).fit(X[tr], t[tr])
        ps[te] = m.predict_proba(X[te])[:, 1]
    return np.clip(ps, 0.01, 0.99)


def standardised_differences(df: pd.DataFrame, weights: np.ndarray | None = None
                             ) -> pd.DataFrame:
    """Standardised mean difference per covariate; |SMD| < 0.1 is the
    conventional bar for "balanced"."""
    t = df["treated"].to_numpy(bool)
    w = np.ones(len(df)) if weights is None else weights
    rows = []
    for c in COVARIATES:
        x = df[c].to_numpy(dtype=float)
        wt, wc = w[t], w[~t]
        mt = np.average(x[t], weights=wt)
        mc = np.average(x[~t], weights=wc)
        vt = np.average((x[t] - mt) ** 2, weights=wt)
        vc = np.average((x[~t] - mc) ** 2, weights=wc)
        pooled = np.sqrt((vt + vc) / 2)
        rows.append({"covariate": c, "mean_treated": mt, "mean_control": mc,
                     "smd": (mt - mc) / pooled if pooled > 0 else 0.0})
    return pd.DataFrame(rows)


@dataclass
class Estimate:
    method: str
    att: float
    se: float
    n_treated: int
    n_used: int
    note: str = ""

    @property
    def ci(self) -> tuple[float, float]:
        return (self.att - 1.96 * self.se, self.att + 1.96 * self.se)


def naive_difference(df: pd.DataFrame) -> Estimate:
    t = df["treated"].to_numpy(bool)
    y = df["y_obs"].to_numpy(float)
    d = y[t].mean() - y[~t].mean()
    se = np.sqrt(y[t].var(ddof=1) / t.sum() + y[~t].var(ddof=1) / (~t).sum())
    return Estimate("naive_difference", d, se, int(t.sum()), len(df),
                    "no adjustment - shown to quantify the bias")


def regression_adjustment(df: pd.DataFrame) -> Estimate:
    """Outcome regression (g-computation): fit E[Y | T, X], then contrast
    predicted outcomes with T forced to 1 vs 0 over the treated rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = _design(df)
    t = df["treated"].to_numpy(int)
    y = df["y_obs"].to_numpy(int)
    XT = np.hstack([X, t.reshape(-1, 1)])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(XT, y)

    treated = t == 1
    X1 = np.hstack([X[treated], np.ones((treated.sum(), 1))])
    X0 = np.hstack([X[treated], np.zeros((treated.sum(), 1))])
    diff = m.predict_proba(X1)[:, 1] - m.predict_proba(X0)[:, 1]
    return Estimate("regression_adjustment", float(diff.mean()),
                    float(diff.std(ddof=1) / np.sqrt(len(diff))),
                    int(treated.sum()), len(df))


def trim_common_support(df: pd.DataFrame, ps: np.ndarray, eps: float = 0.02
                        ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Restrict to the propensity range where both arms actually appear.

    Outside common support there is no counterfactual to borrow, and every
    weighting estimator will happily extrapolate into that void and return
    something confident and wrong. Trimming changes the estimand slightly - it
    becomes the ATT on the overlapping population - which is the honest thing
    to report, and the excluded share is stated so the reader can judge.
    """
    t = df["treated"].to_numpy(bool)
    lo = max(ps[t].min(), ps[~t].min()) + eps
    hi = min(ps[t].max(), ps[~t].max()) - eps
    keep = (ps >= lo) & (ps <= hi)
    info = {
        "ps_low": float(lo), "ps_high": float(hi),
        "kept": int(keep.sum()), "dropped": int((~keep).sum()),
        "dropped_pct": float(100 * (~keep).mean()),
        "treated_kept": int((keep & t).sum()),
    }
    return df[keep].reset_index(drop=True), ps[keep], info


def psm_match(df: pd.DataFrame, ps: np.ndarray, caliper_sd: float = 0.2,
              k: int = 5) -> tuple[Estimate, pd.DataFrame]:
    """1:k nearest-neighbour matching on the propensity score with a caliper.

    The caliper is set on the **logit** of the propensity score at
    `caliper_sd` standard deviations (Austin 2011). A caliper on the raw score
    is not scale-free: near ps=0.02, where most of this data sits, a 0.02-wide
    raw caliper spans a huge range of log-odds and matches badly, while near
    ps=0.5 it is far too strict. Working on the logit scale fixes both ends.

    Matching is with replacement: control rows are scarce in the
    high-propensity region (that is what confounding *is*), and forcing
    without-replacement matches there would pair treated rows to controls far
    away on the score, reintroducing exactly the bias the method exists to
    remove. Treated rows with no control inside the caliper are dropped and
    counted, because an unmatched treated row has no credible counterfactual.
    """
    t = df["treated"].to_numpy(bool)
    y = df["y_obs"].to_numpy(float)

    logit = np.log(ps / (1 - ps))
    caliper = caliper_sd * logit.std()

    ps_t, ps_c = logit[t], logit[~t]
    y_t, y_c = y[t], y[~t]

    order = np.argsort(ps_c)
    ps_c_sorted, y_c_sorted = ps_c[order], y_c[order]

    diffs, matched_treated, matched_ctrl_idx, unmatched = [], [], [], 0
    for i, p in enumerate(ps_t):
        lo = np.searchsorted(ps_c_sorted, p - caliper)
        hi = np.searchsorted(ps_c_sorted, p + caliper)
        if hi <= lo:
            unmatched += 1
            continue
        cand = np.arange(lo, hi)
        take = cand[np.argsort(np.abs(ps_c_sorted[cand] - p))[:k]]
        diffs.append(y_t[i] - y_c_sorted[take].mean())
        matched_treated.append(i)
        matched_ctrl_idx.extend(take.tolist())

    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return Estimate("psm", np.nan, np.nan, 0, 0, "no matches"), pd.DataFrame()

    est = Estimate(
        f"psm_{k}nn_logit_caliper", float(diffs.mean()),
        float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        int(len(diffs)), int(len(diffs) * (1 + k)),
        f"{unmatched} treated rows dropped (no control within caliper)",
    )

    # Reconstruct the matched sample so balance can be re-checked on exactly
    # the rows the estimate was computed from.
    treated_positions = np.flatnonzero(t)[matched_treated]
    ctrl_positions = np.flatnonzero(~t)[order][matched_ctrl_idx]
    matched = pd.concat(
        [df.iloc[treated_positions], df.iloc[ctrl_positions]], ignore_index=True
    )
    return est, matched


def ipw(df: pd.DataFrame, ps: np.ndarray, stabilised: bool = True) -> tuple[Estimate, np.ndarray]:
    """Inverse-probability weighting targeting the ATT.

    ATT weights are 1 for treated rows and ps/(1-ps) for controls, which
    reshapes the control pool to look like the treated pool. Weights are
    trimmed at the 99th percentile: a handful of near-1 propensity controls
    would otherwise dominate the estimate and inflate its variance.
    """
    t = df["treated"].to_numpy(bool)
    y = df["y_obs"].to_numpy(float)

    w = np.ones(len(df))
    odds = ps / (1 - ps)
    w[~t] = odds[~t]
    if stabilised:
        cap = np.quantile(w[~t], 0.99)
        w[~t] = np.minimum(w[~t], cap)

    mt = np.average(y[t], weights=w[t])
    mc = np.average(y[~t], weights=w[~t])
    att = mt - mc

    n_eff = w[~t].sum() ** 2 / (w[~t] ** 2).sum()
    var_c = np.average((y[~t] - mc) ** 2, weights=w[~t])
    se = np.sqrt(y[t].var(ddof=1) / t.sum() + var_c / n_eff)
    return Estimate("ipw_att", float(att), float(se), int(t.sum()), len(df),
                    f"effective control n = {n_eff:.0f}"), w


def aipw(df: pd.DataFrame, ps: np.ndarray) -> Estimate:
    """Doubly-robust AIPW: consistent if *either* the propensity model or the
    outcome model is right. The safest single number in this module."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = _design(df)
    t = df["treated"].to_numpy(int)
    y = df["y_obs"].to_numpy(int)

    mu1 = np.zeros(len(df))
    mu0 = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    for tr, te in skf.split(X, t):
        XT = np.hstack([X[tr], t[tr].reshape(-1, 1)])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(XT, y[tr])
        mu1[te] = m.predict_proba(np.hstack([X[te], np.ones((len(te), 1))]))[:, 1]
        mu0[te] = m.predict_proba(np.hstack([X[te], np.zeros((len(te), 1))]))[:, 1]

    p_t = t.mean()
    # ATT-targeted AIPW influence function. The control odds are trimmed at the
    # 99th percentile for the same reason as in `ipw`: a few near-1 propensity
    # controls otherwise dominate both the point estimate and its variance.
    odds = ps / (1 - ps)
    cap = np.quantile(odds[t == 0], 0.99) if (t == 0).any() else np.inf
    odds = np.minimum(odds, cap)
    psi = (t * (y - mu0) - (1 - t) * odds * (y - mu0)) / p_t
    att = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(len(psi))
    return Estimate("aipw_doubly_robust", float(att), float(se), int(t.sum()), len(df))


def run_all(sim: dict, seed: int = 0) -> dict:
    """Every estimator plus diagnostics, scored against ground truth."""
    from .simulate import true_att

    df_full = build_analysis_frame(sim)
    ps_full = propensity_scores(df_full, seed=seed)

    # The naive number is deliberately computed on the FULL sample: it is the
    # answer someone would get without doing any of this work, and trimming it
    # first would understate the bias being demonstrated.
    est_naive = naive_difference(df_full)

    df, ps, support = trim_common_support(df_full, ps_full)

    est_reg = regression_adjustment(df)
    est_psm, matched = psm_match(df, ps)
    est_ipw, w = ipw(df, ps)
    est_aipw = aipw(df, ps)

    truth = true_att(sim["truth"])
    gt = truth["att"]

    rows = []
    for e in (est_naive, est_reg, est_psm, est_ipw, est_aipw):
        lo, hi = e.ci
        rows.append({
            "method": e.method, "att": e.att, "se": e.se,
            "ci_low": lo, "ci_high": hi,
            "n_treated": e.n_treated, "note": e.note,
            "truth": gt, "abs_error": abs(e.att - gt),
            "pct_error": 100 * (e.att - gt) / gt if gt else np.nan,
            "covers_truth": bool(lo <= gt <= hi),
        })
    results = pd.DataFrame(rows)

    balance_before = standardised_differences(df_full)
    balance_ipw = standardised_differences(df, weights=w)
    balance_matched = standardised_differences(matched) if not matched.empty else balance_ipw
    balance = (
        balance_before[["covariate", "mean_treated", "mean_control", "smd"]]
        .rename(columns={"smd": "smd_unadjusted"})
        .merge(balance_ipw[["covariate", "smd"]].rename(columns={"smd": "smd_ipw"}),
               on="covariate")
        .merge(balance_matched[["covariate", "smd"]].rename(columns={"smd": "smd_matched"}),
               on="covariate")
    )
    balance["max_abs_smd_after"] = balance[["smd_ipw", "smd_matched"]].abs().max(axis=1)

    return {
        "results": results,
        "balance": balance,
        "truth": truth,
        "support": support,
        "propensity": pd.DataFrame({"ps": ps_full, "treated": df_full["treated"]}),
        "analysis_frame": df,
    }
