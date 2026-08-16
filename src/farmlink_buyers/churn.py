"""Churn prediction: who is about to stop ordering, and what should we do?

Two things separate this from a stock churn notebook:

**The split is temporal, not random.** Buyer-weeks are serially correlated -
the same account appears many times - so a random split leaks a buyer's future
into its own training rows and produces an AUC that evaporates in production.
Training stops at a cut-off date; testing happens strictly after it.

**The threshold is chosen by expected value, not by F1.** The question is not
"is this classifier good" but "at what score should we spend LKR X on a
retention call". That depends on the cost of the call, the margin saved, and
the base rate - so the optimal threshold is derived from those, and the payoff
curve is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = [
    "tenure_years", "weeks_inactive", "price_shock", "orders_last_8w",
    "gmv_last_8w_log", "prior_fill_mean", "prior_short_rate", "log_size",
    "basket_volatility", "quality_pref", "gmv_trend", "fill_trend",
    "orders_last_4w", "weeks_since_short_ship",
]


def build_training_frame(sim: dict, horizon_weeks: int = 8) -> pd.DataFrame:
    """Buyer-week rows with strictly historical features and a forward label."""
    panel, buyers = sim["panel"], sim["buyers"]
    p = panel.sort_values(["buyer_id", "week"]).copy()
    g = p.groupby("buyer_id", sort=False)

    p["orders_last_8w"] = g["ordered"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).sum())
    p["orders_last_4w"] = g["ordered"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).sum())
    p["gmv_last_8w"] = g["gmv_lkr"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).sum())
    p["gmv_last_8w_log"] = np.log1p(p["gmv_last_8w"])
    p["gmv_prev_8w"] = g["gmv_lkr"].transform(
        lambda s: s.shift(9).rolling(8, min_periods=1).sum())
    # Trend beats level: a buyer halving their spend is the signal, not the
    # absolute size, which is already captured by log_size.
    p["gmv_trend"] = np.log1p(p["gmv_last_8w"]) - np.log1p(p["gmv_prev_8w"])
    p["prior_fill_mean"] = g["fill_rate"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    p["fill_recent"] = g["fill_rate"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    p["fill_trend"] = p["fill_recent"] - p["prior_fill_mean"]
    p["prior_short_rate"] = g["short_shipped"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean())

    def since_short(s: pd.Series) -> pd.Series:
        out, last = [], np.nan
        for i, v in enumerate(s.to_numpy()):
            out.append(np.nan if np.isnan(last) else i - last)
            if v:
                last = i
        return pd.Series(out, index=s.index)

    p["weeks_since_short_ship"] = g["short_shipped"].transform(since_short).fillna(52)

    # Forward label: does the buyer churn within the next `horizon_weeks`?
    #
    # pandas `.rolling()` is always a TRAILING window, so the natural-looking
    # `s.shift(-1).rolling(h).max()` actually looks mostly backwards and
    # silently builds a label from the past. Reversing the series, rolling,
    # and reversing back is the correct forward window.
    def forward_any(s: pd.Series, w: int) -> pd.Series:
        fwd = s[::-1].rolling(w, min_periods=1).max()[::-1]
        return fwd.shift(-1)

    p["churn_next"] = (
        g["churned_this_week"]
        .transform(lambda s: forward_any(s, horizon_weeks))
        .fillna(0)
        .astype(int)
    )

    df = p.merge(
        buyers[["buyer_id", "segment", "log_size", "basket_volatility", "quality_pref"]],
        on="buyer_id", how="left",
    )
    df["prior_fill_mean"] = df["prior_fill_mean"].fillna(df["prior_fill_mean"].mean())
    df["fill_trend"] = df["fill_trend"].fillna(0.0)
    df["gmv_trend"] = df["gmv_trend"].fillna(0.0)
    df["prior_short_rate"] = df["prior_short_rate"].fillna(0.0)

    # Drop the final horizon: those rows cannot have a complete label yet.
    cutoff = df["week"].max() - pd.Timedelta(weeks=horizon_weeks)
    return df[df["week"] <= cutoff].dropna(subset=FEATURES).reset_index(drop=True)


@dataclass
class ChurnModel:
    model: object
    features: list[str]
    test_dates: tuple

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features]
        return self.model.predict_proba(X)[:, 1]


def train(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 0):
    """Temporal split, gradient-boosted trees, calibrated probabilities."""
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    weeks = np.sort(df["week"].unique())
    split = weeks[int(len(weeks) * (1 - test_fraction))]
    train_df = df[df["week"] < split]
    test_df = df[df["week"] >= split]

    seg = pd.get_dummies(df["segment"], prefix="seg", drop_first=True)
    feats = FEATURES + list(seg.columns)
    Xall = pd.concat([df[FEATURES], seg], axis=1)

    Xtr, ytr = Xall.loc[train_df.index], train_df["churn_next"]
    Xte, yte = Xall.loc[test_df.index], test_df["churn_next"]

    base = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
        min_child_weight=8, eval_metric="logloss", random_state=seed,
        # Churn is rare; without this the model predicts "nobody churns".
        scale_pos_weight=float((ytr == 0).sum() / max(1, (ytr == 1).sum())),
    )
    # Calibration matters because the threshold is chosen on expected value,
    # which needs probabilities that mean what they say.
    model = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xtr, ytr)

    cm = ChurnModel(model, feats, (split, df["week"].max()))
    return cm, (Xtr, ytr, train_df), (Xte, yte, test_df)


def evaluate(cm: ChurnModel, Xte: pd.DataFrame, yte: pd.Series) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)

    p = cm.model.predict_proba(Xte)[:, 1]
    base_rate = float(yte.mean())
    out = {
        "n_test": int(len(yte)),
        "base_rate": base_rate,
        "roc_auc": float(roc_auc_score(yte, p)),
        "pr_auc": float(average_precision_score(yte, p)),
        # Lift over the base rate is the number that survives a class-imbalance
        # sceptic; PR-AUC alone is unreadable without it.
        "pr_auc_lift": float(average_precision_score(yte, p) / base_rate),
        "brier": float(brier_score_loss(yte, p)),
    }
    # Precision in the top decile: the operationally relevant slice, since a
    # retention team can only call so many accounts a week.
    k = max(1, int(0.10 * len(p)))
    top = np.argsort(-p)[:k]
    out["top_decile_precision"] = float(yte.to_numpy()[top].mean())
    out["top_decile_lift"] = out["top_decile_precision"] / base_rate
    return out


def expected_value_threshold(y_true: np.ndarray, p: np.ndarray,
                             contact_cost_lkr: float = 2500.0,
                             margin_saved_lkr: float = 42000.0,
                             save_rate: float = 0.30) -> pd.DataFrame:
    """Payoff of intervening at each score threshold.

    Every buyer above the threshold gets a retention call costing
    `contact_cost_lkr`. If they really were going to churn, the call saves them
    with probability `save_rate`, worth `margin_saved_lkr` of forward margin.
    The optimum is where marginal call cost equals marginal expected save.
    """
    rows = []
    for thr in np.linspace(0.01, 0.9, 90):
        flagged = p >= thr
        n = int(flagged.sum())
        if n == 0:
            continue
        tp = int((flagged & (y_true == 1)).sum())
        value = tp * save_rate * margin_saved_lkr - n * contact_cost_lkr
        rows.append({
            "threshold": float(thr), "n_contacted": n, "true_churners": tp,
            "precision": tp / n,
            "recall": tp / max(1, int((y_true == 1).sum())),
            "net_value_lkr": float(value),
            "value_per_contact_lkr": float(value / n),
        })
    return pd.DataFrame(rows)


def shap_importance(cm: ChurnModel, X: pd.DataFrame, max_rows: int = 4000) -> pd.DataFrame:
    """Mean |SHAP| per feature, from the underlying trees.

    The calibrated wrapper hides the booster, so the first fitted base
    estimator is unwrapped for explanation. That is exact for ranking drivers:
    isotonic calibration is monotone and reorders nothing.
    """
    import shap

    Xs = X.sample(min(max_rows, len(X)), random_state=0)
    booster = cm.model.calibrated_classifiers_[0].estimator
    expl = shap.TreeExplainer(booster)
    vals = expl.shap_values(Xs)
    if isinstance(vals, list):
        vals = vals[-1]
    imp = pd.DataFrame({
        "feature": Xs.columns,
        "mean_abs_shap": np.abs(vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp
