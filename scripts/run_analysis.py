"""End-to-end analysis run.

Reproduces every number quoted in the README:

    python scripts/run_analysis.py

Outputs land in reports/results/ as CSV, plus a headline.json summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farmlink_buyers import causal, churn, experiment, metrics, simulate  # noqa: E402

RESULTS = ROOT / "reports" / "results"
PROCESSED = ROOT / "data" / "processed"
for p in (RESULTS, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    stamp = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    # ---- simulate --------------------------------------------------------
    stamp("simulating order book from real price series")
    cfg = simulate.SimConfig()
    sim = simulate.simulate(cfg)
    orders, buyers, panel = sim["orders"], sim["buyers"], sim["panel"]
    orders.to_parquet(PROCESSED / "orders.parquet", index=False)
    buyers.to_parquet(PROCESSED / "buyers.parquet", index=False)
    panel.to_parquet(PROCESSED / "panel.parquet", index=False)

    stamp(f"{len(orders):,} order lines | {len(buyers)} buyers | "
          f"{len(panel):,} buyer-weeks | GMV {orders.line_gmv_lkr.sum()/1e6:.0f}M LKR")

    # ---- descriptive metrics --------------------------------------------
    stamp("cohorts, RFM, LTV")
    ret = metrics.cohort_retention(panel, buyers)
    ret.to_csv(RESULTS / "cohort_retention.csv")
    curve = metrics.retention_curve(ret)
    curve.to_csv(RESULTS / "retention_curve.csv", index=False)
    print("\n=== AVERAGE RETENTION CURVE ===")
    print(curve.head(13).to_string(index=False))

    rfm_tbl = metrics.rfm(orders)
    rfm_tbl.to_csv(RESULTS / "rfm.csv", index=False)
    seg_summary = (
        rfm_tbl.groupby("rfm_segment", as_index=False)
        .agg(n=("buyer_id", "size"), revenue=("revenue", "sum"),
             avg_fill=("avg_fill", "mean"), freq=("frequency", "mean"))
    )
    seg_summary["revenue_share_pct"] = 100 * seg_summary["revenue"] / seg_summary["revenue"].sum()
    seg_summary = seg_summary.sort_values("revenue", ascending=False, ignore_index=True)
    seg_summary.to_csv(RESULTS / "rfm_segments.csv", index=False)
    print("\n=== RFM SEGMENTS ===")
    print(seg_summary.to_string(index=False))

    personas = metrics.segment_personas(rfm_tbl)
    personas.to_csv(RESULTS / "personas.csv", index=False)

    ltv = metrics.cohort_ltv(orders, buyers)
    ltv.to_csv(RESULTS / "cohort_ltv.csv", index=False)
    ltv_sum = metrics.ltv_summary(ltv)
    ltv_sum.to_csv(RESULTS / "ltv_summary.csv", index=False)
    print("\n=== LTV ===")
    print(ltv_sum.to_string(index=False))

    fill_ret = metrics.fill_rate_vs_retention(panel)
    fill_ret.to_csv(RESULTS / "fill_vs_churn.csv", index=False)
    print("\n=== CHURN BY FILL-RATE BAND (descriptive, confounded) ===")
    print(fill_ret.to_string(index=False))

    ns = metrics.north_star_table(orders, panel)
    ns.to_csv(RESULTS / "north_star_weekly.csv", index=False)

    # ---- churn model -----------------------------------------------------
    stamp("churn model (temporal split)")
    frame = churn.build_training_frame(sim)
    cm, (Xtr, ytr, tr_df), (Xte, yte, te_df) = churn.train(frame)
    ev = churn.evaluate(cm, Xte, yte)
    print("\n=== CHURN MODEL (out-of-time) ===")
    print(json.dumps(ev, indent=2))
    pd.DataFrame([ev]).to_csv(RESULTS / "churn_metrics.csv", index=False)

    # Intervention economics derived from the data rather than assumed: the
    # margin at risk when a buyer leaves is their expected forward 26 weeks of
    # contribution, not a round number picked to make the campaign look good.
    rev_per_buyer_week = float(panel["revenue_lkr"].mean())
    margin_at_risk = rev_per_buyer_week * 26
    print(f"\nrevenue/buyer-week = {rev_per_buyer_week:,.0f} LKR  ->  "
          f"26-week margin at risk = {margin_at_risk:,.0f} LKR")

    p_te = cm.model.predict_proba(Xte)[:, 1]
    ev_tbl = churn.expected_value_threshold(
        yte.to_numpy(), p_te,
        contact_cost_lkr=2500.0,
        margin_saved_lkr=margin_at_risk,
        save_rate=0.30,
    )
    ev_tbl.to_csv(RESULTS / "churn_threshold_value.csv", index=False)
    best = ev_tbl.loc[ev_tbl["net_value_lkr"].idxmax()]
    print("\n=== OPTIMAL INTERVENTION THRESHOLD ===")
    print(best.to_string())

    shap_tbl = churn.shap_importance(cm, Xte)
    shap_tbl.to_csv(RESULTS / "churn_shap.csv", index=False)
    print("\n=== TOP CHURN DRIVERS (mean |SHAP|) ===")
    print(shap_tbl.head(10).to_string(index=False))

    # ---- causal ----------------------------------------------------------
    stamp("causal: effect of short-shipping on churn")
    cres = causal.run_all(sim)
    cres["results"].to_csv(RESULTS / "causal_estimates.csv", index=False)
    cres["balance"].to_csv(RESULTS / "causal_balance.csv", index=False)
    print("\n=== COMMON SUPPORT ===")
    print(json.dumps(cres["support"], indent=2))
    print("\n=== CAUSAL ESTIMATES vs GROUND TRUTH ===")
    print(cres["results"][["method", "att", "ci_low", "ci_high", "truth",
                           "abs_error", "pct_error", "covers_truth"]]
          .round(4).to_string(index=False))
    print("\n=== COVARIATE BALANCE (|SMD| < 0.1 = balanced) ===")
    print(cres["balance"].round(3).to_string(index=False))

    # ---- experiment design ----------------------------------------------
    stamp("experiment design")
    base_churn = float(frame["churn_next"].mean())
    design = experiment.design_table(base_churn, n_buyers=len(buyers))
    design.to_csv(RESULTS / "experiment_design.csv", index=False)
    print("\n=== EXPERIMENT DESIGN ===")
    print(design.round(4).to_string(index=False))

    pre_post = (
        panel.assign(half=np.where(panel["week"] < panel["week"].median(), "pre", "post"))
        .pivot_table(index="buyer_id", columns="half", values="gmv_lkr", aggfunc="sum")
        .dropna()
    )
    cuped = experiment.cuped_variance_reduction(pre_post["pre"], pre_post["post"])
    print("\n=== CUPED ===")
    print(json.dumps(cuped, indent=2))
    pd.DataFrame([cuped]).to_csv(RESULTS / "cuped.csv", index=False)

    seq = experiment.sequential_boundary(5)
    seq.to_csv(RESULTS / "sequential_boundaries.csv", index=False)

    # ---- headline --------------------------------------------------------
    truth = cres["truth"]
    aipw_row = cres["results"].set_index("method").loc["aipw_doubly_robust"]
    psm_row = cres["results"][cres["results"].method.str.startswith("psm")].iloc[0]
    naive_row = cres["results"].set_index("method").loc["naive_difference"]

    headline = {
        "simulation": {
            "buyers": int(len(buyers)),
            "order_lines": int(len(orders)),
            "buyer_weeks": int(len(panel)),
            "weeks": int(panel["week"].nunique()),
            "date_start": str(panel["week"].min().date()),
            "date_end": str(panel["week"].max().date()),
            "gmv_lkr_m": round(float(orders.line_gmv_lkr.sum() / 1e6), 1),
            "revenue_lkr_m": round(float(orders.revenue_lkr.sum() / 1e6), 1),
            "mean_fill_rate": round(float(orders.fill_rate.mean()), 4),
            "short_ship_pct": round(100 * float((orders.fill_rate < 0.9).mean()), 1),
        },
        "retention": {
            "month_1": round(float(curve.loc[curve.months_since == 1, "retention"].iloc[0]), 3),
            "month_3": round(float(curve.loc[curve.months_since == 3, "retention"].iloc[0]), 3),
            "month_6": round(float(curve.loc[curve.months_since == 6, "retention"].iloc[0]), 3),
            "month_12": round(float(curve.loc[curve.months_since == 12, "retention"].iloc[0]), 3)
            if (curve.months_since == 12).any() else None,
        },
        "churn_model": ev,
        "churn_threshold": {
            "threshold": round(float(best["threshold"]), 3),
            "net_value_lkr": round(float(best["net_value_lkr"]), 0),
            "precision": round(float(best["precision"]), 3),
            "recall": round(float(best["recall"]), 3),
            "n_contacted": int(best["n_contacted"]),
        },
        "causal": {
            "ground_truth_att": round(truth["att"], 4),
            "naive_att": round(float(naive_row["att"]), 4),
            "naive_overstatement_x": round(float(naive_row["att"]) / truth["att"], 1),
            "psm_att": round(float(psm_row["att"]), 4),
            "aipw_att": round(float(aipw_row["att"]), 4),
            "aipw_abs_error": round(float(aipw_row["abs_error"]), 4),
            "aipw_covers_truth": bool(aipw_row["covers_truth"]),
            "n_treated": int(truth["n_treated"]),
        },
        "cuped": cuped,
        "base_churn_8w": round(base_churn, 4),
    }
    (RESULTS / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    print("\n=== HEADLINE ===")
    print(json.dumps(headline, indent=2, default=str))
    stamp("done")


if __name__ == "__main__":
    main()
