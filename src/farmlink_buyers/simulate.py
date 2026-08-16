"""Buyer order-book simulator.

Why a simulator
---------------
FarmLink's real transaction ledger is confidential, and no public dataset
describes B2B agricultural buying in South Asia. So the order book is
generated - but it is *not* invented noise. Demand, fill rates and churn are
all driven by the genuine 2016-2026 Sri Lankan wholesale price series, so the
synthetic ledger inherits real seasonality, real volatility clustering and the
real 2022 supply crisis.

The point of building it this way
---------------------------------
Because the data-generating process is written down, **the true causal effect
of a short-shipped order on churn is known exactly**. That turns the causal
section of this repo from an untestable claim into a validated measurement:
the propensity-score estimator is scored against ground truth it never sees.
An estimator that recovers a known answer on data with known confounding is
evidence the same method can be trusted on the real ledger.

Confounding is built in deliberately
------------------------------------
Two structural biases make the naive comparison wrong, exactly as they would be
in production:

* **Large buyers are prioritised** during shortages, so they get better fill
  rates - and they also churn less for unrelated commercial reasons.
* **Price-sensitive segments buy volatile crops**, so they see worse fill rates -
  and they are independently more likely to leave.

Regress churn on fill rate without adjustment and you inherit both biases. The
size of that gap is measured in `causal.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .prices import load_prices, weekly_wholesale


# --------------------------------------------------------------------------
# Ground-truth parameters
# --------------------------------------------------------------------------
@dataclass
class SimConfig:
    """Every knob of the data-generating process, including the causal effects
    the analysis is later required to recover."""

    seed: int = 20260104
    n_buyers: int = 900
    start: str = "2023-01-02"
    end: str = "2026-03-09"

    # --- the estimand -----------------------------------------------------
    # Log-odds bump to the weekly churn hazard from bad-fill memory.
    # Memory is an exponentially-weighted *average* in [0, 1], not a running
    # sum: a sum would grow without bound and drive the hazard to certainty.
    # This is the ground truth `causal.py` must recover without being told.
    beta_bad_fill: float = 1.35
    bad_fill_decay: float = 0.78      # weekly decay of a bad experience
    bad_fill_threshold: float = 0.90  # fill rate below this = short-shipped
    outcome_window_weeks: int = 8     # churn within 8 weeks of the exposure

    # --- confounders (affect BOTH fill rate and churn) --------------------
    # `gamma_*_fill` are stated as effects on *shortfall* log-odds: positive
    # gamma_size_fill means large buyers are protected (less shortfall).
    beta_size_churn: float = -0.85    # large buyers churn less
    gamma_size_fill: float = 0.55     # large buyers are prioritised on fill
    beta_volatility_churn: float = 0.40
    gamma_volatility_fill: float = -0.70  # volatile baskets are short-shipped more

    # --- other churn drivers ---------------------------------------------
    beta_tenure: float = -0.30
    beta_inactivity: float = 0.55
    beta_price_shock: float = 0.45
    base_hazard_logit: float = -4.8   # ~0.8%/week baseline -> ~35%/yr gross churn

    # --- fulfilment -------------------------------------------------------
    shortfall_intercept: float = -2.40   # tuned so ~18% of orders are short-shipped
    shortfall_scarcity: float = 2.50
    max_shortfall: float = 0.45

    # --- demand -----------------------------------------------------------
    price_elasticity: float = -0.45   # log-log quantity response
    margin: float = 0.14              # FarmLink take rate on GMV

    segments: dict = field(default_factory=lambda: {
        # name:        (share, weekly_order_prob, size_mu, price_sens, churn_adj)
        "restaurant":  (0.34, 0.62, 1.00, 1.20, 0.30),
        "hotel":       (0.14, 0.78, 2.30, 0.70, -0.25),
        "supermarket": (0.11, 0.88, 3.60, 0.55, -0.60),
        "canteen":     (0.18, 0.55, 1.60, 1.05, 0.10),
        "exporter":    (0.08, 0.45, 3.10, 0.85, -0.10),
        "wholesaler":  (0.15, 0.70, 2.70, 1.35, 0.20),
    })


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------------------------
# Buyers
# --------------------------------------------------------------------------
def make_buyers(cfg: SimConfig, weeks: pd.DatetimeIndex, items: list[str],
                rng: np.random.Generator) -> pd.DataFrame:
    names = list(cfg.segments)
    shares = np.array([cfg.segments[s][0] for s in names], dtype=float)
    shares = shares / shares.sum()
    seg = rng.choice(names, size=cfg.n_buyers, p=shares)

    # Acquisition ramps up over time, as a growing marketplace's would.
    w_idx = np.arange(len(weeks))
    ramp = 0.25 + 0.75 * (w_idx / max(1, len(weeks) - 1))
    ramp = ramp / ramp.sum()
    signup_i = rng.choice(w_idx, size=cfg.n_buyers, p=ramp)

    size_mu = np.array([cfg.segments[s][2] for s in seg])
    log_size = np.log(size_mu) + rng.normal(0, 0.45, cfg.n_buyers)
    size = np.exp(log_size)

    n_items = rng.integers(3, 9, cfg.n_buyers)
    baskets = [list(rng.choice(items, size=k, replace=False)) for k in n_items]

    buyers = pd.DataFrame({
        "buyer_id": [f"B{i:04d}" for i in range(cfg.n_buyers)],
        "segment": seg,
        "signup_week": weeks[signup_i],
        "size": size,
        "log_size": (log_size - log_size.mean()) / log_size.std(),
        "order_prob": [cfg.segments[s][1] for s in seg],
        "price_sens": [cfg.segments[s][3] for s in seg],
        "churn_adj": [cfg.segments[s][4] for s in seg],
        "basket": baskets,
        "quality_pref": rng.beta(2.5, 2.5, cfg.n_buyers),
    })
    buyers["cohort"] = buyers["signup_week"].dt.to_period("M").astype(str)
    return buyers


# --------------------------------------------------------------------------
# Order generation
# --------------------------------------------------------------------------
def simulate(cfg: SimConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run the full simulation.

    Returns orders, buyers, the weekly buyer panel, and the ground-truth
    potential-outcome table used to validate the causal estimator.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    prices = load_prices()
    wk = weekly_wholesale(prices)
    wk = wk[(wk["week"] >= cfg.start) & (wk["week"] <= cfg.end)]

    weeks = pd.DatetimeIndex(sorted(wk["week"].unique()))
    items = sorted(wk["Item"].unique())
    buyers = make_buyers(cfg, weeks, items, rng)

    # Lookup tables: item x week -> price, scarcity, volatility.
    piv_price = wk.pivot_table(index="week", columns="Item", values="Price").reindex(weeks).ffill()
    piv_scar = wk.pivot_table(index="week", columns="Item", values="scarcity").reindex(weeks).ffill().fillna(0.5)
    piv_vol = wk.pivot_table(index="week", columns="Item", values="vol_13w").reindex(weeks).ffill().fillna(0.1)
    piv_z = wk.pivot_table(index="week", columns="Item", values="price_z").reindex(weeks).ffill().fillna(0.0)

    item_pos = {it: i for i, it in enumerate(piv_price.columns)}
    P = piv_price.to_numpy(float)
    S = piv_scar.to_numpy(float)
    V = piv_vol.to_numpy(float)
    Z = piv_z.to_numpy(float)

    # Long-run mean price per item, the reference for the elasticity response.
    P_ref = np.nanmean(P, axis=0)

    nb, nw = len(buyers), len(weeks)
    basket_idx = [np.array([item_pos[i] for i in b if i in item_pos], dtype=int)
                  for b in buyers["basket"]]
    # A buyer's structural exposure to volatile crops - a confounder.
    basket_vol = np.array([np.nanmean(V[:, bi]) if len(bi) else 0.1 for bi in basket_idx])
    basket_vol_z = (basket_vol - basket_vol.mean()) / (basket_vol.std() + 1e-9)
    buyers["basket_volatility"] = basket_vol_z

    signup_i = np.searchsorted(weeks, buyers["signup_week"].to_numpy())
    log_size = buyers["log_size"].to_numpy()
    order_prob = buyers["order_prob"].to_numpy()
    price_sens = buyers["price_sens"].to_numpy()
    size = buyers["size"].to_numpy()

    order_rows: list[dict] = []
    fill_arr = np.full((nb, nw), np.nan)
    ordered = np.zeros((nb, nw), dtype=bool)
    gmv = np.zeros((nb, nw))
    shock = np.zeros((nb, nw))

    for w in range(nw):
        active = signup_i <= w
        if not active.any():
            continue
        idx = np.flatnonzero(active)

        # Basket-level price pressure this week, in each buyer's own mix.
        b_scar = np.array([S[w, basket_idx[i]].mean() if len(basket_idx[i]) else 0.5 for i in idx])
        b_z = np.array([Z[w, basket_idx[i]].mean() if len(basket_idx[i]) else 0.0 for i in idx])
        shock[idx, w] = b_z

        # --- does the buyer place an order this week? --------------------
        # High prices suppress order frequency, more so for price-sensitive
        # segments. Festival weeks lift it.
        wk_of_year = weeks[w].isocalendar().week
        festival_lift = 0.10 if wk_of_year in (15, 16, 51, 52, 2, 3) else 0.0
        p_order = np.clip(
            order_prob[idx] * (1 - 0.16 * price_sens[idx] * np.clip(b_z, 0, None)) + festival_lift,
            0.02, 0.98,
        )
        places = rng.random(len(idx)) < p_order
        ordering = idx[places]
        ordered[ordering, w] = True

        if len(ordering) == 0:
            continue

        # --- fill rate ----------------------------------------------------
        # Scarcity drives shortfall; large buyers are protected, volatile
        # baskets are exposed. Both links are confounders by construction.
        scar_o = np.array([S[w, basket_idx[i]].mean() if len(basket_idx[i]) else 0.5
                           for i in ordering])
        shortfall_logit = (
            cfg.shortfall_intercept
            + cfg.shortfall_scarcity * (scar_o - 0.5)
            - cfg.gamma_size_fill * log_size[ordering]
            - cfg.gamma_volatility_fill * basket_vol_z[ordering]
            + rng.normal(0, 0.55, len(ordering))
        )
        fill = np.clip(1.0 - cfg.max_shortfall * _sigmoid(shortfall_logit), 0.25, 1.0)
        fill_arr[ordering, w] = fill

        # --- order lines --------------------------------------------------
        for k, bi in enumerate(ordering):
            items_b = basket_idx[bi]
            if len(items_b) == 0:
                continue
            n_lines = min(len(items_b), rng.integers(2, 6))
            chosen = rng.choice(items_b, size=n_lines, replace=False)

            price_now = P[w, chosen]
            ref = P_ref[chosen]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where((ref > 0) & np.isfinite(price_now), price_now / ref, 1.0)
            rel = np.nan_to_num(rel, nan=1.0, posinf=1.0)

            # Log-log demand response, scaled by segment price sensitivity.
            elas = cfg.price_elasticity * price_sens[bi]
            base_qty = 22.0 * size[bi]
            qty_ord = base_qty * np.power(np.clip(rel, 0.2, 5.0), elas)
            qty_ord = np.maximum(1.0, qty_ord * rng.lognormal(0, 0.28, n_lines))
            qty_del = qty_ord * fill[k]

            line_gmv = float(np.nansum(qty_del * np.nan_to_num(price_now)))
            gmv[bi, w] += line_gmv

            for j, ci in enumerate(chosen):
                order_rows.append({
                    "buyer_id": buyers.buyer_id.iloc[bi],
                    "week": weeks[w],
                    "item": piv_price.columns[ci],
                    "qty_ordered_kg": round(float(qty_ord[j]), 2),
                    "qty_delivered_kg": round(float(qty_del[j]), 2),
                    "unit_price_lkr": round(float(np.nan_to_num(price_now[j])), 2),
                    "line_gmv_lkr": round(float(np.nan_to_num(qty_del[j] * price_now[j])), 2),
                    "fill_rate": round(float(fill[k]), 4),
                })

    orders = pd.DataFrame(order_rows)
    orders["revenue_lkr"] = (orders["line_gmv_lkr"] * cfg.margin).round(2)

    panel, truth = _build_panel_and_truth(cfg, buyers, weeks, ordered, fill_arr, gmv,
                                          shock, signup_i, log_size, rng)

    buyers = buyers.drop(columns=["basket"]).merge(
        panel.groupby("buyer_id", as_index=False).agg(
            churn_week=("churned_this_week", lambda s: s.idxmax() if s.any() else np.nan)
        ),
        on="buyer_id", how="left",
    )

    return {"orders": orders, "buyers": buyers, "panel": panel,
            "truth": truth, "config": pd.DataFrame([asdict(cfg)])}


def _build_panel_and_truth(cfg, buyers, weeks, ordered, fill_arr, gmv, shock,
                           signup_i, log_size, rng):
    """Weekly buyer panel, the churn process, and exact potential outcomes.

    The hazard path is written down explicitly, so both potential outcomes -
    what would have happened with and without a short-shipped order in a given
    week - are computed analytically rather than estimated. No Monte Carlo
    noise enters the ground truth.
    """
    nb, nw = ordered.shape
    decay, thr = cfg.bad_fill_decay, cfg.bad_fill_threshold

    bad = (fill_arr < thr) & ~np.isnan(fill_arr)

    # Exponentially-weighted *average* of bad experiences, bounded in [0, 1].
    # The (1 - decay) normaliser is what keeps the hazard in a sane range;
    # an unnormalised running sum saturates the logit within a few months.
    mem = np.zeros((nb, nw))
    for w in range(1, nw):
        mem[:, w] = decay * mem[:, w - 1] + (1 - decay) * bad[:, w - 1].astype(float)

    # Weeks since last order, capped.
    inact = np.zeros((nb, nw))
    for w in range(1, nw):
        inact[:, w] = np.where(ordered[:, w - 1], 0.0, np.minimum(inact[:, w - 1] + 1, 12))

    tenure = np.maximum(0, np.arange(nw)[None, :] - signup_i[:, None]) / 52.0
    active_mask = np.arange(nw)[None, :] >= signup_i[:, None]

    def hazard(mem_arr: np.ndarray) -> np.ndarray:
        lin = (
            cfg.base_hazard_logit
            + cfg.beta_bad_fill * mem_arr
            + cfg.beta_size_churn * log_size[:, None]
            + cfg.beta_volatility_churn * buyers["basket_volatility"].to_numpy()[:, None]
            + cfg.beta_tenure * tenure
            + cfg.beta_inactivity * (inact / 4.0)
            + cfg.beta_price_shock * np.clip(shock, 0, None)
            + buyers["churn_adj"].to_numpy()[:, None]
        )
        return _sigmoid(lin)

    h = hazard(mem)
    h = np.where(active_mask, h, 0.0)

    # Realised churn: first week the hazard fires. Uniforms are fixed so the
    # factual path is reproducible.
    u = rng.random((nb, nw))
    fires = (u < h) & active_mask
    churn_w = np.where(fires.any(axis=1), fires.argmax(axis=1), nw)
    alive = np.arange(nw)[None, :] <= churn_w[:, None]

    churned_this_week = np.zeros((nb, nw), dtype=bool)
    has_churn = churn_w < nw
    churned_this_week[np.flatnonzero(has_churn), churn_w[has_churn]] = True

    # ---- ground-truth potential outcomes --------------------------------
    W = cfg.outcome_window_weeks

    def survive_prob(h_arr: np.ndarray, w: int) -> np.ndarray:
        """P(churn within W weeks after w) under a given hazard path."""
        hi = min(nw, w + 1 + W)
        if hi <= w + 1:
            return np.zeros(nb)
        seg = h_arr[:, w + 1:hi]
        return 1.0 - np.prod(1.0 - seg, axis=1)

    rows = []
    treat_mat = bad & ordered
    for w in range(nw - W):
        cand = np.flatnonzero(active_mask[:, w] & alive[:, w] & ordered[:, w])
        if len(cand) == 0:
            continue

        # Counterfactual memories: force week-w treatment on and off, then
        # propagate forward. Everything else is held fixed.
        mem1 = mem.copy()
        mem0 = mem.copy()
        for k in range(w + 1, nw):
            step = (1 - decay) * decay ** (k - w - 1)
            base = mem[:, k] - step * bad[:, w].astype(float)
            mem1[:, k] = base + step * 1.0
            mem0[:, k] = base

        y1 = survive_prob(np.where(active_mask, hazard(mem1), 0.0), w)
        y0 = survive_prob(np.where(active_mask, hazard(mem0), 0.0), w)

        # Realised (factual) outcome: did this buyer actually churn inside the
        # window? This is the ONLY outcome the observational analysis is
        # allowed to see - y1/y0 are held back purely to score the estimator.
        y_obs = ((churn_w[cand] > w) & (churn_w[cand] <= w + W)).astype(int)

        rows.append(pd.DataFrame({
            "buyer_id": buyers.buyer_id.to_numpy()[cand],
            "week": weeks[w],
            "treated": treat_mat[cand, w].astype(int),
            "y_obs": y_obs,
            "y1": y1[cand],
            "y0": y0[cand],
            "ite": (y1 - y0)[cand],
        }))

    truth = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # ---- observable weekly panel ----------------------------------------
    bidx = np.repeat(np.arange(nb), nw)
    widx = np.tile(np.arange(nw), nb)
    panel = pd.DataFrame({
        "buyer_id": buyers.buyer_id.to_numpy()[bidx],
        "week": weeks[widx],
        "is_active": active_mask.ravel(),
        "alive": alive.ravel(),
        "ordered": ordered.ravel(),
        "fill_rate": fill_arr.ravel(),
        "gmv_lkr": gmv.ravel(),
        "weeks_inactive": inact.ravel(),
        "tenure_years": tenure.ravel(),
        "price_shock": shock.ravel(),
        "churned_this_week": churned_this_week.ravel(),
    })
    panel = panel[panel["is_active"] & panel["alive"]].reset_index(drop=True)
    panel["short_shipped"] = (panel["fill_rate"] < thr).fillna(False)
    panel["revenue_lkr"] = panel["gmv_lkr"] * cfg.margin
    return panel, truth


def true_att(truth: pd.DataFrame) -> dict:
    """Ground-truth ATT: the average effect among buyer-weeks actually treated."""
    t = truth[truth["treated"] == 1]
    return {
        "att": float(t["ite"].mean()),
        "ate": float(truth["ite"].mean()),
        "n_treated": int(len(t)),
        "n_total": int(len(truth)),
        "naive_gap": float(
            truth.loc[truth.treated == 1, "y1"].mean()
            - truth.loc[truth.treated == 0, "y0"].mean()
        ),
    }
