"""Real Sri Lankan wholesale prices, used to drive the order simulation.

This repo is deliberately standalone, so it fetches the same public source as
the price-forecasting project rather than importing from it: daily Dambulla /
Pettah wholesale and retail prices published in Central Bank of Sri Lanka daily
reports and redistributed as the CRAN dataset `vegetablesSriLanka` (GPL-3).

Prices are the *exogenous driver* of the simulation. Demand, fill rates and
churn all respond to the genuine 2016-2026 price path - including the 2022
crisis - so the synthetic order book inherits real seasonality, real
volatility clustering and real supply shocks instead of invented noise.
"""
from __future__ import annotations

import io
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

CRAN_URL = "https://cran.r-project.org/src/contrib/vegetablesSriLanka_1.1.0.tar.gz"
RDA_MEMBER = "vegetablesSriLanka/data/vegetables.srilanka.rda"


def load_prices(force: bool = False) -> pd.DataFrame:
    """Tidy daily prices: Date, Item, Type, Market, Price (LKR/kg)."""
    cache = RAW / "prices_raw.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    import rdata

    rda = RAW / "vegetables_srilanka.rda"
    if not rda.exists() or force:
        resp = requests.get(CRAN_URL, timeout=120,
                            headers={"User-Agent": "farmlink-buyer-analytics/1.0"})
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            member = tar.extractfile(RDA_MEMBER)
            if member is None:
                raise RuntimeError("dataset missing from CRAN tarball")
            rda.write_bytes(member.read())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = rdata.conversion.convert(rdata.parser.parse_file(rda))
    raw = conv["vegetables.srilanka"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(raw["Date"], unit="D", origin="1970-01-01"),
        "Item": raw["Item"].astype(str),
        "Type": raw["Type"].astype(str),
        "Market": raw["Market"].astype(str),
        "Price": pd.to_numeric(raw["Price"], errors="coerce"),
    })
    df["Price"] = df["Price"].where(df["Price"] >= 1.0)
    df = df.sort_values(["Item", "Market", "Type", "Date"], ignore_index=True)
    df.to_parquet(cache, index=False)
    return df


def weekly_wholesale(prices: pd.DataFrame, market: str = "Dambulla") -> pd.DataFrame:
    """Weekly wholesale price per item, plus the volatility and shock signals
    the simulator uses to drive supply scarcity.

    `price_z` is a trailing 26-week z-score: how unusual this week's price is
    for this crop. It stands in for scarcity, because in a supply-constrained
    market a price spike and a shortage are the same event.
    """
    ws = prices[(prices["Type"] == "Wholesale") & (prices["Market"] == market)].copy()
    ws = ws.dropna(subset=["Price"])
    ws["week"] = ws["Date"].dt.to_period("W-MON").dt.start_time

    wk = (
        ws.groupby(["Item", "week"], as_index=False)["Price"]
        .mean()
        .sort_values(["Item", "week"], ignore_index=True)
    )

    g = wk.groupby("Item")["Price"]
    wk["log_price"] = np.log(wk["Price"])
    wk["ret_1w"] = wk.groupby("Item")["log_price"].diff()
    roll_mean = g.transform(lambda s: np.log(s).rolling(26, min_periods=8).mean())
    roll_std = g.transform(lambda s: np.log(s).rolling(26, min_periods=8).std())
    wk["price_z"] = ((wk["log_price"] - roll_mean) / roll_std.replace(0, np.nan)).fillna(0.0)
    wk["vol_13w"] = (
        wk.groupby("Item")["ret_1w"].transform(lambda s: s.rolling(13, min_periods=5).std())
    ).fillna(0.0)

    # Scarcity index in [0, 1]: high when price is far above its own recent norm.
    wk["scarcity"] = 1.0 / (1.0 + np.exp(-1.2 * wk["price_z"]))
    return wk
