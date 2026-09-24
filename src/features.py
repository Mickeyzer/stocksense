"""Feature engineering for the global LightGBM model.

Every demand-derived feature is lagged by at least the forecast horizon (28 days), so
one model forecasts all 28 days at once without recursion and without leakage. Prices,
calendar and events are known in advance (M5 publishes planned prices), so they are
used as of the target day.
"""
import numpy as np
import pandas as pd

from src import config

LAGS = [28, 35, 42, 49, 56, 364]
ROLL_WINDOWS = [7, 28, 56]
CATEGORICAL = ["item_id", "store_id", "event_type", "dow"]


def add_features(panel: pd.DataFrame, extra: bool = False) -> pd.DataFrame:
    df = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()
    g = df.groupby("unique_id", sort=False)["y"]
    h = config.HORIZON

    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")

    base = df[f"lag_{h}"]
    gb = base.groupby(df["unique_id"], sort=False)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = gb.transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")
    df["roll_std_28"] = gb.transform(lambda s: s.rolling(28, min_periods=7).std()).astype("float32")
    # Share of zero-sale days: how intermittent the item has recently been.
    df["zero_rate_28"] = (base == 0).astype("float32").groupby(df["unique_id"], sort=False) \
        .transform(lambda s: s.rolling(28, min_periods=7).mean()).astype("float32")

    p = df.groupby("unique_id", sort=False)["sell_price"]
    df["price_rel_max"] = (df["sell_price"] / p.cummax()).astype("float32")
    df["price_change_7"] = (df["sell_price"] / p.shift(7) - 1).astype("float32")
    df["price_rel_mean"] = (df["sell_price"] / p.transform("mean")).astype("float32")

    if extra:
        # Same-weekday average over four weeks (lags 28/35/42/49 share the target's weekday).
        df["dow_mean_4"] = df[["lag_28", "lag_35", "lag_42", "lag_49"]].mean(axis=1).astype("float32")
        df["ewm_28"] = gb.transform(lambda s: s.ewm(alpha=0.1, ignore_na=True).mean()).astype("float32")
        df["price_change_28"] = (df["sell_price"] / p.shift(28) - 1).astype("float32")
        # Chain-wide popularity of the item: its recent demand averaged across the four stores.
        df["item_roll_28"] = df.groupby(["item_id", "ds"], observed=True)["roll_mean_28"] \
            .transform("mean").astype("float32")
        df["store_share"] = (df["roll_mean_28"] / (df["item_roll_28"] + 1e-3)).astype("float32")

    df["dow"] = df["ds"].dt.dayofweek.astype("int8")
    df["dom"] = df["ds"].dt.day.astype("int8")
    df["month"] = df["ds"].dt.month.astype("int8")
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype("int8")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"unique_id", "ds", "y", "event_name_1"}
    return [c for c in df.columns if c not in exclude]
