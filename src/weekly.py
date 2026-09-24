"""Direct weekly demand forecasts (mean + 10 quantiles) for replenishment planning.

Replenishment happens weekly, so the planning model predicts the *sum* of next week's
demand directly (quantiles of a sum are not the sum of daily quantiles). One row is
(series, forecast origin, weeks-ahead k); features only use data up to the origin,
plus prices/calendar that are known in advance for the target week.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.data import cutoffs, load_panel

HISTORY_START = "2014-06-01"  # earliest forecast origin used for training
WEEKS_AHEAD = 4
BASE_PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
               "seed": config.SEED, "verbose": -1, "num_threads": 0}
ROUNDS = 400


class Panel:
    """Dense series x day matrices, so window sums are O(1) via cumulative sums."""

    def __init__(self, panel: pd.DataFrame):
        self.days = pd.date_range(panel["ds"].min(), panel["ds"].max(), freq="D")
        y = panel.pivot(index="unique_id", columns="ds", values="y").reindex(columns=self.days)
        price = panel.pivot(index="unique_id", columns="ds", values="sell_price").reindex(columns=self.days)
        self.ids = y.index.to_numpy()
        meta = panel.drop_duplicates("unique_id").set_index("unique_id").loc[self.ids]
        self.item = meta["item_id"].to_numpy()
        self.store = meta["store_id"].to_numpy()

        Y = y.to_numpy(dtype="float64")
        self.launch = np.argmax(~np.isnan(Y), axis=1)
        self.Y = np.nan_to_num(Y)
        self.P = price.ffill(axis=1).bfill(axis=1).to_numpy(dtype="float64")
        zero = ((Y == 0) & ~np.isnan(Y)).astype("float64")
        pad = lambda a: np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
        self.C, self.CZ, self.CP = pad(self.Y), pad(zero), pad(self.P)

        cal = panel.drop_duplicates("ds").set_index("ds").reindex(self.days)
        self.cev = np.concatenate([[0], np.cumsum(cal["event"].fillna(0).to_numpy())])
        self.csnap = np.concatenate([[0], np.cumsum(cal["snap"].fillna(0).to_numpy())])

    def idx(self, day) -> int:
        return int((pd.Timestamp(day) - self.days[0]).days)

    def window(self, C, s, e):
        """Sum over day indices [s, e) for every series (clipped at 0)."""
        return C[:, max(e, 0)] - C[:, max(s, 0)]


def build_rows(pn: Panel, origin_idx: int, ks) -> pd.DataFrame:
    t1 = origin_idx + 1  # first day after the origin
    last7 = pn.window(pn.C, t1 - 7, t1)
    last28 = pn.window(pn.C, t1 - 28, t1)
    last56 = pn.window(pn.C, t1 - 56, t1)
    weeks = np.stack([pn.window(pn.C, t1 - 7 * (j + 1), t1 - 7 * j) for j in range(8)], axis=1)
    zero28 = pn.window(pn.CZ, t1 - 28, t1) / 28
    price_now = pn.P[:, origin_idx]
    age = origin_idx - pn.launch

    frames = []
    for k in ks:
        s, e = t1 + 7 * (k - 1), t1 + 7 * k
        target = pn.window(pn.C, s, e) if e <= pn.Y.shape[1] else np.full(len(pn.ids), np.nan)
        ly = pn.window(pn.C, s - 364, e - 364) if s - 364 >= 0 else np.full(len(pn.ids), np.nan)
        ly[pn.launch > s - 364] = np.nan
        price_wk = pn.window(pn.CP, s, min(e, pn.P.shape[1])) / max(min(e, pn.P.shape[1]) - s, 1)
        wk_day = pn.days[min(s, len(pn.days) - 1)]
        frames.append(pd.DataFrame({
            "unique_id": pn.ids, "item_id": pn.item, "store_id": pn.store,
            "origin": pn.days[origin_idx], "k": k,
            "week_start": wk_day,
            "y": target,
            "last7": last7, "last28_wk": last28 / 4, "last56_wk": last56 / 8,
            "wk_std8": weeks.std(axis=1), "wk_max8": weeks.max(axis=1),
            "trend": last28 / 4 - last56 / 8,
            "same_wk_ly": ly, "zero_rate28": zero28,
            "price": price_wk, "price_vs_now": price_wk / price_now - 1,
            "price_rel_mean": price_wk / pn.P.mean(axis=1),
            "events": pn.cev[min(e, len(pn.cev) - 1)] - pn.cev[min(s, len(pn.cev) - 1)],
            "snap_days": pn.csnap[min(e, len(pn.csnap) - 1)] - pn.csnap[min(s, len(pn.csnap) - 1)],
            "weekofyear": wk_day.isocalendar().week, "month": wk_day.month,
            "age_days": age,
        }))
    df = pd.concat(frames, ignore_index=True)
    # Need a full 8 weeks of history at the origin.
    return df[df["age_days"] >= 56]


FEATURES = ["item_id", "store_id", "k", "last7", "last28_wk", "last56_wk", "wk_std8", "wk_max8",
            "trend", "same_wk_ly", "zero_rate28", "price", "price_vs_now", "price_rel_mean",
            "events", "snap_days", "weekofyear", "month", "age_days"]


def to_X(df: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in ("item_id", "store_id"):
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def training_rows(pn: Panel, cutoff) -> pd.DataFrame:
    c = pn.idx(cutoff)
    rows = []
    for o in range(pn.idx(HISTORY_START), c, 7):
        ks = [k for k in range(1, WEEKS_AHEAD + 1) if o + 7 * k <= c]  # target fully observed
        if ks:
            rows.append(build_rows(pn, o, ks))
    return pd.concat(rows, ignore_index=True)


def prediction_rows(pn: Panel, cutoff) -> pd.DataFrame:
    """Origins at the cutoff and at each later week start inside the 28-day window."""
    c = pn.idx(cutoff)
    rows = []
    for j in range(WEEKS_AHEAD):
        ks = list(range(1, WEEKS_AHEAD - j + 1))
        rows.append(build_rows(pn, c + 7 * j, ks))
    df = pd.concat(rows, ignore_index=True)
    df["cutoff"] = pd.Timestamp(cutoff)
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cats = {c: sorted(train[c].unique()) for c in ("item_id", "store_id")}
    Xtr, Xte = to_X(train, cats), to_X(test, cats)
    out = test[["unique_id", "item_id", "store_id", "cutoff", "origin", "k", "week_start", "y",
                "price", "last28_wk"]].copy()

    mean_model = lgb.train({**BASE_PARAMS, "objective": "tweedie", "tweedie_variance_power": 1.2},
                           lgb.Dataset(Xtr, train["y"]), ROUNDS)
    out["mean"] = np.clip(mean_model.predict(Xte), 0, None)

    qcols = []
    for tau in config.QUANTILES:
        m = lgb.train({**BASE_PARAMS, "objective": "quantile", "alpha": tau},
                      lgb.Dataset(Xtr, train["y"]), ROUNDS)
        col = f"q{int(round(tau * 100)):02d}"
        out[col] = np.clip(m.predict(Xte), 0, None)
        qcols.append(col)
    # Enforce non-crossing quantiles.
    out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    return out


def main():
    panel = load_panel()
    pn = Panel(panel)
    preds = []
    for cutoff in cutoffs(panel):
        t0 = time.perf_counter()
        train = training_rows(pn, cutoff)
        test = prediction_rows(pn, cutoff)
        preds.append(fit_predict(train, test))
        print(f"cutoff {cutoff.date()}: {len(train):,} train rows, {time.perf_counter() - t0:.0f}s", flush=True)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(config.RESULTS_DIR / "weekly_forecasts.parquet", index=False)

    one = preds[preds["k"] == 1]
    qcols = [c for c in preds.columns if c.startswith("q") and c[1:].isdigit()]
    cover = {c: float((one["y"] <= one[c]).mean()) for c in qcols}
    print("coverage (share of actuals at or below each quantile, 1 week ahead):")
    print({k: round(v, 3) for k, v in cover.items()})
    pd.Series(cover, name="coverage").to_csv(config.RESULTS_DIR / "quantile_coverage.csv")


if __name__ == "__main__":
    main()
