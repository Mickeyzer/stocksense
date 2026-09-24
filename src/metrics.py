"""Forecast accuracy metrics. All functions take long-format frames keyed by unique_id."""
import numpy as np
import pandas as pd


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error over total demand."""
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed over(+)/under(-) forecast as a share of total demand."""
    return float((yhat - y).sum() / y.sum())


def rmsse_scale(train: pd.DataFrame) -> pd.Series:
    """M5 scale: mean squared one-step naive error, from each series' first sale onward."""
    t = train.sort_values(["unique_id", "ds"])
    started = t.groupby("unique_id")["y"].transform(lambda s: s.gt(0).cummax())
    t = t[started]
    diff2 = t.groupby("unique_id")["y"].diff() ** 2
    return diff2.groupby(t["unique_id"]).mean().rename("scale")


def dollar_weights(train: pd.DataFrame, days: int = 28) -> pd.Series:
    """M5 weights: each series' share of dollar sales over the last `days` of training."""
    last = train["ds"].max()
    recent = train[train["ds"] > last - pd.Timedelta(days=days)]
    dollars = (recent["y"] * recent["sell_price"]).groupby(recent["unique_id"]).sum()
    return (dollars / dollars.sum()).rename("weight")


def rmsse_table(fc: pd.DataFrame, model: str, scale: pd.Series) -> pd.Series:
    mse = ((fc["y"] - fc[model]) ** 2).groupby(fc["unique_id"]).mean()
    s = scale.reindex(mse.index)
    return np.sqrt(mse / s.where(s > 0)).rename(model)


def evaluate(fc: pd.DataFrame, models: list[str], train: pd.DataFrame) -> pd.DataFrame:
    """Score every model on one backtest window."""
    scale = rmsse_scale(train)
    w = dollar_weights(train)
    rows = []
    for m in models:
        r = rmsse_table(fc, m, scale)
        ww = w.reindex(r.index).fillna(0)
        valid = r.notna()
        rows.append({
            "model": m,
            "WAPE": wape(fc["y"].values, fc[m].values),
            "RMSSE": float(r[valid].mean()),
            "WRMSSE": float((r[valid] * ww[valid]).sum() / ww[valid].sum()),
            "Bias": bias(fc["y"].values, fc[m].values),
        })
    return pd.DataFrame(rows)


def pinball(y: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))
