"""Rolling-origin backtest of daily item-store forecasts (28-day horizon, 3 windows).

Compares statistical baselines, intermittent-demand methods and a global LightGBM
model on accuracy (WAPE, RMSSE, dollar-weighted RMSSE) and on compute time.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, CrostonOptimized, IMAPA, SeasonalNaive, WindowAverage

from src import config
from src.data import cutoffs, load_panel
from src.features import add_features, feature_columns
from src.metrics import evaluate

LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,  # many zeros plus a long right tail
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "seed": config.SEED,
    "verbose": -1,
    "num_threads": 0,
}
LGB_ROUNDS = 800
TUNED = config.RESULTS_DIR / "lgb_tuned.json"


def tuned_settings() -> tuple[dict, int, bool, float]:
    """Settings chosen by src/tune.py on the validation window, or the defaults."""
    if not TUNED.exists():
        return LGB_PARAMS, LGB_ROUNDS, False, 0.5
    t = json.loads(TUNED.read_text())
    return {**LGB_PARAMS, **t["params_change"]}, t["rounds"], t["extra_features"], t["ensemble_weight_lgb"]

STAT_MODELS = {
    "SeasonalNaive": SeasonalNaive(season_length=config.SEASON),
    "MovingAvg28": WindowAverage(window_size=28),
    "AutoETS": AutoETS(season_length=config.SEASON),
    "Croston": CrostonOptimized(),
    "IMAPA": IMAPA(),
}


def run_statistical(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = panel[["unique_id", "ds", "y"]]
    fcs, times = [], {}
    for name, model in STAT_MODELS.items():
        sf = StatsForecast(models=[model], freq="D", n_jobs=-1)
        t0 = time.perf_counter()
        cv = sf.cross_validation(df=df, h=config.HORIZON, n_windows=config.N_WINDOWS,
                                 step_size=config.HORIZON)
        times[name] = time.perf_counter() - t0
        col = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")][0]
        cv = cv.rename(columns={col: name})
        fcs.append(cv.set_index(["unique_id", "ds", "cutoff"])[[name] if fcs else ["y", name]])
        print(f"  {name}: {times[name]:.1f}s", flush=True)
    return pd.concat(fcs, axis=1).reset_index(), times


def run_lightgbm(feat: pd.DataFrame, params: dict, rounds: int) -> tuple[pd.DataFrame, float, lgb.Booster]:
    cols = feature_columns(feat)
    first_usable = feat["ds"].min() + pd.Timedelta(days=56)
    out, total, booster = [], 0.0, None
    for cutoff in cutoffs(feat):
        train = feat[(feat["ds"] <= cutoff) & (feat["ds"] >= first_usable)]
        test = feat[(feat["ds"] > cutoff) & (feat["ds"] <= cutoff + pd.Timedelta(days=config.HORIZON))]
        t0 = time.perf_counter()
        booster = lgb.train(params, lgb.Dataset(train[cols], train["y"]), rounds)
        pred = np.clip(booster.predict(test[cols]), 0, None)
        total += time.perf_counter() - t0
        out.append(pd.DataFrame({"unique_id": test["unique_id"].values, "ds": test["ds"].values,
                                 "cutoff": cutoff, "LightGBM": pred}))
        print(f"  LightGBM cutoff {cutoff.date()}: {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.concat(out), total, booster


def main(lgb_only: bool = False):
    """lgb_only=True reuses saved statistical forecasts and only refits LightGBM."""
    config.RESULTS_DIR.mkdir(exist_ok=True)
    panel = load_panel()
    print(f"{panel['unique_id'].nunique()} series, {len(panel):,} rows")

    if lgb_only:
        saved = pd.read_parquet(config.RESULTS_DIR / "daily_forecasts.parquet")
        stat_fc = saved[["unique_id", "ds", "cutoff", "y"] + list(STAT_MODELS)]
        times = pd.read_csv(config.RESULTS_DIR / "fit_times.csv", index_col=0)["seconds"].to_dict()
    else:
        print("Statistical models")
        stat_fc, times = run_statistical(panel)

    params, rounds, extra, w = tuned_settings()
    print(f"LightGBM ({rounds} rounds, extra features={extra}, ensemble weight={w})")
    feat = add_features(panel, extra=extra)
    lgb_fc, times["LightGBM"], booster = run_lightgbm(feat, params, rounds)

    fc = stat_fc.merge(lgb_fc, on=["unique_id", "ds", "cutoff"], how="inner")
    # Blend of the ML and statistical models; the weight was chosen on the validation window.
    fc["Ensemble"] = w * fc["LightGBM"] + (1 - w) * fc["AutoETS"]
    times["Ensemble"] = times["LightGBM"] + times["AutoETS"]
    fc.to_parquet(config.RESULTS_DIR / "daily_forecasts.parquet", index=False)
    pd.Series(times, name="seconds").to_csv(config.RESULTS_DIR / "fit_times.csv")

    imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    (imp / imp.sum()).sort_values(ascending=False).to_csv(config.RESULTS_DIR / "feature_importance.csv",
                                                          header=["gain_share"])
    score(fc, panel, times)


def score(fc: pd.DataFrame, panel: pd.DataFrame, times: dict):
    models = list(STAT_MODELS) + ["LightGBM", "Ensemble"]
    scores = []
    for cutoff, window in fc.groupby("cutoff"):
        s = evaluate(window, models, panel[panel["ds"] <= cutoff])
        s["cutoff"] = cutoff
        scores.append(s)
    scores = pd.concat(scores)
    scores.to_csv(config.RESULTS_DIR / "daily_scores_by_window.csv", index=False)

    summary = scores.groupby("model")[["WAPE", "RMSSE", "WRMSSE", "Bias"]].mean()
    summary["fit_predict_seconds"] = pd.Series(times)
    summary = summary.sort_values("WRMSSE")
    summary.to_csv(config.RESULTS_DIR / "daily_scores.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main(lgb_only="--lgb-only" in sys.argv)
