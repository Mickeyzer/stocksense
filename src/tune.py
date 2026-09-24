"""Select LightGBM settings and the ensemble weight on a validation window.

The validation window is the 28 days *before* the first backtest cutoff, so none of
the three test windows is seen during selection. Early stopping on that window also
fixes the number of boosting rounds used in the backtest.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from src import config
from src.backtest import LGB_PARAMS
from src.data import cutoffs, load_panel
from src.features import add_features, feature_columns
from src.metrics import evaluate

CANDIDATES = {
    "base": {},
    "tweedie_1.2": {"tweedie_variance_power": 1.2},
    "tweedie_1.3": {"tweedie_variance_power": 1.3},
    "poisson": {"objective": "poisson"},
    "leaves_255": {"num_leaves": 255, "min_data_in_leaf": 50},
    "leaves_63": {"num_leaves": 63, "min_data_in_leaf": 200},
    "feature_frac_0.5": {"feature_fraction": 0.5},
    "lr_0.03": {"learning_rate": 0.03},
}
MAX_ROUNDS = 2500


def split(feat, cutoff):
    first_usable = feat["ds"].min() + pd.Timedelta(days=56)
    train = feat[(feat["ds"] <= cutoff) & (feat["ds"] >= first_usable)]
    valid = feat[(feat["ds"] > cutoff) & (feat["ds"] <= cutoff + pd.Timedelta(days=config.HORIZON))]
    return train, valid


def fit_eval(feat, cutoff, params, panel):
    cols = feature_columns(feat)
    train, valid = split(feat, cutoff)
    t0 = time.perf_counter()
    booster = lgb.train(params, lgb.Dataset(train[cols], train["y"]), MAX_ROUNDS,
                        valid_sets=[lgb.Dataset(valid[cols], valid["y"])],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = np.clip(booster.predict(valid[cols], num_iteration=booster.best_iteration), 0, None)
    fc = pd.DataFrame({"unique_id": valid["unique_id"].values, "ds": valid["ds"].values,
                       "y": valid["y"].values, "LightGBM": pred})
    s = evaluate(fc, ["LightGBM"], panel[panel["ds"] <= cutoff]).iloc[0]
    return s, booster.best_iteration, time.perf_counter() - t0, fc


def main():
    panel = load_panel()
    val_cutoff = cutoffs(panel)[0] - pd.Timedelta(days=config.HORIZON)
    print("validation window after", val_cutoff.date(), flush=True)

    rows, best = [], None
    feats = {False: add_features(panel, extra=False)}
    for name, change in CANDIDATES.items():
        params = {**LGB_PARAMS, **change}
        s, rounds, secs, fc = fit_eval(feats[False], val_cutoff, params, panel)
        rows.append({"config": name, "extra_features": False, "rounds": rounds, "seconds": secs, **s.drop("model")})
        print(f"  {name}: WRMSSE {s.WRMSSE:.4f}, {rounds} rounds, {secs:.0f}s", flush=True)
        if best is None or s.WRMSSE < best[0]:
            best = (s.WRMSSE, name, False, rounds, fc)

    # Extra features, tried with the best configuration so far.
    feats[True] = add_features(panel, extra=True)
    name = best[1]
    s, rounds, secs, fc = fit_eval(feats[True], val_cutoff, {**LGB_PARAMS, **CANDIDATES[name]}, panel)
    rows.append({"config": name, "extra_features": True, "rounds": rounds, "seconds": secs, **s.drop("model")})
    print(f"  {name} + extra features: WRMSSE {s.WRMSSE:.4f}, {rounds} rounds", flush=True)
    if s.WRMSSE < best[0]:
        best = (s.WRMSSE, name, True, rounds, fc)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "tuning.csv", index=False)

    # Ensemble weight on the same validation window.
    df = panel.loc[panel["ds"] <= val_cutoff + pd.Timedelta(days=config.HORIZON), ["unique_id", "ds", "y"]]
    ets = StatsForecast(models=[AutoETS(season_length=config.SEASON)], freq="D", n_jobs=-1) \
        .cross_validation(df=df, h=config.HORIZON, n_windows=1)
    fc = best[4].merge(ets[["unique_id", "ds", "AutoETS"]], on=["unique_id", "ds"])
    weights = {}
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        fc[f"w{w}"] = w * fc["LightGBM"] + (1 - w) * fc["AutoETS"]
        weights[float(w)] = evaluate(fc, [f"w{w}"], panel[panel["ds"] <= val_cutoff]).iloc[0]["WRMSSE"]
    w_best = min(weights, key=weights.get)
    print("ensemble weight on LightGBM:", w_best, {k: round(v, 4) for k, v in weights.items()}, flush=True)

    chosen = {"config": best[1], "params_change": CANDIDATES[best[1]], "extra_features": best[2],
              "rounds": int(best[3]), "ensemble_weight_lgb": w_best,
              "validation_wrmsse": float(best[0]), "validation_cutoff": str(val_cutoff.date())}
    (config.RESULTS_DIR / "lgb_tuned.json").write_text(json.dumps(chosen, indent=2))
    print(chosen)


if __name__ == "__main__":
    main()
