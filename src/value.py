"""What is forecast accuracy worth in dollars?

Every forecasting model is pushed through the same replenishment simulation, so its
accuracy can be read as inventory cost. To isolate accuracy, all models share one
uncertainty model: next week's demand is negative binomial with the model's own mean
and a per-series dispersion estimated from the eight weeks before the forecast origin.
The direct weekly LightGBM also runs with its own learned quantiles.

Setting: forecasts are made once at the start of each 28-day window and used for the
four weekly orders in that window (so daily models and the weekly model see the same
information), zero lead time, capacity 1.15x average weekly sales, default costs.
A perfect-foresight run (true demand, same capacity) gives the ceiling on savings.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.data import load_panel
from src.metrics import wape
from src.simulate import FACTORS, QCOLS, run_weeks, summarize
from src.weekly import Panel, build_rows

CAPACITY = 1.15
DAILY_MODELS = ["SeasonalNaive", "MovingAvg28", "Croston", "IMAPA", "AutoETS", "LightGBM", "Ensemble"]
LEVELS = np.array(config.QUANTILES)


def weekly_from_daily() -> pd.DataFrame:
    """Sum each daily model's 28-day forecast into four weekly buckets."""
    fc = pd.read_parquet(config.RESULTS_DIR / "daily_forecasts.parquet")
    fc["k"] = ((fc["ds"] - fc["cutoff"]).dt.days - 1) // 7 + 1
    return fc.groupby(["unique_id", "cutoff", "k"])[DAILY_MODELS].sum().reset_index()


def dispersion(panel: pd.DataFrame, cutoffs) -> pd.DataFrame:
    """Negative-binomial overdispersion per series from its last 8 weekly totals."""
    pn = Panel(panel)
    out = []
    for c in cutoffs:
        r = build_rows(pn, pn.idx(c), [1])
        m, s = r["last56_wk"].to_numpy(), r["wk_std8"].to_numpy()
        phi = np.where(m > 0, np.maximum(s ** 2 - m, 0) / np.maximum(m, 1e-9) ** 2, 0)
        out.append(pd.DataFrame({"unique_id": r["unique_id"], "cutoff": pd.Timestamp(c),
                                 "phi": np.clip(phi, 0, 5)}))
    return pd.concat(out)


def nb_quantiles(mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Scenario matrix (rows x 10) of negative-binomial (Poisson if phi=0) quantiles."""
    mu = np.maximum(mu, 1e-6)
    q = np.empty((len(mu), len(LEVELS)))
    pois = phi < 1e-6
    for j, tau in enumerate(LEVELS):
        q[pois, j] = stats.poisson.ppf(tau, mu[pois])
        n = 1 / phi[~pois]
        q[~pois, j] = stats.nbinom.ppf(tau, n, n / (n + mu[~pois]))
    return q


def evaluate_model(weeks: pd.DataFrame, name: str, mean: np.ndarray, Q: np.ndarray) -> dict:
    w = weeks.copy()
    w["mean"] = mean
    w[QCOLS] = Q
    point_cost = {f: summarize(run_weeks(w, "ML point", CAPACITY, f))["total_cost"].iloc[0] for f in FACTORS}
    best_f = min(point_cost, key=point_cost.get)
    lp = summarize(run_weeks(w, "ML + LP", CAPACITY)).iloc[0]
    return {"model": name, "weekly_WAPE": wape(w["y"].to_numpy(), mean),
            "cost_point_policy": point_cost[best_f], "point_factor": best_f,
            "cost_LP_policy": lp["total_cost"], "fill_rate_LP": lp["fill_rate"], "revenue_LP": lp["revenue"]}


def main():
    wf = pd.read_parquet(config.RESULTS_DIR / "weekly_forecasts.parquet")
    weeks = wf[wf["origin"] == wf["cutoff"]].copy()
    daily = weekly_from_daily()
    weeks = weeks.merge(daily, on=["unique_id", "cutoff", "k"], how="inner")
    panel = load_panel()
    weeks = weeks.merge(dispersion(panel, sorted(weeks["cutoff"].unique())), on=["unique_id", "cutoff"])
    weeks = weeks.sort_values(["store_id", "week_start", "unique_id"]).reset_index(drop=True)
    native_q = weeks[QCOLS].to_numpy()
    phi = weeks["phi"].to_numpy()
    print(f"{len(weeks):,} item-store-weeks, {weeks['week_start'].nunique()} weeks", flush=True)

    rows = []
    for m in DAILY_MODELS + ["WeeklyLightGBM"]:
        mu = weeks["mean"].to_numpy() if m == "WeeklyLightGBM" else weeks[m].to_numpy()
        rows.append(evaluate_model(weeks, m, mu, nb_quantiles(mu, phi)))
        print(f"  {m}: {rows[-1]}", flush=True)
    rows.append(evaluate_model(weeks, "WeeklyLightGBM (learned quantiles)", weeks["mean"].to_numpy(), native_q))
    rows[-1]["cost_point_policy"] = np.nan  # same point forecast as the row above
    print(f"  learned quantiles: {rows[-1]}", flush=True)

    rule = summarize(run_weeks(weeks, "Rule of thumb", CAPACITY,
                               min(FACTORS, key=lambda f: summarize(run_weeks(weeks, "Rule of thumb", CAPACITY, f))
                                   ["total_cost"].iloc[0])))["total_cost"].iloc[0]
    y = weeks["y"].to_numpy()
    oracle = summarize(run_weeks(weeks.assign(**{c: y for c in QCOLS}), "ML + LP", CAPACITY)).iloc[0]

    res = pd.DataFrame(rows)
    res["rule_of_thumb_cost"] = rule
    # Near zero: knowing demand exactly removes almost all cost. Reported for context only,
    # since demand randomness makes it unreachable for any forecast.
    res["perfect_foresight_cost"] = oracle["total_cost"]
    res.to_csv(config.RESULTS_DIR / "value_of_accuracy.csv", index=False)
    print(f"\nrule of thumb ${rule:,.0f}   perfect foresight ${oracle['total_cost']:,.0f}")
    print(res[["model", "weekly_WAPE", "cost_point_policy", "cost_LP_policy"]].round(3).to_string())
    print(decompose(res).round(3).to_string())


def decompose(res: pd.DataFrame) -> pd.DataFrame:
    """Split the saving from the rule of thumb to the full system into its sources."""
    r = res.set_index("model")
    steps = [("Rule of thumb", r["rule_of_thumb_cost"].iloc[0]),
             ("Better forecast (simple policy)", r.loc["WeeklyLightGBM", "cost_point_policy"]),
             ("LP optimizer", r.loc["WeeklyLightGBM", "cost_LP_policy"]),
             ("Learned uncertainty", r.loc["WeeklyLightGBM (learned quantiles)", "cost_LP_policy"])]
    d = pd.DataFrame(steps, columns=["step", "cost"])
    d["saving"] = -d["cost"].diff().fillna(0)
    total = d["cost"].iloc[0] - d["cost"].iloc[-1]
    d["share_of_total_saving"] = d["saving"] / total
    d.to_csv(config.RESULTS_DIR / "value_decomposition.csv", index=False)
    return d


if __name__ == "__main__":
    main()
