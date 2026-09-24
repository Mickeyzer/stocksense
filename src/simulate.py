"""Replay the 12 backtest weeks and compare replenishment policies in dollars.

Each week, per store, a policy sets order-up-to levels for every item, the week's
real demand is served from stock, unmet demand is lost and leftovers carry over.
Weekly review with orders arriving at the start of the week (zero lead time).

Policies
  Rule of thumb  : cover = recent 4-week average weekly sales x factor
  ML point       : cover = LightGBM weekly mean forecast x factor
  ML newsvendor  : critical-ratio quantile of the LightGBM quantile forecast
  ML + LP        : quantile scenarios fed to the capacity-constrained LP (optimize.py)
The two rule-based policies get their cover factor tuned on the same weeks (best case
for them), so the reported savings of the forecast-driven policies are conservative.
When stock would exceed capacity, non-LP policies shrink top-ups proportionally.
"""
import numpy as np
import pandas as pd

from src import config
from src.optimize import optimal_levels, quantile_at, scale_to_capacity

LEVELS = np.array(config.QUANTILES)
QCOLS = [f"q{int(round(t * 100)):02d}" for t in config.QUANTILES]
FACTORS = np.round(np.arange(0.8, 2.61, 0.1), 2)
CAPACITY_GRID = [1.0, 1.15, 1.3, 1.5, np.inf]


def targets(week: pd.DataFrame, policy: str, factor: float) -> np.ndarray:
    if policy == "Rule of thumb":
        return week["last28_wk"].to_numpy() * factor
    if policy == "ML point":
        return week["mean"].to_numpy() * factor
    cr = config.STOCKOUT_COST_FRAC / (config.STOCKOUT_COST_FRAC + config.HOLDING_COST_FRAC)
    return quantile_at(week[QCOLS].to_numpy(), LEVELS, cr)


def run_policy(weeks: pd.DataFrame, policy: str, cap_mult: float, factor: float = 1.0) -> pd.DataFrame:
    out = []
    for store, sw in weeks.groupby("store_id"):
        first = sw[sw["week_start"] == sw["week_start"].min()]
        capacity = cap_mult * first["last28_wk"].sum()  # store's recent weekly unit sales
        on_hand: dict[str, float] = {}
        for week_start, w in sw.groupby("week_start"):
            oh = w["unique_id"].map(on_hand).fillna(0).to_numpy()
            price = w["price"].to_numpy()
            h = config.HOLDING_COST_FRAC * price
            p = config.STOCKOUT_COST_FRAC * price
            if policy == "ML + LP":
                level = optimal_levels(w[QCOLS].to_numpy(), h, p, oh, capacity)
            else:
                level = scale_to_capacity(targets(w, policy, factor), oh, capacity)
            demand = w["y"].to_numpy()
            sold = np.minimum(level, demand)
            left = level - sold
            lost = demand - sold
            on_hand.update(zip(w["unique_id"], left))
            out.append({"store_id": store, "week_start": week_start, "policy": policy,
                        "demand": demand.sum(), "sold": sold.sum(), "lost_units": lost.sum(),
                        "ending_stock": left.sum(), "ordered": (level - oh).sum(),
                        "holding_cost": (h * left).sum(), "stockout_cost": (p * lost).sum(),
                        "revenue": (price * sold).sum()})
    return pd.DataFrame(out)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    s = sim.groupby("policy")[["demand", "sold", "lost_units", "ending_stock",
                              "holding_cost", "stockout_cost", "revenue"]].sum()
    s["total_cost"] = s["holding_cost"] + s["stockout_cost"]
    s["fill_rate"] = s["sold"] / s["demand"]
    s["avg_weekly_stock"] = sim.groupby("policy")["ending_stock"].sum() / sim["week_start"].nunique()
    return s


def best_factor(weeks, policy, cap_mult):
    costs = {f: summarize(run_policy(weeks, policy, cap_mult, f))["total_cost"].iloc[0] for f in FACTORS}
    return min(costs, key=costs.get)


def main():
    fc = pd.read_parquet(config.RESULTS_DIR / "weekly_forecasts.parquet")
    weeks = fc[fc["k"] == 1].sort_values(["store_id", "week_start", "unique_id"]).reset_index(drop=True)
    print(f"{weeks['week_start'].nunique()} weeks x {weeks['unique_id'].nunique()} item-stores")

    rows, detail = [], []
    for cap in CAPACITY_GRID:
        sims = []
        for policy in ["Rule of thumb", "ML point"]:
            f = best_factor(weeks, policy, cap)
            sim = run_policy(weeks, policy, cap, f)
            sim["factor"] = f
            sims.append(sim)
        sims += [run_policy(weeks, "ML newsvendor", cap), run_policy(weeks, "ML + LP", cap)]
        sim = pd.concat(sims)
        sim["capacity_multiplier"] = cap
        detail.append(sim)
        s = summarize(sim)
        base = s.loc["Rule of thumb", "total_cost"]
        s["cost_saving_vs_rule"] = 1 - s["total_cost"] / base
        s["capacity_multiplier"] = cap
        rows.append(s.reset_index())
        print(f"\ncapacity = {cap} x weekly sales")
        print(s[["total_cost", "holding_cost", "stockout_cost", "fill_rate", "avg_weekly_stock",
                 "cost_saving_vs_rule"]].round(3).to_string())

    pd.concat(rows).to_csv(config.RESULTS_DIR / "inventory_summary.csv", index=False)
    pd.concat(detail).to_csv(config.RESULTS_DIR / "inventory_by_week.csv", index=False)


if __name__ == "__main__":
    main()
