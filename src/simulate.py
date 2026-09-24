"""Replay the backtest weeks and compare replenishment policies in dollars.

Each week, per store, a policy sets order-up-to levels for every item, the week's
real demand is served from stock, unmet demand is lost and leftovers carry over.

Two operating modes
  lead time 0 : the order placed at the start of a week arrives immediately
                (12 simulated weeks: 3 windows x 4 weeks, 1-week-ahead forecasts).
  lead time 1 : the order placed at the start of week t arrives at the start of
                week t+1, so it is sized with 2-week-ahead forecasts and a projection
                of the stock that will be left after week t. Each 28-day window is
                warm-started by its own policy in week 1, then weeks 2-4 are costed
                (9 simulated weeks). Deliveries that would overflow capacity are
                trimmed proportionally.

Policies
  Rule of thumb  : cover = recent 4-week average weekly sales x factor
  ML point       : cover = LightGBM weekly mean forecast x factor
  ML newsvendor  : critical-ratio quantile of the LightGBM quantile forecast
  ML + LP        : quantile scenarios fed to the capacity-constrained LP (optimize.py)
The two rule-based policies get their cover factor tuned on the same weeks (best case
for them), so the reported savings of the forecast-driven policies are conservative.
When stock would exceed capacity, non-LP policies shrink top-ups proportionally.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.optimize import optimal_levels, quantile_at, scale_to_capacity

LEVELS = np.array(config.QUANTILES)
QCOLS = [f"q{int(round(t * 100)):02d}" for t in config.QUANTILES]
FACTORS = np.round(np.arange(0.8, 2.61, 0.1), 2)
CAPACITY_GRID = [1.0, 1.15, 1.3, 1.5, np.inf]
POLICIES = ["Rule of thumb", "ML point", "ML newsvendor", "ML + LP"]
HOLDING_GRID = [0.01, 0.02, 0.05]
STOCKOUT_GRID = [0.15, 0.30, 0.50]


@dataclass(frozen=True)
class Costs:
    holding: float = config.HOLDING_COST_FRAC
    stockout: float = config.STOCKOUT_COST_FRAC

    @property
    def critical_ratio(self) -> float:
        return self.stockout / (self.stockout + self.holding)


def decide(rows: pd.DataFrame, policy: str, factor: float, floor: np.ndarray,
           capacity: float, costs: Costs) -> np.ndarray:
    """Order-up-to levels for one store-week, given stock that will already be there."""
    price = rows["price"].to_numpy()
    if policy == "ML + LP":
        return optimal_levels(rows[QCOLS].to_numpy(), costs.holding * price,
                              costs.stockout * price, floor, capacity)
    if policy == "Rule of thumb":
        target = rows["last28_wk"].to_numpy() * factor
    elif policy == "ML point":
        target = rows["mean"].to_numpy() * factor
    else:
        target = quantile_at(rows[QCOLS].to_numpy(), LEVELS, costs.critical_ratio)
    return scale_to_capacity(target, floor, capacity)


def expected_demand(rows: pd.DataFrame, policy: str) -> np.ndarray:
    """Each policy projects stock with its own view of the coming week."""
    return rows["last28_wk"].to_numpy() if policy == "Rule of thumb" else rows["mean"].to_numpy()


def record(store, week_start, policy, demand, level, price, costs):
    sold = np.minimum(level, demand)
    left, lost = level - sold, demand - sold
    return left, {"store_id": store, "week_start": week_start, "policy": policy,
                  "demand": demand.sum(), "sold": sold.sum(), "lost_units": lost.sum(),
                  "ending_stock": left.sum(),
                  "holding_cost": (costs.holding * price * left).sum(),
                  "stockout_cost": (costs.stockout * price * lost).sum(),
                  "revenue": (price * sold).sum()}


def store_capacity(sw: pd.DataFrame, cap_mult: float) -> float:
    first = sw[sw["week_start"] == sw["week_start"].min()]
    return cap_mult * first["last28_wk"].sum()  # store's recent average weekly unit sales


def run_policy(fc: pd.DataFrame, policy: str, cap_mult: float, factor: float = 1.0,
               costs: Costs = Costs()) -> pd.DataFrame:
    """Lead time 0, re-forecasting at the start of every week (1-week-ahead forecasts)."""
    return run_weeks(fc[fc["k"] == 1], policy, cap_mult, factor, costs)


def run_weeks(weeks: pd.DataFrame, policy: str, cap_mult: float, factor: float = 1.0,
              costs: Costs = Costs()) -> pd.DataFrame:
    """Lead time 0 over pre-selected rows: exactly one forecast row per item-store-week."""
    out = []
    for store, sw in weeks.groupby("store_id"):
        capacity = store_capacity(sw, cap_mult)
        on_hand: dict[str, float] = {}
        for week_start, w in sw.groupby("week_start"):
            oh = w["unique_id"].map(on_hand).fillna(0).to_numpy()
            level = decide(w, policy, factor, oh, capacity, costs)
            left, row = record(store, week_start, policy, w["y"].to_numpy(), level,
                               w["price"].to_numpy(), costs)
            on_hand.update(zip(w["unique_id"], left))
            out.append(row)
    return pd.DataFrame(out)


def run_policy_lead(fc: pd.DataFrame, policy: str, cap_mult: float, factor: float = 1.0,
                    costs: Costs = Costs()) -> pd.DataFrame:
    """Lead time 1 week."""
    out = []
    for (store, cutoff), sw in fc.groupby(["store_id", "cutoff"]):
        capacity = store_capacity(sw[sw["k"] == 1], cap_mult)
        origins = sorted(sw["origin"].unique())
        one = {o: sw[(sw["origin"] == o) & (sw["k"] == 1)].set_index("unique_id") for o in origins}
        two = {o: sw[(sw["origin"] == o) & (sw["k"] == 2)].set_index("unique_id") for o in origins}

        # Week 1 warm start: stock placed by this policy one week earlier.
        w0 = one[origins[0]]
        stock = pd.Series(decide(w0, policy, factor, np.zeros(len(w0)), capacity, costs), index=w0.index)

        for j, o in enumerate(origins):
            week = one[o]
            s_now = stock.reindex(week.index).fillna(0).to_numpy()
            order = None
            if o in two and len(two[o]):
                nxt = two[o]
                ids = nxt.index
                exp = pd.Series(expected_demand(week, policy), index=week.index)
                projected = np.maximum(stock.reindex(ids).fillna(0) - exp.reindex(ids).fillna(0), 0).to_numpy()
                level = decide(nxt, policy, factor, projected, capacity, costs)
                order = pd.Series(level - projected, index=ids)

            left, row = record(store, week["week_start"].iloc[0], policy, week["y"].to_numpy(), s_now,
                               week["price"].to_numpy(), costs)
            if j > 0:  # week 1 is warm-up only
                out.append(row)
            if order is None:
                break
            left = pd.Series(left, index=week.index).reindex(order.index).fillna(0)
            room = max(capacity - left.sum(), 0)
            if order.sum() > room:  # delivery would overflow the backroom
                order *= room / order.sum()
            stock = left + order
    return pd.DataFrame(out)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    s = sim.groupby("policy")[["demand", "sold", "lost_units", "ending_stock",
                              "holding_cost", "stockout_cost", "revenue"]].sum()
    s["total_cost"] = s["holding_cost"] + s["stockout_cost"]
    s["fill_rate"] = s["sold"] / s["demand"]
    s["avg_weekly_stock"] = sim.groupby("policy")["ending_stock"].sum() / sim["week_start"].nunique()
    return s


def compare(fc, runner, cap_mult, costs=Costs()) -> tuple[pd.DataFrame, pd.DataFrame]:
    sims = []
    for policy in POLICIES:
        if policy in ("Rule of thumb", "ML point"):
            best = min(FACTORS, key=lambda f: summarize(runner(fc, policy, cap_mult, f, costs))["total_cost"].iloc[0])
            sim = runner(fc, policy, cap_mult, best, costs)
            sim["factor"] = best
        else:
            sim = runner(fc, policy, cap_mult, 1.0, costs)
        sims.append(sim)
    sim = pd.concat(sims)
    s = summarize(sim).loc[POLICIES]
    s["cost_saving_vs_rule"] = 1 - s["total_cost"] / s.loc["Rule of thumb", "total_cost"]
    s["revenue_vs_rule"] = s["revenue"] / s.loc["Rule of thumb", "revenue"] - 1
    return s, sim


MODES = {"lead time 0": run_policy, "lead time 1 week": run_policy_lead}


def main():
    fc = pd.read_parquet(config.RESULTS_DIR / "weekly_forecasts.parquet")
    fc = fc.sort_values(["store_id", "origin", "k", "unique_id"]).reset_index(drop=True)
    show = ["total_cost", "fill_rate", "revenue_vs_rule", "cost_saving_vs_rule"]

    rows, detail = [], []
    for mode, runner in MODES.items():
        for cap in CAPACITY_GRID:
            s, sim = compare(fc, runner, cap)
            s["capacity_multiplier"], s["mode"] = cap, mode
            sim["capacity_multiplier"], sim["mode"] = cap, mode
            rows.append(s.reset_index())
            detail.append(sim)
            print(f"\n[{mode}] capacity = {cap} x weekly sales")
            print(s[show].round(3).to_string(), flush=True)
    pd.concat(rows).to_csv(config.RESULTS_DIR / "inventory_summary.csv", index=False)
    pd.concat(detail).to_csv(config.RESULTS_DIR / "inventory_by_week.csv", index=False)

    sens = []
    for mode, runner in MODES.items():
        for hc in HOLDING_GRID:
            for sc in STOCKOUT_GRID:
                s, _ = compare(fc, runner, 1.15, Costs(hc, sc))
                s = s.reset_index()[["policy", "total_cost", "fill_rate", "cost_saving_vs_rule", "revenue_vs_rule"]]
                s["mode"], s["holding_frac"], s["stockout_frac"] = mode, hc, sc
                sens.append(s)
                lp = s.set_index("policy").loc["ML + LP"]
                print(f"[{mode}] h={hc:.2f} p={sc:.2f}: LP saving {lp.cost_saving_vs_rule:+.1%}", flush=True)
    pd.concat(sens).to_csv(config.RESULTS_DIR / "cost_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
