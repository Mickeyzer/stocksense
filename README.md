# StockSense: demand forecasting and capacity-constrained replenishment

StockSense forecasts daily and weekly demand for **3,292 Walmart item-store series** and turns those forecasts into
stocking decisions. The decisions come from a linear program that decides how to split limited store capacity
across items. Every number below comes from a rolling-origin backtest on real data. The last test window is the
official **M5 competition evaluation period** (22 May – 19 June 2016). Model settings were chosen on a separate
validation window *before* all test windows.

| | Result |
|---|---|
| Daily forecast accuracy (WRMSSE, lower is better) | **0.866**, which is **23.1% better** than seasonal naive (1.126) |
| Daily WAPE | **62.9%**, compared with 77.3% for seasonal naive |
| Weekly 90% prediction interval coverage | 86.3% (P95 quantile covers 93.3% of actuals) |
| Inventory cost vs a tuned rule of thumb (1.15x capacity) | **21.3% lower** with zero lead time, **12.6% lower** with a 1-week lead time |
| Robustness | ML + LP had the lowest cost in **all 18 scenarios tested** (9 cost settings × 2 lead times) |
| Revenue captured with the same shelf capacity | **+3.2% to +4.4%** vs the rule of thumb in all 18 scenarios |
| Where the saving comes from | **48% from the optimizer**, 33% from the better forecast, 19% from learned uncertainty |

## Problem

A store must decide each week how many units of each of ~820 food items to stock. If it stocks too little, it loses
margin on sales it can't make. If it stocks too much, it pays holding and spoilage costs. Backroom space is limited,
so stocking more of one item means stocking less of another. That's two problems:

1. **Forecasting:** how much will each item sell? What matters is the whole range of likely outcomes, not just
   the average.
2. **Optimization:** given those ranges, item margins and a capacity limit, how should the space be split?

## Data

[M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Walmart). Scope: department FOODS_3
across the four California stores, from 2013 to 2016. That's 3.79M daily rows, 46% of them zero-sales days.
Prices, SNAP days and events are included as features.

## Approach

### 1. Daily forecasting benchmark (`src/backtest.py`, `src/tune.py`)
There are 3 rolling origins with a 28-day horizon, and a model is refit at each origin. The models compared are:

- **Baselines:** seasonal naive and a 28-day moving average.
- **Statistical:** AutoETS (weekly seasonality), using `statsforecast` across 12 cores.
- **Intermittent-demand:** Croston (optimized) and IMAPA.
- **Global LightGBM:** one model for all series, with a Tweedie loss for zero-inflated counts. Every demand
  feature is lagged by at least 28 days, so a single model forecasts the whole horizon without recursion and
  without leakage. Features are lags, rolling and exponentially weighted means, same-weekday averages, recent
  zero-sales share, price ratios and price changes, the item's demand across all four stores, calendar, SNAP
  and events.
- **Ensemble:** a weighted average of LightGBM and AutoETS.

**How the settings were chosen.** `src/tune.py` fits 8 LightGBM configurations (loss, tree size, learning rate,
feature sampling) plus an extended feature set on a **validation window that ends before the first test window**.
Early stopping on that window sets the number of boosting rounds. The same window chose the ensemble weight,
which came out at 50/50 (validation WRMSSE 0.826, against 0.858 for ETS alone and 0.863 for LightGBM alone). The
test windows were used once, for the final scores.

| Model | WAPE | RMSSE | WRMSSE | Bias | Fit + predict (3 windows) |
|---|---|---|---|---|---|
| **Ensemble (LightGBM + ETS)** | 62.9% | **0.789** | **0.866** | -1.9% | 475 s |
| LightGBM (tuned) | 64.4% | 0.802 | 0.888 | -3.8% | 123 s |
| AutoETS | 65.4% | 0.805 | 0.905 | +0.1% | 352 s |
| Moving average (28) | 65.8% | 0.810 | 0.923 | -2.1% | 5 s |
| Croston | 69.1% | 0.829 | 0.928 | +5.3% | 8 s |
| IMAPA | 66.6% | 0.812 | 0.937 | +3.9% | 9 s |
| Seasonal naive | 77.3% | 1.041 | 1.126 | -0.1% | 6 s |

WRMSSE is RMSSE weighted by each series' dollar sales, as in M5, computed at the item-store level only. **It is not
comparable to the M5 leaderboard**, which averages 12 levels of aggregation, most of them far easier to forecast.
Tuning improved LightGBM's WRMSSE from 0.897 to 0.888 and **halved its training time** (258s to 123s). LightGBM
now beats AutoETS by 1.8% while running 2.9× faster, and the ensemble wins in all three test windows. The most
important features were the exponentially weighted demand level, the 56-day rolling mean and the item identity.

### 2. Weekly probabilistic forecasts (`src/weekly.py`)
Stock is replenished weekly, so a separate LightGBM model predicts the **sum** of next week's demand directly.
Quantiles of a sum can't be built by adding daily quantiles. The output is a Tweedie mean model plus
**10 quantile models** (P5 to P95), with the quantiles sorted so they never cross. Forecasts are made 1–4 weeks
ahead, and training uses ~1.2M (series, origin, weeks-ahead) rows per window.

### 3. Capacity-constrained optimization (`src/optimize.py`)
The 10 quantiles are the midpoints of 10 equal-probability bins, so they work as 10 equally likely demand
scenarios. For each store and week, a linear program chooses order-up-to levels to minimize expected cost:

```
minimize   (1/K) * sum_i sum_k  h_i * over_ik + p_i * under_ik
subject to over_ik  >= y_i - d_ik,   under_ik >= d_ik - y_i,   over, under >= 0
           sum_i y_i <= capacity                 (store backroom limit)
           y_i >= on_hand_i                      (stock cannot be removed)
```

`h` is the holding cost (2% of price per unit per week) and `p` is the lost margin (30% of price per unmet unit).
Both are varied in the sensitivity analysis below. It is solved with CVXPY and HiGHS, about 16k variables per
solve. Without the capacity constraint, the LP gives the textbook newsvendor critical-ratio quantile, and a unit
test checks this.

### 4. Policy simulation (`src/simulate.py`)
The simulation replays real demand across 4 stores (3,292 item-stores). Leftover stock carries over, and unmet
demand is lost. It runs in two modes:

- **Zero lead time:** orders arrive at the start of the week they are placed. Covers 12 weeks.
- **1-week lead time:** the order placed at the start of week *t* arrives at week *t+1*. It is sized from
  **2-week-ahead** forecasts and a projection of the stock that will be left after week *t*. Deliveries that
  would overflow the backroom are trimmed. Each 28-day window is warm-started by the policy itself, and weeks
  2–4 are costed, for 9 weeks in total.

Four policies are compared:
- **Rule of thumb:** recent average sales × a tuned cover factor.
- **ML point:** the forecast mean × a tuned cover factor.
- **ML newsvendor:** each item's critical-ratio quantile, scaled down to fit capacity.
- **ML + LP:** the optimizer above.

The two rule-based policies have their cover factor tuned on the same weeks, which is their best case, so the
savings below are conservative.

**At capacity = 1.15x average weekly sales:**

| Policy | Cost saving, lead time 0 | Cost saving, lead time 1 week | Revenue vs rule (lead 0 / lead 1) |
|---|---|---|---|
| Rule of thumb (tuned) | – | – | – |
| ML point forecast (tuned) | 5.3% | 2.8% | +0.9% / +0.7% |
| ML newsvendor, scaled to fit capacity | -17.9% | -9.2% | -2.9% / -2.2% |
| **ML quantiles + LP optimizer** | **21.3%** | **12.6%** | **+4.3% / +3.8%** |

**ML + LP cost saving by capacity level:**

| Capacity | 1.0x | 1.15x | 1.3x | 1.5x | unlimited |
|---|---|---|---|---|---|
| Lead time 0 | 23.9% | 21.3% | 19.9% | 21.3% | 24.4% |
| Lead time 1 week | 14.9% | 12.6% | 12.0% | 13.9% | 17.0% |

**Sensitivity to the cost assumptions** (ML + LP saving vs rule, capacity 1.15x):

| Holding cost \ lost margin | 15% | 30% | 50% |
|---|---|---|---|
| 1% (lead 0 / lead 1) | 21.2% / 12.6% | 23.9% / 14.9% | 25.5% / 15.9% |
| 2% | 16.4% / 9.3% | **21.3% / 12.6%** | 23.4% / 14.4% |
| 5% | 9.4% / 4.3% | 14.4% / 7.8% | 18.6% / 10.7% |

**What the results show:**
- **The optimizer wins every time.** ML + LP had the lowest cost in all 18 scenarios. How large the saving is
  depends on the economics: it's biggest when lost sales are expensive compared with holding stock, which is
  when getting the stock mix right matters most.
- **Lead time roughly halves the benefit, but it stays clearly positive.** Ordering a week ahead adds
  uncertainty that every policy has to absorb.
- **A good probabilistic forecast is not enough when capacity is tight.** If every item gets its ideal
  newsvendor quantile and the orders are then shrunk proportionally to fit, the result is worse than the rule
  of thumb. That approach fills space with low-margin, high-variance items. The LP gives the space to the items
  where it's worth most in dollars. With unlimited capacity, the LP and the newsvendor give nearly the same
  result, as the theory predicts.

### 5. What is forecast accuracy worth? (`src/value.py`)
Every forecasting model is pushed through the same replenishment simulation, so its accuracy can be read as
dollars. Forecasts are made once at the start of each 28-day window and used for that window's four weekly
orders, so the daily models and the weekly model see the same information. To isolate accuracy, all models share
**one uncertainty model**: next week's demand is negative binomial, with the model's own mean and a per-series
spread estimated from the 8 weeks before the forecast. Capacity is 1.15x, lead time is zero, and costs are the
defaults. The simulation covers 12 weeks and 4 stores.

| Forecast | Weekly WAPE | Cost, simple policy (tuned cover) | Cost, LP optimizer |
|---|---|---|---|
| Seasonal naive | 39.3% | $99.1k | $88.6k |
| Moving average (28) | 36.9% | $91.4k | $80.3k |
| Croston | 39.8% | $94.5k | $82.6k |
| IMAPA | 37.8% | $91.5k | $81.4k |
| AutoETS | 38.6% | $93.9k | $84.1k |
| LightGBM (daily, summed) | 37.5% | $91.7k | $80.9k |
| Ensemble (daily, summed) | 35.5% | $85.7k | $75.7k |
| **Weekly LightGBM** | **34.7%** | **$84.8k** | $75.1k |
| **Weekly LightGBM + its own learned quantiles** | 34.7% | – | **$71.2k** |

The rule of thumb costs $91.4k. The $20.2k (22.1%) saving from the rule of thumb to the full system breaks down as:

| Step | Cost | Saving | Share |
|---|---|---|---|
| Rule of thumb | $91.4k | | |
| Best forecast, same simple policy | $84.8k | $6.6k | 33% |
| + LP optimizer | $75.1k | $9.7k | **48%** |
| + learned uncertainty instead of a fixed distribution | $71.2k | $3.9k | 19% |

**What this shows:**
- **Better decisions are worth more than a better model.** The optimizer running on the *simplest* forecast
  (moving average, $80.3k) beats the *best* forecast run with a simple policy ($84.8k). The optimizer cuts cost
  by 10–13% whichever forecast it's given.
- **Measure accuracy at the level decisions are made.** AutoETS beats the moving average on daily WRMSSE, but it's
  worse at the weekly level, where orders are placed, and it costs more. Summing daily forecasts loses to
  forecasting the weekly total directly.
- **Forecast accuracy does matter.** Across models, weekly error and cost are strongly correlated (r = 0.89 under
  the LP), and the most accurate forecasts give the lowest costs under both policies.
- **Learning the uncertainty matters too.** Swapping the fixed distribution for the model's own quantiles saves
  another $3.9k. The quantile models set a separate spread for each item-week from the same features as the
  forecast (prices, events, recent volatility), instead of relying on history alone.

## Dashboard

```
streamlit run app.py
```

The dashboard has four tabs:
- **Forecast accuracy:** the model table, an accuracy-vs-compute chart, feature importance and quantile calibration.
- **Value of accuracy:** a waterfall of where the saving comes from, and forecast error vs inventory cost for every model.
- **Item explorer:** daily forecasts per model and a weekly fan chart for any item and store.
- **Replenishment policies:** switch between lead times, see cost breakdowns, the capacity sensitivity analysis
  and a heatmap of cost assumptions.
- **Order planner:** re-runs the LP live for any store and week with your own capacity and cost settings, and
  exports the order plan as CSV.

## Reproduce

```
python -m venv .venv && .venv/Scripts/pip install -r requirements-pipeline.txt
python run_all.py            # downloads M5 (~50 MB), about 1 hour on 12 cores
python -m src.backtest --lgb-only   # refit only LightGBM, reusing saved statistical forecasts
pytest -q                    # metric and optimizer unit tests
```

## Limitations
- Each week is treated as a single bucket, so stock-outs within a week are not modelled. The lead time is fixed
  at 0 or 1 week, not random.
- Stock levels are continuous (not rounded to whole units), and capacity is counted in units because M5 has
  no item volumes.
- The cost parameters are assumptions. The sensitivity grid and the dashboard sliders show how the results change.

## Structure
```
src/config.py     scope, horizon, cost and capacity settings
src/data.py       raw M5 -> tidy daily panel
src/features.py   LightGBM daily features
src/tune.py       validation-window tuning (LightGBM settings, ensemble weight)
src/backtest.py   daily model benchmark
src/metrics.py    WAPE, RMSSE, WRMSSE, bias, pinball
src/weekly.py     weekly mean + quantile models
src/optimize.py   CVXPY replenishment LP
src/simulate.py   policy simulation (two lead-time modes, cost sensitivity)
src/value.py      value of forecast accuracy: every model through the same simulation
app.py            Streamlit dashboard
tests/            unit tests
```
