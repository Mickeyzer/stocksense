# StockSense: demand forecasting and capacity-constrained replenishment

StockSense forecasts daily and weekly demand for **3,292 Walmart item-store series** and turns those forecasts into
stocking decisions. The decisions come from a linear program that decides how to split limited store capacity
across items. Every number below comes from a rolling-origin backtest on real data. The last test window is the
official **M5 competition evaluation period** (22 May – 19 June 2016).

| | Result |
|---|---|
| Daily forecast accuracy (WRMSSE, lower is better) | **0.868**, which is **22.9% better** than seasonal naive (1.126) |
| Daily WAPE | **62.7%**, compared with 77.3% for seasonal naive |
| Weekly 90% prediction interval coverage | 86.3% (P95 quantile covers 93.3% of actuals) |
| Inventory cost vs a tuned rule of thumb | **21.3% lower** at 1.15x capacity, and 20–24% lower at every capacity level tested |
| Revenue captured with the same shelf capacity | **+4.3%** ($1.58M vs $1.51M over 12 weeks, 4 stores) |

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

### 1. Daily forecasting benchmark (`src/backtest.py`)
There are 3 rolling origins with a 28-day horizon, and a model is refit at each origin. The models compared are:

- **Baselines:** seasonal naive and a 28-day moving average.
- **Statistical:** AutoETS (weekly seasonality), using `statsforecast` across 12 cores.
- **Intermittent-demand:** Croston (optimized) and IMAPA.
- **Global LightGBM:** one model for all series, with a Tweedie loss for zero-inflated counts. Every demand
  feature is lagged by at least 28 days, so a single model forecasts the whole horizon without recursion and
  without leakage. Features are lags, rolling mean/std, recent zero-sales share, price ratios, calendar,
  SNAP and events.
- **Ensemble:** an equal-weight average of LightGBM and AutoETS. The weights are fixed, not tuned on test data.

| Model | WAPE | RMSSE | WRMSSE | Bias | Fit + predict (3 windows) |
|---|---|---|---|---|---|
| **Ensemble (LightGBM + ETS)** | **62.7%** | **0.789** | **0.868** | -2.4% | 610 s |
| LightGBM | 64.2% | 0.802 | 0.897 | -4.8% | 258 s |
| AutoETS | 65.4% | 0.805 | 0.905 | +0.1% | 352 s |
| Moving average (28) | 65.8% | 0.810 | 0.923 | -2.1% | 5 s |
| Croston | 69.1% | 0.829 | 0.928 | +5.3% | 8 s |
| IMAPA | 66.6% | 0.812 | 0.937 | +3.9% | 9 s |
| Seasonal naive | 77.3% | 1.041 | 1.126 | -0.1% | 6 s |

WRMSSE is RMSSE weighted by each series' dollar sales, as in M5 (computed at the item-store level). The ensemble
wins in all three windows. LightGBM beats AutoETS while training in about 27% less time. The strongest features
were the 56- and 28-day rolling means and the item identity.

### 2. Weekly probabilistic forecasts (`src/weekly.py`)
Stock is replenished weekly, so a separate LightGBM model predicts the **sum** of next week's demand directly.
Quantiles of a sum can't be built by adding daily quantiles. The output is a Tweedie mean model plus
**10 quantile models** (P5 to P95), with the quantiles sorted so they never cross. Training uses
~1.2M (series, origin, weeks-ahead) rows per window.

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
It is solved with CVXPY and HiGHS, about 16k variables per solve. Without the capacity constraint, the LP gives
the textbook newsvendor critical-ratio quantile, and a unit test checks this.

### 4. Policy simulation (`src/simulate.py`)
The simulation replays 12 real weeks across 4 stores (3,292 item-stores). Leftover stock carries over, and unmet
demand is lost. Four policies are compared:

| Policy (capacity = 1.15x avg weekly sales) | Total cost | Fill rate | Revenue | Saving vs rule |
|---|---|---|---|---|
| Rule of thumb (recent avg x tuned cover factor) | $79.3k | 86.3% | $1.51M | – |
| ML point forecast x tuned cover factor | $75.1k | 87.3% | $1.52M | 5.3% |
| ML newsvendor quantile, scaled to fit capacity | $93.5k | 83.5% | $1.47M | -17.9% |
| **ML quantiles + LP optimizer** | **$62.4k** | 84.9% | **$1.58M** | **21.3%** |

Cost saving by capacity level:

| Capacity | 1.0x | 1.15x | 1.3x | 1.5x | unlimited |
|---|---|---|---|---|---|
| ML point | 4.2% | 5.3% | 7.1% | 10.7% | 15.0% |
| ML newsvendor | -18.5% | -17.9% | -8.0% | 10.6% | 24.8% |
| **ML + LP** | **23.9%** | **21.3%** | **19.9%** | **21.3%** | **24.4%** |

The main finding: **a good probabilistic forecast is not enough when capacity is tight.** If every item gets its
ideal newsvendor quantile and the orders are then shrunk proportionally to fit, the result is worse than the rule
of thumb. That approach fills space with low-margin, high-variance items. The LP gives the space to the items
where it's worth most in dollars. With unlimited capacity the LP and the newsvendor give the same result, as
the theory predicts. The two rule-based policies had their cover factor tuned on the same weeks (their best case),
so the savings above are conservative.

## Dashboard

```
streamlit run app.py
```

The dashboard has four tabs:
- **Forecast accuracy:** the model table, an accuracy-vs-compute chart, feature importance and quantile calibration.
- **Item explorer:** daily forecasts per model and a weekly fan chart for any item and store.
- **Replenishment policies:** cost breakdowns and the capacity sensitivity analysis.
- **Order planner:** re-runs the LP live for any store and week with your own capacity and cost settings, and
  exports the order plan as CSV.

## Reproduce

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python run_all.py        # downloads M5 (~50 MB), about 25 minutes on 12 cores
pytest -q                # metric and optimizer unit tests
```

## Limitations
- Orders arrive at the start of the week (zero lead time), and each week is treated as a single bucket, so
  stock-outs within a week are not modelled.
- Stock levels are continuous (not rounded to whole units), and capacity is counted in units because M5 has
  no item volumes.
- The cost parameters are assumptions. The dashboard lets you change them.

## Structure
```
src/config.py     scope, horizon, cost and capacity settings
src/data.py       raw M5 -> tidy daily panel
src/features.py   LightGBM daily features
src/backtest.py   daily model benchmark
src/metrics.py    WAPE, RMSSE, WRMSSE, bias, pinball
src/weekly.py     weekly mean + quantile models
src/optimize.py   CVXPY replenishment LP
src/simulate.py   policy simulation
app.py            Streamlit dashboard
tests/            unit tests
```
