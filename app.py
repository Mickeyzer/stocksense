"""StockSense dashboard: forecast accuracy, item explorer and replenishment planning."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.optimize import optimal_levels, quantile_at, scale_to_capacity

R = config.RESULTS_DIR
QCOLS = [f"q{int(round(t * 100)):02d}" for t in config.QUANTILES]
LEVELS = np.array(config.QUANTILES)
MODEL_COLORS = {"Ensemble": "#1e3a8a", "LightGBM": "#2563eb", "AutoETS": "#f59e0b", "SeasonalNaive": "#9ca3af",
                "MovingAvg28": "#10b981", "Croston": "#a855f7", "IMAPA": "#ef4444"}

st.set_page_config(page_title="StockSense", layout="wide")


@st.cache_data
def load():
    return {
        "scores": pd.read_csv(R / "daily_scores.csv", index_col=0),
        "by_window": pd.read_csv(R / "daily_scores_by_window.csv", parse_dates=["cutoff"]),
        "importance": pd.read_csv(R / "feature_importance.csv", index_col=0),
        "daily": pd.read_parquet(R / "daily_forecasts.parquet"),
        "weekly": pd.read_parquet(R / "weekly_forecasts.parquet"),
        "history": pd.read_parquet(R / "history.parquet"),
        "inv": pd.read_csv(R / "inventory_summary.csv"),
        "inv_week": pd.read_csv(R / "inventory_by_week.csv", parse_dates=["week_start"]),
        "coverage": pd.read_csv(R / "quantile_coverage.csv", index_col=0),
        "sens": pd.read_csv(R / "cost_sensitivity.csv"),
        "value": pd.read_csv(R / "value_of_accuracy.csv"),
        "decomp": pd.read_csv(R / "value_decomposition.csv"),
    }


d = load()
scores, inv = d["scores"], d["inv"]

st.title("StockSense")
st.caption("Demand forecasting and capacity-constrained replenishment for 3,292 Walmart "
           "item-store series (M5, FOODS_3, California), backtested over 3 rolling 28-day windows.")

naive = scores.loc["SeasonalNaive"]
best = scores.loc["Ensemble"]
HEADLINE_MODE = "lead time 1 week"
main_cap = inv[(inv["capacity_multiplier"] == 1.15) & (inv["mode"] == HEADLINE_MODE)].set_index("policy")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ensemble WRMSSE", f"{best.WRMSSE:.3f}", f"{best.WRMSSE / naive.WRMSSE - 1:+.1%} vs seasonal naive",
          delta_color="inverse")
c2.metric("Ensemble WAPE", f"{best.WAPE:.1%}", f"{best.WAPE - naive.WAPE:+.1%} pts vs seasonal naive",
          delta_color="inverse")
c3.metric("Inventory cost saving (ML + LP)", f"{main_cap.loc['ML + LP', 'cost_saving_vs_rule']:.1%}",
          "vs tuned rule of thumb, 1-week lead time")
c4.metric("Revenue captured (ML + LP)", f"${main_cap.loc['ML + LP', 'revenue'] / 1e6:.2f}M",
          f"{main_cap.loc['ML + LP', 'revenue'] / main_cap.loc['Rule of thumb', 'revenue'] - 1:+.1%} vs rule, same capacity")

tab1, tab5, tab2, tab3, tab4 = st.tabs(["Forecast accuracy", "Value of accuracy", "Item explorer",
                                        "Replenishment policies", "Order planner"])

with tab1:
    st.subheader("Daily forecast accuracy, averaged over 3 backtest windows")
    st.dataframe(scores.style.format({"WAPE": "{:.1%}", "RMSSE": "{:.3f}", "WRMSSE": "{:.3f}",
                                      "Bias": "{:+.1%}", "fit_predict_seconds": "{:.0f}"}),
                 width="stretch")
    left, right = st.columns(2)
    fig = go.Figure(go.Bar(x=scores.index, y=scores["WRMSSE"],
                           marker_color=[MODEL_COLORS.get(m, "#999") for m in scores.index]))
    fig.update_layout(title="Dollar-weighted RMSSE (lower is better)", height=360)
    left.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=scores["fit_predict_seconds"], y=scores["WRMSSE"], mode="markers+text",
                               text=scores.index, textposition="top center",
                               marker=dict(size=12, color=[MODEL_COLORS.get(m, "#999") for m in scores.index])))
    fig.update_layout(title="Accuracy vs compute (3 windows, fit + predict)", xaxis_title="seconds",
                      yaxis_title="WRMSSE", height=360)
    right.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    imp = d["importance"].head(15)[::-1]
    fig = go.Figure(go.Bar(x=imp["gain_share"], y=imp.index, orientation="h", marker_color="#2563eb"))
    fig.update_layout(title="LightGBM feature importance (share of gain)", height=420)
    left.plotly_chart(fig, width="stretch")
    cov = d["coverage"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=LEVELS, y=cov["coverage"], mode="lines+markers", name="observed"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash")))
    fig.update_layout(title="Weekly quantile calibration (1 week ahead)", xaxis_title="nominal quantile",
                      yaxis_title="share of actuals below", height=420)
    right.plotly_chart(fig, width="stretch")

with tab2:
    hist, daily, weekly = d["history"], d["daily"], d["weekly"]
    s1, s2, s3 = st.columns(3)
    store = s1.selectbox("Store", sorted(hist["store_id"].unique()))
    items = hist[hist["store_id"] == store].groupby("item_id")["y"].sum().sort_values(ascending=False)
    item = s2.selectbox("Item (sorted by volume)", items.index)
    cutoff = s3.selectbox("Forecast origin", sorted(daily["cutoff"].unique()),
                          index=len(daily["cutoff"].unique()) - 1, format_func=lambda c: str(pd.Timestamp(c).date()))
    uid = f"{item}_{store}"
    h = hist[hist["unique_id"] == uid]
    f = daily[(daily["unique_id"] == uid) & (daily["cutoff"] == cutoff)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["ds"], y=h["y"], name="actual", line=dict(color="#111827")))
    for m in ["Ensemble", "LightGBM", "AutoETS", "SeasonalNaive"]:
        fig.add_trace(go.Scatter(x=f["ds"], y=f[m], name=m, line=dict(color=MODEL_COLORS[m])))
    fig.add_vline(x=pd.Timestamp(cutoff), line_dash="dot")
    fig.update_layout(title=f"Daily sales and 28-day forecasts: {uid}", height=420)
    st.plotly_chart(fig, width="stretch")

    w = weekly[(weekly["unique_id"] == uid) & (weekly["cutoff"] == cutoff) &
               (weekly["origin"] == weekly["cutoff"])].sort_values("k")
    if len(w):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["q95"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["q05"], fill="tonexty", line=dict(width=0),
                                 name="90% interval", fillcolor="rgba(37,99,235,0.15)"))
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["q75"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["q25"], fill="tonexty", line=dict(width=0),
                                 name="50% interval", fillcolor="rgba(37,99,235,0.3)"))
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["mean"], name="mean forecast", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["y"], name="actual", mode="markers+lines",
                                 line=dict(color="#111827")))
        fig.update_layout(title="Weekly demand: probabilistic forecast, 1-4 weeks ahead", height=380)
        st.plotly_chart(fig, width="stretch")

with tab3:
    caps = sorted(inv["capacity_multiplier"].unique())
    mode = st.radio("Order lead time", ["lead time 1 week", "lead time 0"], horizontal=True,
                    format_func=lambda m: "1 week (orders arrive next week)" if m.endswith("week") else "none (orders arrive immediately)")
    inv_m = inv[inv["mode"] == mode]
    cap = st.select_slider("Store capacity (multiple of recent average weekly sales)", caps, value=1.15,
                           format_func=lambda c: "unlimited" if np.isinf(c) else f"{c:.2f}x")
    s = inv_m[inv_m["capacity_multiplier"] == cap].set_index("policy").loc[
        ["Rule of thumb", "ML point", "ML newsvendor", "ML + LP"]]
    st.dataframe(s[["total_cost", "holding_cost", "stockout_cost", "fill_rate", "avg_weekly_stock",
                    "cost_saving_vs_rule"]].style.format({
        "total_cost": "${:,.0f}", "holding_cost": "${:,.0f}", "stockout_cost": "${:,.0f}",
        "fill_rate": "{:.1%}", "avg_weekly_stock": "{:,.0f}", "cost_saving_vs_rule": "{:+.1%}"}),
        width="stretch")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.index, y=s["holding_cost"], name="holding"))
    fig.add_trace(go.Bar(x=s.index, y=s["stockout_cost"], name="lost margin"))
    n_weeks = 9 if mode.endswith("week") else 12
    fig.update_layout(barmode="stack", title=f"Total cost over {n_weeks} weeks, 4 stores", height=380)
    st.plotly_chart(fig, width="stretch")
    sens = inv_m.pivot(index="capacity_multiplier", columns="policy", values="cost_saving_vs_rule")
    sens.index = ["unlimited" if np.isinf(i) else f"{i:.2f}x" for i in sens.index]
    st.markdown("**Cost saving vs rule of thumb, by capacity**")
    st.dataframe(sens.drop(columns="Rule of thumb").style.format("{:+.1%}"), width="stretch")
    cs = d["sens"][(d["sens"]["mode"] == mode) & (d["sens"]["policy"] == "ML + LP")]
    grid = cs.pivot(index="holding_frac", columns="stockout_frac", values="cost_saving_vs_rule")
    fig = go.Figure(go.Heatmap(z=grid.values * 100, x=[f"{c:.0%}" for c in grid.columns],
                               y=[f"{i:.0%}" for i in grid.index], colorscale="Blues",
                               text=[[f"{v:+.1%}" for v in row] for row in grid.values], texttemplate="%{text}"))
    fig.update_layout(title="ML + LP cost saving vs rule, across cost assumptions (capacity 1.15x)",
                      xaxis_title="lost margin per unit short (% of price)",
                      yaxis_title="holding cost per unit-week (% of price)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("Rule-of-thumb and ML-point policies use the cover factor that minimised their own cost "
               "(best case for them). Unmet demand is lost. With a 1-week lead time, orders are sized from "
               "2-week-ahead forecasts and projected stock; weeks 2-4 of each window are costed.")

with tab5:
    st.subheader("What is a better forecast worth?")
    st.markdown("Every model is pushed through the same 12-week replenishment simulation (forecast at the start of "
                "each 28-day window, weekly orders, capacity 1.15x). All models share one uncertainty model "
                "(negative binomial around the model's own mean), so cost differences come from accuracy alone.")
    dec = d["decomp"]
    fig = go.Figure(go.Waterfall(
        x=dec["step"], measure=["absolute"] + ["relative"] * (len(dec) - 1),
        y=[dec["cost"].iloc[0]] + list(-dec["saving"].iloc[1:]),
        text=[f"${dec['cost'].iloc[0] / 1e3:,.1f}k"] + [f"-${v / 1e3:,.1f}k ({sh:.0%})" for v, sh in
                                                        zip(dec["saving"].iloc[1:], dec["share_of_total_saving"].iloc[1:])],
        textposition="outside", decreasing=dict(marker_color="#2563eb")))
    fig.update_layout(title="Where the saving comes from (12 weeks, 4 stores)", yaxis_title="inventory cost ($)",
                      height=420, yaxis_range=[0, dec["cost"].iloc[0] * 1.15])
    st.plotly_chart(fig, width="stretch")
    v = d["value"]
    pts = v[v["model"] != "WeeklyLightGBM (learned quantiles)"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts["weekly_WAPE"], y=pts["cost_point_policy"], mode="markers+text", name="simple policy",
                             text=pts["model"], textposition="top center", marker=dict(size=11, color="#9ca3af")))
    fig.add_trace(go.Scatter(x=pts["weekly_WAPE"], y=pts["cost_LP_policy"], mode="markers+text", name="LP optimizer",
                             text=pts["model"], textposition="bottom center", marker=dict(size=11, color="#2563eb")))
    fig.add_hline(y=v["rule_of_thumb_cost"].iloc[0], line_dash="dash", annotation_text="rule of thumb")
    fig.update_layout(title="Weekly forecast error vs inventory cost", xaxis_title="weekly WAPE",
                      yaxis_title="inventory cost ($)", xaxis_tickformat=".0%", height=480)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(v[["model", "weekly_WAPE", "cost_point_policy", "cost_LP_policy", "fill_rate_LP", "revenue_LP"]]
                 .set_index("model").style.format({"weekly_WAPE": "{:.1%}", "cost_point_policy": "${:,.0f}",
                                                   "cost_LP_policy": "${:,.0f}", "fill_rate_LP": "{:.1%}",
                                                   "revenue_LP": "${:,.0f}"}, na_rep="-"), width="stretch")
    st.caption("Takeaways: the optimizer running on the simplest forecast (28-day moving average) costs less than the "
               "best forecast run with a simple policy. Daily accuracy rankings do not carry over: AutoETS beats the "
               "moving average on daily WRMSSE but is worse at the weekly level, where orders are placed, and costs more.")

with tab4:
    weekly = d["weekly"]
    one = weekly[weekly["k"] == 1]
    c1, c2, c3 = st.columns(3)
    store = c1.selectbox("Store ", sorted(one["store_id"].unique()))
    wk = c2.selectbox("Week", sorted(one["week_start"].unique()), format_func=lambda x: str(pd.Timestamp(x).date()))
    cap_mult = c3.slider("Capacity (x avg weekly sales)", 0.8, 2.0, 1.15, 0.05)
    c4, c5 = st.columns(2)
    hold = c4.slider("Holding cost (% of price per week)", 0.5, 10.0, config.HOLDING_COST_FRAC * 100, 0.5) / 100
    short = c5.slider("Lost margin per unit short (% of price)", 5.0, 60.0, config.STOCKOUT_COST_FRAC * 100, 1.0) / 100
    w = one[(one["store_id"] == store) & (one["week_start"] == wk)].reset_index(drop=True)
    capacity = cap_mult * w["last28_wk"].sum()
    hh, pp = hold * w["price"].to_numpy(), short * w["price"].to_numpy()
    Q = w[QCOLS].to_numpy()
    zeros = np.zeros(len(w))
    lp = optimal_levels(Q, hh, pp, zeros, capacity)
    nv = scale_to_capacity(quantile_at(Q, LEVELS, short / (short + hold)), zeros, capacity)

    def realised(level):
        sold = np.minimum(level, w["y"].to_numpy())
        return (hh * (level - sold)).sum() + (pp * (w["y"].to_numpy() - sold)).sum(), sold.sum() / w["y"].sum()

    (c_lp, f_lp), (c_nv, f_nv) = realised(lp), realised(nv)
    m1, m2, m3 = st.columns(3)
    m1.metric("Units to stock (LP)", f"{lp.sum():,.0f}", f"capacity {capacity:,.0f}")
    m2.metric("Realised cost, LP vs scaled newsvendor", f"${c_lp:,.0f}", f"{c_lp / c_nv - 1:+.1%}", delta_color="inverse")
    m3.metric("Realised fill rate (LP)", f"{f_lp:.1%}", f"{f_lp - f_nv:+.1%} pts")
    plan = pd.DataFrame({"item_id": w["item_id"], "price": w["price"], "forecast_mean": w["mean"],
                         "forecast_p95": w["q95"], "order_up_to_LP": lp.round(1),
                         "order_up_to_newsvendor": nv.round(1), "actual_demand": w["y"]})
    plan = plan.sort_values("order_up_to_LP", ascending=False)
    st.dataframe(plan, width="stretch", height=380)
    st.download_button("Download order plan (CSV)", plan.to_csv(index=False), f"order_plan_{store}_{pd.Timestamp(wk).date()}.csv")
