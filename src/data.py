"""Build a tidy daily panel (one row per item-store-day) from the raw M5 files."""
import numpy as np
import pandas as pd

from src import config


def build_panel() -> pd.DataFrame:
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    # Train file covers d_1..d_1941; the test file holds the official M5 evaluation
    # period (d_1942..d_1969), which becomes our final backtest window.
    train = pd.read_csv(config.RAW_DIR / "sales_train_evaluation.csv")
    test = pd.read_csv(config.RAW_DIR / "sales_test_evaluation.csv")
    sales = train.merge(test, on=id_cols, how="inner")
    sales = sales[(sales["dept_id"] == config.DEPT) & (sales["store_id"].isin(config.STORES))]

    long = sales.melt(id_vars=id_cols, var_name="d", value_name="y")

    cal = pd.read_csv(config.RAW_DIR / "calendar.csv", parse_dates=["date"])
    cal["d"] = "d_" + (np.arange(len(cal)) + 1).astype(str)  # calendar rows are d_1, d_2, ...
    long = long.merge(cal, on="d", how="left")
    long = long[long["date"] >= config.START_DATE]

    prices = pd.read_csv(config.RAW_DIR / "sell_prices.csv")
    prices = prices[prices["store_id"].isin(config.STORES)]
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long["unique_id"] = long["item_id"] + "_" + long["store_id"]
    long = long.rename(columns={"date": "ds"})

    # An item has no price before its launch; drop those pre-launch rows.
    long = long[long["sell_price"].notna()]
    long = long.sort_values(["unique_id", "ds"])

    snap_col = "snap_" + long["state_id"].iloc[0]
    long["snap"] = long[snap_col].astype("int8")
    long["event"] = long["event_name_1"].notna().astype("int8")
    long["event_type"] = long["event_type_1"].fillna("none")

    keep = ["unique_id", "item_id", "store_id", "ds", "y", "sell_price",
            "snap", "event", "event_type", "event_name_1"]
    panel = long[keep].reset_index(drop=True)
    panel["y"] = panel["y"].astype("float32")
    panel["sell_price"] = panel["sell_price"].astype("float32")
    return panel


def load_panel() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "panel.parquet"
    if not path.exists():
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        build_panel().to_parquet(path, index=False)
    return pd.read_parquet(path)


def cutoffs(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """Forecast origins: the last day of training data for each backtest window."""
    last = pd.Timestamp(panel["ds"].max())
    return [last - pd.Timedelta(days=config.HORIZON * k) for k in range(config.N_WINDOWS, 0, -1)]


if __name__ == "__main__":
    p = load_panel()
    print(p.shape, p["unique_id"].nunique(), "series,", p["ds"].min().date(), "to", p["ds"].max().date())
    print("zero-demand share:", round(float((p["y"] == 0).mean()), 3))
    print("cutoffs:", [c.date() for c in cutoffs(p)])
