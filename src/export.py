"""Write the slim history file the dashboard needs, so the app runs without raw data."""
import pandas as pd

from src import config
from src.data import cutoffs, load_panel


def main():
    panel = load_panel()
    start = cutoffs(panel)[0] - pd.Timedelta(days=120)
    hist = panel.loc[panel["ds"] >= start, ["unique_id", "item_id", "store_id", "ds", "y"]]
    hist.to_parquet(config.RESULTS_DIR / "history.parquet", index=False)
    print(f"history: {len(hist):,} rows")


if __name__ == "__main__":
    main()
