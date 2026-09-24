"""Reproduce every result in results/ from the raw M5 data, in order."""
from datasetsforecast.m5 import M5

from src import backtest, config, export, simulate, tune, value, weekly

if __name__ == "__main__":
    if not (config.RAW_DIR / "sales_test_evaluation.csv").exists():
        M5.download(directory=str(config.ROOT / "data"))
    tune.main()       # LightGBM settings + ensemble weight on a pre-test validation window
    backtest.main()   # daily models, 3 rolling windows
    weekly.main()     # weekly mean + quantile models
    simulate.main()   # replenishment policy simulation
    value.main()      # value of forecast accuracy in dollars
    export.main()     # slim history for the dashboard
