"""Project-wide settings. Everything that changes an experiment lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "m5" / "datasets"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Scope: one department across the four California stores.
DEPT = "FOODS_3"
STORES = ["CA_1", "CA_2", "CA_3", "CA_4"]
START_DATE = "2013-01-01"  # older history adds rows but little signal

# Backtesting: M5 horizon, three rolling origins ending at the last observed day.
HORIZON = 28
N_WINDOWS = 3
SEASON = 7

# Inventory economics (per unit, as a fraction of the item's selling price).
HOLDING_COST_FRAC = 0.02  # per unit per week left on the shelf (capital, space, spoilage)
STOCKOUT_COST_FRAC = 0.30  # lost gross margin per unit of unmet demand
# Backroom capacity per store per week, as a multiple of that store's average weekly sales.
CAPACITY_MULTIPLIER = 1.15

QUANTILES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
SEED = 42
