import numpy as np
import pandas as pd

from src.metrics import bias, pinball, rmsse_scale, wape
from src.optimize import optimal_levels, quantile_at, scale_to_capacity

LEVELS = np.arange(0.05, 1.0, 0.1)


def test_wape_and_bias():
    y = np.array([10.0, 0.0, 10.0])
    yhat = np.array([12.0, 1.0, 7.0])
    assert np.isclose(wape(y, yhat), 6 / 20)
    assert np.isclose(bias(y, yhat), 0.0)


def test_pinball_is_zero_for_perfect_forecast():
    y = np.array([1.0, 5.0])
    assert pinball(y, y, 0.9) == 0.0


def test_rmsse_scale_ignores_leading_zeros():
    train = pd.DataFrame({"unique_id": "a", "ds": pd.date_range("2020-01-01", periods=5),
                          "y": [0, 0, 2, 4, 2]})
    assert np.isclose(rmsse_scale(train)["a"], (4 + 4) / 2)


def test_lp_without_capacity_matches_newsvendor_quantile():
    rng = np.random.default_rng(0)
    Q = np.sort(rng.gamma(2, 5, size=(20, 10)), axis=1)
    h, p = np.full(20, 1.0), np.full(20, 3.0)  # critical ratio 0.75 -> 8th of 10 scenarios
    y = optimal_levels(Q, h, p, np.zeros(20), np.inf)
    assert np.allclose(y, Q[:, 7], atol=1e-4)


def test_lp_respects_capacity_and_on_hand():
    Q = np.tile(np.arange(1, 11, dtype=float), (5, 1))
    on_hand = np.array([6.0, 0, 0, 0, 0])
    y = optimal_levels(Q, np.ones(5), np.full(5, 9.0), on_hand, capacity=20)
    assert y.sum() <= 20 + 1e-6
    assert y[0] >= 6 - 1e-6


def test_lp_prioritises_high_margin_items():
    Q = np.tile(np.arange(1, 11, dtype=float), (2, 1))
    y = optimal_levels(Q, np.ones(2), np.array([2.0, 20.0]), np.zeros(2), capacity=12)
    assert y[1] > y[0]


def test_scale_to_capacity():
    y = scale_to_capacity(np.array([10.0, 10.0]), np.array([2.0, 0.0]), capacity=12)
    assert np.isclose(y.sum(), 12) and y[0] >= 2


def test_quantile_interpolation():
    Q = np.array([LEVELS * 100])
    assert np.isclose(quantile_at(Q, LEVELS, 0.5)[0], 50)
