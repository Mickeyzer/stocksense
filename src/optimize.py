"""Capacity-constrained replenishment as a linear program.

For each item i choose an order-up-to level y_i. Demand is represented by K equally
likely scenarios: the forecast quantiles at the midpoints of K equal-probability bins
(0.05, 0.15, ..., 0.95). Expected cost is

    (1/K) * sum_i sum_k [ h_i * (y_i - d_ik)^+  +  p_i * (d_ik - y_i)^+ ]

with h = holding cost per leftover unit and p = lost margin per unmet unit. The
positive parts become auxiliary variables, giving an LP. Constraints: total stock
after ordering fits the store's capacity, and stock already on the shelf cannot be
removed (y_i >= on_hand_i). Without the capacity constraint the solution is the
classic newsvendor critical-ratio quantile p / (p + h) for every item.
"""
import cvxpy as cp
import numpy as np


def optimal_levels(Q: np.ndarray, h: np.ndarray, p: np.ndarray,
                   on_hand: np.ndarray, capacity: float) -> np.ndarray:
    n, K = Q.shape
    y = cp.Variable(n)
    over = cp.Variable((n, K), nonneg=True)
    under = cp.Variable((n, K), nonneg=True)
    Y = cp.reshape(y, (n, 1), order="C") @ np.ones((1, K))
    cons = [over >= Y - Q, under >= Q - Y, y >= on_hand]
    if np.isfinite(capacity):
        cons.append(cp.sum(y) <= max(capacity, on_hand.sum()))
    cost = cp.sum(cp.multiply(h[:, None], over) + cp.multiply(p[:, None], under)) / K
    cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.HIGHS)
    return np.maximum(y.value, on_hand)


def scale_to_capacity(target: np.ndarray, on_hand: np.ndarray, capacity: float) -> np.ndarray:
    """Heuristic used by the baseline policies: shrink every top-up proportionally."""
    target = np.maximum(target, on_hand)
    need = target - on_hand
    room = capacity - on_hand.sum()
    if not np.isfinite(capacity) or need.sum() <= room:
        return target
    return on_hand + need * max(room, 0) / need.sum()


def quantile_at(Q: np.ndarray, levels: np.ndarray, tau: float) -> np.ndarray:
    """Per-row linear interpolation of a quantile level from the forecast quantiles."""
    return np.array([np.interp(tau, levels, row) for row in Q])
