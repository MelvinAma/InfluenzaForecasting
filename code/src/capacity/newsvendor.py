"""Newsvendor capacity-planning evaluation on probabilistic forecasts.

A forecast for week t in region r is a quantile distribution
(levels, values). Given cost parameters c_u (under-capacity) and c_o
(over-capacity), the optimal stocking level is the critical-ratio
quantile q* = F^{-1}(c_u/(c_u+c_o)). Weekly realized cost is
c_u * max(D - q*, 0) + c_o * max(q* - D, 0).

Costs are expressed in forecast-native units (bed-weeks for US NHSN,
case-weeks for Sweden). The absolute unit is irrelevant for the
ratio-sweep analysis used in Methodology Section 3.8 -- only c_u/c_o
matters for q*, and all reported differences are between models under
the same cost ratio.

Usage:
    from src.capacity import evaluate_newsvendor, critical_ratio
    result = evaluate_newsvendor(
        qpred=frames,           # (T, m, Q)
        gt=ground_truth,        # (T, m)
        levels=FLUSIGHT_LEVELS, # (Q,)
        c_u=3.0, c_o=1.0,
    )
    print(result["total_cost"], result["overall_service_level"])
"""
import numpy as np


def critical_ratio(c_u: float, c_o: float) -> float:
    if c_u < 0 or c_o < 0:
        raise ValueError(f"costs must be non-negative (got c_u={c_u}, c_o={c_o})")
    if c_u + c_o == 0:
        raise ValueError("c_u + c_o must be positive")
    return c_u / (c_u + c_o)


def optimal_stock_from_quantiles(
    levels: np.ndarray, values: np.ndarray, cr: float
) -> float:
    """Interpolate the quantile distribution at the critical ratio."""
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    if levels.shape != values.shape:
        raise ValueError("levels and values must have the same shape")
    if not np.all(np.diff(levels) > 0):
        raise ValueError("levels must be strictly increasing")
    values_sorted = np.sort(values)
    return float(np.interp(cr, levels, values_sorted))


def realized_cost(
    q_star: float, demand: float, c_u: float, c_o: float
) -> tuple[float, float, float]:
    """Return (total_cost, under_cost, over_cost) for a single period."""
    under = max(demand - q_star, 0.0) * c_u
    over = max(q_star - demand, 0.0) * c_o
    return under + over, under, over


def evaluate_newsvendor(
    qpred: np.ndarray,
    gt: np.ndarray,
    levels: np.ndarray,
    c_u: float,
    c_o: float,
) -> dict:
    """Run newsvendor evaluation over a (T, m, Q) quantile array.

    Parameters
    ----------
    qpred : (T, m, Q) forecast quantiles. Assumed sorted on the last axis
            (the FluSight hub parser enforces this).
    gt    : (T, m) realized demand.
    levels: (Q,) quantile levels in (0, 1), strictly increasing.
    c_u, c_o : non-negative cost weights.

    Returns
    -------
    dict with keys:
        q_star                     : (T, m) chosen capacity per week-region
        under_cost, over_cost      : (T, m) per-cell costs
        total_cost_per_region      : (m,) summed across weeks
        under_cost_per_region      : (m,)
        over_cost_per_region       : (m,)
        service_level_per_region   : (m,) fraction of weeks demand met
        total_cost, total_under, total_over : scalars
        overall_service_level      : scalar in [0, 1]
        n_weeks, n_regions         : ints
        critical_ratio             : c_u/(c_u+c_o)
    """
    qpred = np.asarray(qpred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    levels = np.asarray(levels, dtype=float)

    if qpred.ndim != 3:
        raise ValueError(f"qpred must be (T, m, Q), got shape {qpred.shape}")
    T, m, Q = qpred.shape
    if gt.shape != (T, m):
        raise ValueError(f"gt shape {gt.shape} does not match (T={T}, m={m})")
    if levels.shape != (Q,):
        raise ValueError(f"levels shape {levels.shape} does not match Q={Q}")
    if np.isnan(qpred).any():
        raise ValueError("qpred contains NaN; filter incomplete submissions before evaluation")
    if np.isnan(gt).any():
        raise ValueError("gt contains NaN; filter missing-truth weeks before evaluation")

    cr = critical_ratio(c_u, c_o)

    q_star = np.empty((T, m))
    for t in range(T):
        for r in range(m):
            q_star[t, r] = np.interp(cr, levels, qpred[t, r])

    under_cost = np.maximum(gt - q_star, 0.0) * c_u
    over_cost = np.maximum(q_star - gt, 0.0) * c_o
    total_cost = under_cost + over_cost
    met = gt <= q_star

    return {
        "q_star": q_star,
        "under_cost": under_cost,
        "over_cost": over_cost,
        "total_cost_per_region": total_cost.sum(axis=0),
        "under_cost_per_region": under_cost.sum(axis=0),
        "over_cost_per_region": over_cost.sum(axis=0),
        "service_level_per_region": met.mean(axis=0),
        "total_cost": float(total_cost.sum()),
        "total_under": float(under_cost.sum()),
        "total_over": float(over_cost.sum()),
        "overall_service_level": float(met.mean()),
        "n_weeks": T,
        "n_regions": m,
        "critical_ratio": cr,
    }
