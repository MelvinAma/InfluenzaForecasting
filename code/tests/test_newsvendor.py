"""Tests for src/capacity/newsvendor.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.capacity import (
    critical_ratio,
    optimal_stock_from_quantiles,
    realized_cost,
    evaluate_newsvendor,
)


def test_critical_ratio_symmetric():
    assert critical_ratio(1.0, 1.0) == pytest.approx(0.5)


def test_critical_ratio_extreme():
    assert critical_ratio(9.0, 1.0) == pytest.approx(0.9)
    assert critical_ratio(1.0, 9.0) == pytest.approx(0.1)


def test_critical_ratio_rejects_negative():
    with pytest.raises(ValueError):
        critical_ratio(-1.0, 1.0)


def test_critical_ratio_rejects_zero_sum():
    with pytest.raises(ValueError):
        critical_ratio(0.0, 0.0)


def test_optimal_stock_at_median():
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array([5.0, 10.0, 20.0])
    assert optimal_stock_from_quantiles(levels, values, 0.5) == pytest.approx(10.0)


def test_optimal_stock_interpolates():
    levels = np.array([0.25, 0.75])
    values = np.array([4.0, 12.0])
    # cr=0.5 is halfway between the two levels
    assert optimal_stock_from_quantiles(levels, values, 0.5) == pytest.approx(8.0)


def test_realized_cost_under():
    total, under, over = realized_cost(q_star=5.0, demand=10.0, c_u=3.0, c_o=1.0)
    assert under == pytest.approx(15.0)
    assert over == pytest.approx(0.0)
    assert total == pytest.approx(15.0)


def test_realized_cost_over():
    total, under, over = realized_cost(q_star=10.0, demand=5.0, c_u=3.0, c_o=1.0)
    assert under == pytest.approx(0.0)
    assert over == pytest.approx(5.0)
    assert total == pytest.approx(5.0)


def test_realized_cost_exact_match():
    total, under, over = realized_cost(q_star=5.0, demand=5.0, c_u=3.0, c_o=1.0)
    assert total == pytest.approx(0.0)


def test_evaluate_newsvendor_shape_and_totals():
    # Simple deterministic case: 2 weeks, 2 regions, 3 quantile levels.
    levels = np.array([0.1, 0.5, 0.9])
    # Build qpred so that q_star at cr=0.5 is the middle column.
    qpred = np.array(
        [
            [[0.0, 10.0, 20.0], [5.0, 15.0, 25.0]],  # week 0
            [[1.0, 11.0, 21.0], [6.0, 16.0, 26.0]],  # week 1
        ]
    )
    gt = np.array([[12.0, 14.0], [11.0, 16.0]])  # mix of under/over/exact

    # c_u=c_o=1 -> cr=0.5, picks the median column.
    result = evaluate_newsvendor(qpred, gt, levels, c_u=1.0, c_o=1.0)

    assert result["q_star"].shape == (2, 2)
    assert result["q_star"][0, 0] == pytest.approx(10.0)
    assert result["q_star"][0, 1] == pytest.approx(15.0)
    assert result["q_star"][1, 0] == pytest.approx(11.0)
    assert result["q_star"][1, 1] == pytest.approx(16.0)

    # Week 0: under (12-10)*1=2, over (15-14)*1=1 -> 3
    # Week 1: under (11-11)=0, over (16-16)=0 -> 0
    assert result["total_under"] == pytest.approx(2.0)
    assert result["total_over"] == pytest.approx(1.0)
    assert result["total_cost"] == pytest.approx(3.0)
    assert result["n_weeks"] == 2
    assert result["n_regions"] == 2


def test_evaluate_newsvendor_high_ratio_pushes_stock_up():
    """A high c_u/c_o ratio picks a higher quantile, so service level rises."""
    levels = np.array([0.1, 0.5, 0.9])
    qpred = np.tile(np.array([5.0, 10.0, 20.0]), (5, 3, 1))  # (T=5, m=3, Q=3)
    gt = np.full((5, 3), 15.0)  # demand between median and 90th percentile

    low = evaluate_newsvendor(qpred, gt, levels, c_u=1.0, c_o=1.0)
    high = evaluate_newsvendor(qpred, gt, levels, c_u=9.0, c_o=1.0)

    assert high["overall_service_level"] >= low["overall_service_level"]
    # At cr=0.5 q*=10 < 15, always under-capacity.
    assert low["overall_service_level"] == pytest.approx(0.0)
    # At cr=0.9 q*=20 > 15, always over-capacity, service level = 1.
    assert high["overall_service_level"] == pytest.approx(1.0)


def test_evaluate_newsvendor_rejects_bad_shape():
    levels = np.array([0.1, 0.5, 0.9])
    qpred = np.zeros((2, 3))
    gt = np.zeros((2, 3))
    with pytest.raises(ValueError):
        evaluate_newsvendor(qpred, gt, levels, c_u=1.0, c_o=1.0)


def test_evaluate_newsvendor_rejects_nan_qpred():
    levels = np.array([0.1, 0.5, 0.9])
    qpred = np.array([[[0.0, np.nan, 20.0], [5.0, 15.0, 25.0]]])
    gt = np.array([[12.0, 14.0]])
    with pytest.raises(ValueError, match="qpred contains NaN"):
        evaluate_newsvendor(qpred, gt, levels, c_u=1.0, c_o=1.0)


def test_evaluate_newsvendor_rejects_nan_gt():
    levels = np.array([0.1, 0.5, 0.9])
    qpred = np.array([[[0.0, 10.0, 20.0], [5.0, 15.0, 25.0]]])
    gt = np.array([[12.0, np.nan]])
    with pytest.raises(ValueError, match="gt contains NaN"):
        evaluate_newsvendor(qpred, gt, levels, c_u=1.0, c_o=1.0)


def test_evaluate_newsvendor_q_star_matches_scalar_helper():
    """q_star from the batch path must agree cell-by-cell with the scalar
    helper on sorted inputs (single source of truth for the inverse CDF)."""
    rng = np.random.default_rng(0)
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    T, m, Q = 4, 3, len(levels)
    qpred = np.sort(rng.uniform(0, 100, size=(T, m, Q)), axis=2)
    gt = rng.uniform(0, 100, size=(T, m))
    cr = 3.0 / 4.0
    result = evaluate_newsvendor(qpred, gt, levels, c_u=3.0, c_o=1.0)
    assert result["critical_ratio"] == pytest.approx(cr)
    for t in range(T):
        for r in range(m):
            expected = optimal_stock_from_quantiles(levels, qpred[t, r], cr)
            assert result["q_star"][t, r] == pytest.approx(expected)


def test_evaluate_newsvendor_aggregation_invariants():
    """total_cost must equal total_under + total_over, sum of per-region
    totals must equal the global total, service levels must live in [0, 1]."""
    rng = np.random.default_rng(1)
    levels = np.array([0.1, 0.5, 0.9])
    T, m, Q = 5, 4, len(levels)
    qpred = np.sort(rng.uniform(0, 50, size=(T, m, Q)), axis=2)
    gt = rng.uniform(0, 60, size=(T, m))
    result = evaluate_newsvendor(qpred, gt, levels, c_u=2.0, c_o=1.0)

    assert result["total_cost"] == pytest.approx(
        result["total_under"] + result["total_over"]
    )
    assert result["total_cost"] == pytest.approx(
        float(result["total_cost_per_region"].sum())
    )
    per_region_sum = (
        result["under_cost_per_region"] + result["over_cost_per_region"]
    )
    assert np.allclose(per_region_sum, result["total_cost_per_region"])
    assert np.all(
        (result["service_level_per_region"] >= 0.0)
        & (result["service_level_per_region"] <= 1.0)
    )


def test_optimal_stock_at_stored_level():
    """cr that lies exactly on a stored quantile level returns that value."""
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array([5.0, 10.0, 20.0])
    assert optimal_stock_from_quantiles(levels, values, 0.1) == pytest.approx(5.0)
    assert optimal_stock_from_quantiles(levels, values, 0.9) == pytest.approx(20.0)
