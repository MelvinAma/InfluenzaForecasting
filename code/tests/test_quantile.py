"""Tests for src/eval/quantile.py.

Covers:
  - compute_wis: known-value checks, decomposition, shape, non-negativity
  - fit_quantile_model: shape, coverage
  - predict_quantiles: shape, monotone ordering
  - FLUSIGHT_LEVELS: symmetry invariant
"""
import sys
import pytest
import numpy as np
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.eval.quantile import (
    QuantileModel,
    FLUSIGHT_LEVELS,
    FLUSIGHT_5_LEVELS,
    fit_quantile_model,
    predict_quantiles,
    compute_wis,
    wis_summary,
)

# ── helpers ────────────────────────────────────────────────────────────────

THREE_LEVELS = [0.025, 0.5, 0.975]  # K=1 pair, norm=1.5


def _make_perfect_quantiles(true: np.ndarray, levels: list[float]) -> np.ndarray:
    """Return quantile_preds equal to true for every quantile level."""
    T, m = true.shape
    Q = len(levels)
    out = np.stack([true] * Q, axis=2)
    return out


def _make_wide_quantiles(true: np.ndarray, width: float,
                         levels: list[float]) -> np.ndarray:
    """Return symmetric central intervals of fixed width around true."""
    T, m = true.shape
    Q = len(levels)
    sorted_levels = sorted(levels)
    median_idx = int(np.argmin([abs(l - 0.5) for l in sorted_levels]))
    out = np.zeros((T, m, Q))
    for q_idx, tau in enumerate(sorted_levels):
        offset = (tau - 0.5) * width
        out[:, :, q_idx] = true + offset
    return out


# ── TestComputeWIS ─────────────────────────────────────────────────────────

class TestComputeWIS:
    """Tests for compute_wis using three-level and five-level quantile sets."""

    def test_wis_nonneg_random(self):
        rng = np.random.default_rng(0)
        T, m, Q = 20, 5, 3
        true = rng.uniform(100, 2000, (T, m))
        qpred = rng.uniform(50, 2500, (T, m, Q))
        qpred.sort(axis=2)
        result = compute_wis(qpred, true, THREE_LEVELS)
        assert result["wis_mean"] >= 0.0

    def test_perfect_forecast_zero_wis(self):
        rng = np.random.default_rng(1)
        T, m = 10, 3
        true = rng.uniform(100, 1000, (T, m))
        qpred = _make_perfect_quantiles(true, THREE_LEVELS)
        result = compute_wis(qpred, true, THREE_LEVELS)
        assert result["wis_mean"] < 1e-6

    def test_known_wis_inside_interval(self):
        # T=1, m=1, levels=[0.025, 0.5, 0.975], K=1 pair
        # truth y=10, interval [5, 15], median=10
        # median_ae=0, width=10, under=0, over=0
        # WIS = (1/1.5) * (0 + 0.025*10) = 0.25/1.5 ≈ 0.1667
        true = np.array([[10.0]])
        qpred = np.array([[[5.0, 10.0, 15.0]]])  # (1, 1, 3)
        result = compute_wis(qpred, true, THREE_LEVELS)
        expected = (0.025 * 10.0) / 1.5
        assert abs(result["wis_mean"] - expected) < 1e-6

    def test_known_wis_outside_interval(self):
        # Same setup but y=20 (above interval)
        # median_ae = |20-10| = 10
        # width=10, under=0, over=max(20-15,0)=5
        # IS_0.05 = 10 + (2/0.05)*5 = 210
        # WIS = (1/1.5) * (0.5*10 + 0.025*210) = (5 + 5.25)/1.5 = 6.833...
        true = np.array([[20.0]])
        qpred = np.array([[[5.0, 10.0, 15.0]]])
        result = compute_wis(qpred, true, THREE_LEVELS)
        expected = (0.5 * 10.0 + 0.025 * 210.0) / 1.5
        assert abs(result["wis_mean"] - expected) < 1e-6

    def test_known_wis_below_interval(self):
        # y=2 (below interval [5, 15])
        # median_ae = |2-10| = 8
        # under=max(5-2,0)=3, over=0
        # IS = 10 + (2/0.05)*3 = 10 + 120 = 130
        # WIS = (1/1.5)*(0.5*8 + 0.025*130) = (4 + 3.25)/1.5 = 4.833...
        true = np.array([[2.0]])
        qpred = np.array([[[5.0, 10.0, 15.0]]])
        result = compute_wis(qpred, true, THREE_LEVELS)
        expected = (0.5 * 8.0 + 0.025 * 130.0) / 1.5
        assert abs(result["wis_mean"] - expected) < 1e-6

    def test_decomposition_sharpness_plus_calibration_equals_wis(self):
        rng = np.random.default_rng(2)
        T, m = 15, 4
        true = rng.uniform(100, 2000, (T, m))
        qpred = _make_wide_quantiles(true, width=200.0, levels=FLUSIGHT_5_LEVELS)
        result = compute_wis(qpred, true, FLUSIGHT_5_LEVELS)
        total = result["sharpness"] + result["calibration"]
        assert abs(total - result["wis_mean"]) < 1e-6

    def test_output_keys(self):
        rng = np.random.default_rng(3)
        T, m = 5, 2
        true = rng.uniform(0, 100, (T, m))
        qpred = _make_perfect_quantiles(true, THREE_LEVELS)
        result = compute_wis(qpred, true, THREE_LEVELS)
        for key in ("wis_mean", "wis_per_region", "wis_per_week",
                    "sharpness", "calibration", "n_levels", "n_pairs"):
            assert key in result

    def test_per_region_shape(self):
        T, m = 10, 7
        true = np.ones((T, m))
        qpred = _make_perfect_quantiles(true, THREE_LEVELS)
        result = compute_wis(qpred, true, THREE_LEVELS)
        assert result["wis_per_region"].shape == (m,)

    def test_per_week_shape(self):
        T, m = 10, 7
        true = np.ones((T, m))
        qpred = _make_perfect_quantiles(true, THREE_LEVELS)
        result = compute_wis(qpred, true, THREE_LEVELS)
        assert result["wis_per_week"].shape == (T,)

    def test_n_pairs_five_levels(self):
        # [0.025, 0.25, 0.5, 0.75, 0.975] → 2 pairs
        T, m = 5, 2
        true = np.ones((T, m))
        qpred = _make_perfect_quantiles(true, FLUSIGHT_5_LEVELS)
        result = compute_wis(qpred, true, FLUSIGHT_5_LEVELS)
        assert result["n_pairs"] == 2

    def test_n_pairs_three_levels(self):
        T, m = 5, 2
        true = np.ones((T, m))
        qpred = _make_perfect_quantiles(true, THREE_LEVELS)
        result = compute_wis(qpred, true, THREE_LEVELS)
        assert result["n_pairs"] == 1

    def test_wider_interval_higher_sharpness_cost(self):
        T, m = 10, 3
        true = np.ones((T, m)) * 500.0
        qpred_narrow = _make_wide_quantiles(true, width=10.0, levels=FLUSIGHT_5_LEVELS)
        qpred_wide   = _make_wide_quantiles(true, width=500.0, levels=FLUSIGHT_5_LEVELS)
        r_narrow = compute_wis(qpred_narrow, true, FLUSIGHT_5_LEVELS)
        r_wide   = compute_wis(qpred_wide,   true, FLUSIGHT_5_LEVELS)
        assert r_wide["sharpness"] > r_narrow["sharpness"]

    def test_shape_mismatch_raises(self):
        true   = np.ones((5, 3))
        qpred  = np.ones((5, 3, 5))   # wrong Q
        with pytest.raises(ValueError):
            compute_wis(qpred, true, THREE_LEVELS)


# ── TestFitQuantileModel ───────────────────────────────────────────────────

class TestFitQuantileModel:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.T, self.m = 80, 4
        self.preds = rng.uniform(100, 2000, (self.T, self.m)).astype(float)
        noise = rng.normal(0, 100, (self.T, self.m))
        self.true = self.preds + noise

    def test_returns_quantile_model(self):
        qm = fit_quantile_model(self.preds, self.true, FLUSIGHT_5_LEVELS)
        assert isinstance(qm, QuantileModel)

    def test_coefficients_shape(self):
        Q = len(FLUSIGHT_5_LEVELS)
        qm = fit_quantile_model(self.preds, self.true, FLUSIGHT_5_LEVELS)
        assert qm.coefficients.shape == (Q, self.m, 2)

    def test_levels_stored_sorted(self):
        levels = [0.975, 0.025, 0.5]
        qm = fit_quantile_model(self.preds, self.true, levels)
        assert qm.levels == sorted(levels)

    def test_approximate_coverage(self):
        # For level τ, approximately τ fraction of residuals should be negative
        # (pred_quantile > true), i.e. true < q_τ
        qm = fit_quantile_model(self.preds, self.true, FLUSIGHT_5_LEVELS)
        qpred = predict_quantiles(qm, self.preds)

        for q_idx, tau in enumerate(FLUSIGHT_5_LEVELS):
            # Coverage: fraction of times true <= predicted quantile
            coverage = np.mean(self.true <= qpred[:, :, q_idx])
            # Allow generous tolerance (calibration on training data)
            assert abs(coverage - tau) < 0.25, (
                f"Coverage at tau={tau}: {coverage:.3f}, expected ~{tau}"
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_quantile_model(self.preds, self.true[:10], FLUSIGHT_5_LEVELS)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            fit_quantile_model(self.preds[:, 0], self.true[:, 0], FLUSIGHT_5_LEVELS)


# ── TestPredictQuantiles ───────────────────────────────────────────────────

class TestPredictQuantiles:
    def setup_method(self):
        rng = np.random.default_rng(7)
        T_cal, self.m = 50, 3
        self.T_test = 20
        cal_preds = rng.uniform(100, 1000, (T_cal, self.m)).astype(float)
        cal_true  = cal_preds + rng.normal(0, 80, (T_cal, self.m))
        self.qm = fit_quantile_model(cal_preds, cal_true, FLUSIGHT_5_LEVELS)
        self.test_preds = rng.uniform(100, 1000, (self.T_test, self.m)).astype(float)

    def test_output_shape(self):
        out = predict_quantiles(self.qm, self.test_preds)
        assert out.shape == (self.T_test, self.m, len(FLUSIGHT_5_LEVELS))

    def test_monotone_ordering_enforced(self):
        out = predict_quantiles(self.qm, self.test_preds, enforce_monotone=True)
        # Each (t, r) slice should be non-decreasing
        diffs = np.diff(out, axis=2)
        assert np.all(diffs >= -1e-9)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            predict_quantiles(self.qm, self.test_preds[:, 0])

    def test_median_close_to_point_pred(self):
        # Median quantile prediction should be close to the point prediction
        # when calibration data has low noise
        rng = np.random.default_rng(99)
        T_cal, m = 200, 2
        preds_cal = rng.uniform(100, 1000, (T_cal, m)).astype(float)
        true_cal  = preds_cal + rng.normal(0, 5, (T_cal, m))  # low noise
        qm = fit_quantile_model(preds_cal, true_cal, FLUSIGHT_5_LEVELS)
        test_preds = rng.uniform(200, 800, (30, m)).astype(float)
        out = predict_quantiles(qm, test_preds)
        median_idx = FLUSIGHT_5_LEVELS.index(0.5)
        # Median should be within 50 units of point prediction
        assert np.all(np.abs(out[:, :, median_idx] - test_preds) < 50.0)


# ── TestFluSightLevels ─────────────────────────────────────────────────────

class TestFluSightLevels:
    def test_23_levels_count(self):
        assert len(FLUSIGHT_LEVELS) == 23

    def test_5_levels_count(self):
        assert len(FLUSIGHT_5_LEVELS) == 5

    def test_levels_sorted(self):
        assert FLUSIGHT_LEVELS == sorted(FLUSIGHT_LEVELS)
        assert FLUSIGHT_5_LEVELS == sorted(FLUSIGHT_5_LEVELS)

    def test_median_present(self):
        assert 0.5 in FLUSIGHT_LEVELS
        assert 0.5 in FLUSIGHT_5_LEVELS

    def _approx_in(self, value, seq, tol=1e-9):
        return any(abs(value - x) < tol for x in seq)

    def test_23_levels_symmetric_around_half(self):
        for tau in FLUSIGHT_LEVELS:
            if abs(tau - 0.5) > 1e-9:
                assert self._approx_in(1.0 - tau, FLUSIGHT_LEVELS), (
                    f"1-{tau}={1-tau} not in FLUSIGHT_LEVELS"
                )

    def test_5_levels_symmetric_around_half(self):
        for tau in FLUSIGHT_5_LEVELS:
            if abs(tau - 0.5) > 1e-9:
                assert self._approx_in(1.0 - tau, FLUSIGHT_5_LEVELS), (
                    f"1-{tau}={1-tau} not in FLUSIGHT_5_LEVELS"
                )


# ── TestWISSummary ─────────────────────────────────────────────────────────

class TestWISSummary:
    def test_returns_model_label(self):
        wis_result = {
            "wis_mean": 12.3, "sharpness": 5.0, "calibration": 7.3,
            "wis_per_region": np.array([12.3]),
            "wis_per_week": np.array([12.3]),
            "n_levels": 5, "n_pairs": 2,
        }
        row = wis_summary(wis_result, "EpiGNN")
        assert row["Model"] == "EpiGNN"
        assert row["WIS"] == 12.3
