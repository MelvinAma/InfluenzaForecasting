"""Post-hoc quantile regression and Weighted Interval Score (WIS) for flu forecasting.

Fits a linear quantile model per (quantile level, region) on calibration predictions
and ground truth.  The single predictor is the point prediction ŷ, which captures
heteroskedasticity: uncertainty scales with predicted magnitude, a realistic
assumption for count data.

WIS formula follows Bracher et al. (2021) / FluSight hub evaluation protocol.
The decomposition into sharpness (interval width) and calibration (undercoverage
penalty) matches the components reported by the FluSight hub.

Usage:
    from src.eval.quantile import fit_quantile_model, predict_quantiles, compute_wis
    from src.eval.quantile import FLUSIGHT_5_LEVELS
    from src.eval.evaluate import inference_epignn

    # Get validation predictions for calibration
    val_pred, val_true = inference_epignn(best_state, config, config_path, split="val")

    # Fit quantile model on validation split
    qmodel = fit_quantile_model(val_pred, val_true, quantile_levels=FLUSIGHT_5_LEVELS)

    # Apply to test predictions
    test_pred, test_true = inference_epignn(best_state, config, config_path, split="test")
    quant_preds = predict_quantiles(qmodel, test_pred)  # (T, m, Q)

    wis = compute_wis(quant_preds, test_true, FLUSIGHT_5_LEVELS)
    print(f"WIS: {wis['wis_mean']:.2f}  (sharpness {wis['sharpness']:.2f}, "
          f"calibration {wis['calibration']:.2f})")

Limitations:
    Linear quantile regression post-hoc is less principled than training with
    pinball loss.  Quantile coverage holds approximately on the calibration
    distribution but may degrade on held-out seasons.  Document as a limitation.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import QuantileRegressor

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# FluSight standard 23-level quantile set (Bracher 2021 / CDC protocol)
FLUSIGHT_LEVELS: list[float] = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
]

# 5-level subset matching the FluSight forecast CSV in Data/flu_forecasts_data.csv
FLUSIGHT_5_LEVELS: list[float] = [0.025, 0.25, 0.5, 0.75, 0.975]


@dataclass
class QuantileModel:
    """Fitted post-hoc linear quantile model for all levels and regions.

    Attributes
    ----------
    levels : list of float
        Quantile levels this model was fitted for, sorted ascending.
    coefficients : np.ndarray, shape (Q, m, 2)
        ``coefficients[q, r] = [slope, intercept]`` for level ``levels[q]``
        and region ``r``.  Prediction: ``q_τ(ŷ) = slope * ŷ + intercept``.
    """

    levels: list[float]
    coefficients: np.ndarray  # (Q, m, 2)


def fit_quantile_model(
    cal_preds: np.ndarray,
    cal_true: np.ndarray,
    quantile_levels: list[float] = FLUSIGHT_LEVELS,
    alpha: float = 0.0,
) -> QuantileModel:
    """Fit linear quantile regression for each quantile level and each region.

    Parameters
    ----------
    cal_preds : np.ndarray, shape (T_cal, m)
        Point predictions on calibration data in original scale.
        Typically the validation split predictions from the best-epoch model.
    cal_true : np.ndarray, shape (T_cal, m)
        Ground truth on calibration data in original scale.
    quantile_levels : list of float
        Quantile levels to fit.  Default: FluSight 23-level set.
    alpha : float
        L1 regularisation strength for QuantileRegressor (0 = no regularisation).
        Keep at 0 unless the calibration set is very small.

    Returns
    -------
    QuantileModel with ``coefficients`` shape (Q, m, 2).
    """
    if cal_preds.shape != cal_true.shape:
        raise ValueError(
            f"cal_preds {cal_preds.shape} and cal_true {cal_true.shape} must match"
        )
    if cal_preds.ndim != 2:
        raise ValueError("cal_preds must be 2-D (T_cal, m)")

    T_cal, m = cal_preds.shape
    levels = sorted(quantile_levels)
    Q = len(levels)
    coefficients = np.zeros((Q, m, 2), dtype=float)

    for q_idx, tau in enumerate(levels):
        for r in range(m):
            X = cal_preds[:, r].reshape(-1, 1).astype(float)
            y = cal_true[:, r].astype(float)
            reg = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
            reg.fit(X, y)
            coefficients[q_idx, r, 0] = reg.coef_[0]
            coefficients[q_idx, r, 1] = reg.intercept_

    return QuantileModel(levels=levels, coefficients=coefficients)


def predict_quantiles(
    qmodel: QuantileModel,
    point_preds: np.ndarray,
    enforce_monotone: bool = True,
) -> np.ndarray:
    """Apply a fitted QuantileModel to new point predictions.

    Parameters
    ----------
    qmodel : QuantileModel
        Fitted model from ``fit_quantile_model``.
    point_preds : np.ndarray, shape (T, m)
        Point predictions in original scale.
    enforce_monotone : bool
        If True, sort the Q quantile values at each (t, r) to guarantee
        monotone ordering.  Quantile crossing can occur at extreme values
        outside the calibration range.

    Returns
    -------
    np.ndarray, shape (T, m, Q)
        Quantile forecasts in original scale.  ``out[t, r, q]`` is the
        predicted quantile ``qmodel.levels[q]`` for time step ``t``, region ``r``.
    """
    if point_preds.ndim != 2:
        raise ValueError("point_preds must be 2-D (T, m)")
    T, m = point_preds.shape
    Q = len(qmodel.levels)

    out = np.zeros((T, m, Q), dtype=float)
    for q_idx in range(Q):
        slopes     = qmodel.coefficients[q_idx, :, 0]  # (m,)
        intercepts = qmodel.coefficients[q_idx, :, 1]  # (m,)
        out[:, :, q_idx] = point_preds * slopes + intercepts

    if enforce_monotone:
        out.sort(axis=2)

    return out


def compute_wis(
    quantile_preds: np.ndarray,
    true_vals: np.ndarray,
    quantile_levels: list[float],
) -> dict:
    """Compute WIS and its sharpness/calibration decomposition.

    Follows Bracher et al. (2021), Equation (2):

        WIS = (1 / (K + 0.5)) * [0.5 * |y - m̂|
              + Σ_{k=1}^K (α_k/2) * IS_{α_k}(l_k, u_k, y)]

    where IS_α(l, u, y) = (u - l) + (2/α)*max(l-y, 0) + (2/α)*max(y-u, 0)
    and K is the number of central prediction intervals identified from the
    symmetric pairs in ``quantile_levels``.

    Parameters
    ----------
    quantile_preds : np.ndarray, shape (T, m, Q)
        Quantile forecasts (monotone non-decreasing along Q axis).
    true_vals : np.ndarray, shape (T, m)
        Observed ground-truth values.
    quantile_levels : list of float
        Quantile levels corresponding to the Q axis of ``quantile_preds``,
        sorted ascending.

    Returns
    -------
    dict with keys:
        wis_mean        -- float, mean WIS over all (t, r)
        wis_per_region  -- np.ndarray (m,), mean WIS per region
        wis_per_week    -- np.ndarray (T,), mean WIS per time step
        sharpness       -- float, mean interval-width component
        calibration     -- float, mean undercoverage penalty component
        n_levels        -- int, Q
        n_pairs         -- int, K (symmetric interval pairs found)
    """
    levels = sorted(quantile_levels)
    Q = len(levels)
    T, m = true_vals.shape

    if quantile_preds.shape != (T, m, Q):
        raise ValueError(
            f"quantile_preds shape {quantile_preds.shape} expected ({T}, {m}, {Q})"
        )

    diffs = np.diff(quantile_preds, axis=2)
    if not np.all(diffs >= -1e-9):
        max_violation = float(np.min(diffs))
        if max_violation < -1.0:
            raise ValueError(
                f"quantile_preds has large monotonicity violations "
                f"(max decrease: {max_violation:.2f}); sort before calling compute_wis"
            )
        quantile_preds = quantile_preds.copy()
        quantile_preds.sort(axis=2)

    # Locate median index
    median_idx = int(np.argmin([abs(l - 0.5) for l in levels]))

    # Identify symmetric pairs (τ, 1-τ) for τ < 0.5
    pairs: list[tuple[int, int, float]] = []  # (lower_idx, upper_idx, alpha)
    for i, tau in enumerate(levels):
        if tau >= 0.5:
            break
        partner = 1.0 - tau
        dists = [abs(l - partner) for l in levels]
        j = int(np.argmin(dists))
        if abs(levels[j] - partner) < 1e-9:
            pairs.append((i, j, 2.0 * tau))

    K = len(pairs)
    norm = K + 0.5

    # Accumulate WIS components: shape (T, m)
    wis   = np.zeros((T, m), dtype=float)
    sharp = np.zeros((T, m), dtype=float)
    calib = np.zeros((T, m), dtype=float)

    # Median absolute error
    med_ae = np.abs(true_vals - quantile_preds[:, :, median_idx])
    wis   += 0.5 * med_ae
    calib += 0.5 * med_ae

    for lower_idx, upper_idx, alpha in pairs:
        l_pred = quantile_preds[:, :, lower_idx]
        u_pred = quantile_preds[:, :, upper_idx]
        w = alpha / 2.0

        width     = u_pred - l_pred
        undershot = np.maximum(l_pred - true_vals, 0.0)
        overshot  = np.maximum(true_vals - u_pred, 0.0)

        interval_score = width + (2.0 / alpha) * (undershot + overshot)
        wis   += w * interval_score
        sharp += w * width
        calib += undershot + overshot

    wis   /= norm
    sharp /= norm
    calib /= norm

    return {
        "wis_mean":       float(np.mean(wis)),
        "wis_per_region": np.mean(wis, axis=0),
        "wis_per_week":   np.mean(wis, axis=1),
        "sharpness":      float(np.mean(sharp)),
        "calibration":    float(np.mean(calib)),
        "n_levels":       Q,
        "n_pairs":        K,
    }


def compute_relative_wis(
    model_wis_per_region: np.ndarray,
    baseline_wis_per_region: np.ndarray,
) -> float:
    """Compute relative WIS following FluSight methodology.

    Uses geometric mean across regions to reduce impact of magnitude
    differences across jurisdictions, matching the CDC FluSight
    2024-25 evaluation protocol.

    Parameters
    ----------
    model_wis_per_region : np.ndarray, shape (m,)
        Mean WIS per region for the model being evaluated.
    baseline_wis_per_region : np.ndarray, shape (m,)
        Mean WIS per region for the baseline model.

    Returns
    -------
    float
        Relative WIS.  Values < 1 indicate the model outperforms
        the baseline; values > 1 indicate worse performance.
    """
    if model_wis_per_region.shape != baseline_wis_per_region.shape:
        raise ValueError(
            f"Shape mismatch: model {model_wis_per_region.shape} "
            f"vs baseline {baseline_wis_per_region.shape}"
        )
    eps = 1e-12
    log_ratio = np.mean(
        np.log(model_wis_per_region + eps)
        - np.log(baseline_wis_per_region + eps)
    )
    return float(np.exp(log_ratio))


def wis_summary(wis_result: dict, model_label: str) -> dict:
    """Extract a flat summary row from a compute_wis result dict.

    Suitable for building comparison DataFrames.
    """
    return {
        "Model":       model_label,
        "WIS":         round(wis_result["wis_mean"], 2),
        "Sharpness":   round(wis_result["sharpness"], 2),
        "Calibration": round(wis_result["calibration"], 2),
    }
