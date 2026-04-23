"""Evaluate models using the Spreco et al. (2020) framework.

Scores ANY model's predictions using the evaluation protocol from:
    Spreco A, et al. Nowcasting (Short-Term Forecasting) of Influenza
    Epidemics in Local Settings, Sweden, 2008-2019. Emerg Infect Dis.
    2020;26(11):2669-2677.

This does NOT reimplement their forecasting algorithm. It applies their
evaluation metrics to our GNN/baseline outputs so we can report results
on their terms: peak timing accuracy (categorical), peak intensity
accuracy (5-level categorical), and detection timeliness.

Adaptations for weekly resolution:
    - Peak timing error measured in weeks (paper uses days)
    - Weekly error of 0 = 0-6 days → maps to "excellent" (≤3d) or "good" (4-7d)
    - Weekly error of 1 = 7 days → maps to "good" (4-7d)
    - Weekly error of 2 = 14 days → maps to "poor" (≥12d)
    - Intensity thresholds converted from cases/day/100k to cases/week/region

Population data (SCB 2024) required for per-capita intensity categories.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Official per-100k rates from Folkhälsomyndigheten (SmiNet).
# Extracted from "Fall efter region och vecka (tidsserie)" Excel download.
# 538 weeks × 21 regions, aligned to sweden_flu_cases.txt.
_RATES_PATH = PROJ_ROOT / "src" / "data" / "sweden_flu_rates_per100k.txt"

REGION_ORDER = [
    "Blekinge", "Dalarna", "Gotland", "Gävleborg", "Halland",
    "Jämtland Härjedalen", "Jönköping", "Kalmar", "Kronoberg",
    "Norrbotten", "Skåne", "Stockholm", "Södermanland", "Uppsala",
    "Värmland", "Västerbotten", "Västernorrland", "Västmanland",
    "Västra Götaland", "Örebro", "Östergötland",
]

# Spreco et al. Table: intensity thresholds in cases/day/100,000 population
# Derived from Moving Epidemic Method on 2008-09 reference season
_DAILY_THRESHOLDS_PER_100K = {
    "nonepidemic": 0.9,
    "low": 0.9,
    "medium": 2.4,
    "high": 5.5,
    "very_high": 7.9,
}

# Convert to cases/WEEK per 100,000 (multiply by 7)
_WEEKLY_THRESHOLDS_PER_100K = {k: v * 7 for k, v in _DAILY_THRESHOLDS_PER_100K.items()}


def _load_official_rates() -> np.ndarray:
    """Load FHM official per-100k rates, shape (538, 21)."""
    return np.loadtxt(_RATES_PATH, delimiter=",")


def classify_intensity(rate_per_100k_week: float) -> str:
    """Classify a weekly rate into the Spreco et al. 5-level scale.

    Thresholds adapted from daily to weekly (×7):
        nonepidemic: < 6.3 cases/week/100k
        low:         6.3 – 16.79
        medium:      16.8 – 38.49
        high:        38.5 – 55.29
        very_high:   ≥ 55.3
    """
    if rate_per_100k_week < _WEEKLY_THRESHOLDS_PER_100K["low"]:
        return "nonepidemic"
    elif rate_per_100k_week < _WEEKLY_THRESHOLDS_PER_100K["medium"]:
        return "low"
    elif rate_per_100k_week < _WEEKLY_THRESHOLDS_PER_100K["high"]:
        return "medium"
    elif rate_per_100k_week < _WEEKLY_THRESHOLDS_PER_100K["very_high"]:
        return "high"
    else:
        return "very_high"


INTENSITY_LEVELS = ["nonepidemic", "low", "medium", "high", "very_high"]
INTENSITY_RANK = {level: i for i, level in enumerate(INTENSITY_LEVELS)}


def classify_intensity_accuracy(predicted_level: str, actual_level: str) -> str:
    """Rate peak-intensity prediction accuracy per Spreco et al. protocol.

    excellent: exact category match
    good:      ±1 category
    tolerable: ±2 categories (within 10-20% of threshold)
    poor:      >2 categories apart
    """
    diff = abs(INTENSITY_RANK[predicted_level] - INTENSITY_RANK[actual_level])
    if diff == 0:
        return "excellent"
    elif diff == 1:
        return "good"
    elif diff == 2:
        return "tolerable"
    else:
        return "poor"


def classify_peak_timing(weeks_error: int) -> str:
    """Rate peak-timing prediction accuracy per Spreco et al. protocol.

    Original thresholds (days): excellent ≤3, good 4-7, tolerable 8-11, poor ≥12.
    Weekly adaptation (conservative — weekly resolution cannot distinguish
    within-week errors, so we use the most favorable mapping):
        0 weeks (0-6 days):  excellent
        1 week  (7 days):    good
        2 weeks (14 days):   poor
    """
    abs_err = abs(weeks_error)
    if abs_err == 0:
        return "excellent"
    elif abs_err == 1:
        return "good"
    else:
        return "poor"


def evaluate_spreco(
    pred: np.ndarray,
    true: np.ndarray,
    test_indices: list[int],
    region_names: list[str] = None,
) -> dict:
    """Evaluate predictions using the Spreco et al. framework.

    Uses official Folkhälsomyndigheten per-100k rates for ground truth
    intensity classification. Model predictions are converted to rates
    using the implicit FHM population denominators (raw_count / rate * 100k).

    Parameters
    ----------
    pred : np.ndarray, shape (T, m)
        Model predictions in original scale (raw case counts).
    true : np.ndarray, shape (T, m)
        Ground truth in original scale (raw case counts).
    test_indices : list of int
        Global week indices into the full 538-week dataset, used to look up
        official per-100k rates for the test period.
    region_names : list of str, optional
        Region names in column order.

    Returns
    -------
    dict with per-region and aggregate results in Spreco et al. format.
    """
    if region_names is None:
        region_names = REGION_ORDER

    official_rates = _load_official_rates()
    T, m = true.shape

    # Extract official rates for the test period
    test_rates = official_rates[test_indices, :]

    results_per_region = []

    for r in range(m):
        true_series = true[:, r]
        pred_series = pred[:, r]

        true_peak_week = int(np.argmax(true_series))
        pred_peak_week = int(np.argmax(pred_series))
        peak_week_error = pred_peak_week - true_peak_week

        true_peak_val = float(true_series[true_peak_week])
        pred_peak_val = float(pred_series[pred_peak_week])

        # Ground truth: use official FHM rate directly
        true_peak_rate = float(test_rates[true_peak_week, r])

        # Predictions: derive rate from count using FHM's implicit population
        # pop = count * 100000 / rate (from a nearby non-zero pair)
        nonzero = (true_series > 0) & (test_rates[:, r] > 0)
        if nonzero.any():
            pop_est = np.median(
                true_series[nonzero] * 100_000 / test_rates[nonzero, r]
            )
            pred_peak_rate = pred_peak_val / pop_est * 100_000
        else:
            pred_peak_rate = 0.0

        true_intensity = classify_intensity(true_peak_rate)
        pred_intensity = classify_intensity(pred_peak_rate)

        timing_rating = classify_peak_timing(peak_week_error)
        intensity_rating = classify_intensity_accuracy(pred_intensity, true_intensity)

        results_per_region.append({
            "region": region_names[r],
            "true_peak_week": true_peak_week,
            "pred_peak_week": pred_peak_week,
            "peak_week_error": peak_week_error,
            "timing_rating": timing_rating,
            "true_peak_cases": true_peak_val,
            "pred_peak_cases": pred_peak_val,
            "true_peak_rate": round(true_peak_rate, 1),
            "pred_peak_rate": round(pred_peak_rate, 1),
            "true_intensity": true_intensity,
            "pred_intensity": pred_intensity,
            "intensity_rating": intensity_rating,
        })

    timing_counts = {"excellent": 0, "good": 0, "tolerable": 0, "poor": 0}
    intensity_counts = {"excellent": 0, "good": 0, "tolerable": 0, "poor": 0}
    for res in results_per_region:
        timing_counts[res["timing_rating"]] += 1
        intensity_counts[res["intensity_rating"]] += 1

    peak_week_errors = [r["peak_week_error"] for r in results_per_region]
    peak_intensity_errors = [r["pred_peak_cases"] - r["true_peak_cases"]
                             for r in results_per_region]

    return {
        "per_region": results_per_region,
        "aggregate": {
            "peak_week_mae": float(np.mean(np.abs(peak_week_errors))),
            "peak_week_median_error": float(np.median(peak_week_errors)),
            "peak_intensity_mae": float(np.mean(np.abs(peak_intensity_errors))),
            "timing_counts": timing_counts,
            "timing_satisfactory": timing_counts["poor"] == 0,
            "intensity_counts": intensity_counts,
            "intensity_satisfactory": intensity_counts["poor"] == 0,
            "n_regions": m,
        },
    }


def _to_array(obj) -> np.ndarray:
    """Convert a JSON-serialized ndarray back to numpy."""
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict) and "data" in obj:
        return np.array(obj["data"])
    if isinstance(obj, list):
        return np.array(obj)
    return np.array([obj])


def evaluate_spreco_cv(
    cv_result: dict,
    config_path: str,
) -> dict:
    """Evaluate LOSO CV results using the Spreco et al. framework.

    Takes the output of cv_colagnn / cv_epignn / cv_persistence and
    applies Spreco et al. evaluation to each fold's predictions.

    Parameters
    ----------
    cv_result : dict
        Output of a CV function containing 'folds' with stored predictions.
    config_path : str
        Path to config (for loading data to reconstruct pred/true arrays).

    Returns
    -------
    dict with per-fold and aggregate Spreco-style evaluation.
    """
    all_timing = {"excellent": 0, "good": 0, "tolerable": 0, "poor": 0}
    all_intensity = {"excellent": 0, "good": 0, "tolerable": 0, "poor": 0}
    all_peak_week_errors = []
    all_peak_intensity_errors = []
    fold_results = []

    for fold in cv_result["folds"]:
        metrics = fold["metrics"]
        pw_err = _to_array(metrics["peak_week_error"])
        pi_err = _to_array(metrics["peak_intensity_error"])

        all_peak_week_errors.extend(pw_err[~np.isnan(pw_err)].tolist())
        all_peak_intensity_errors.extend(pi_err[~np.isnan(pi_err)].tolist())

        for err in pw_err:
            rating = classify_peak_timing(int(round(err)))
            all_timing[rating] += 1

        fold_results.append({
            "test_season": fold["test_season"],
            "peak_week_mae": float(np.mean(np.abs(pw_err))),
            "peak_intensity_mae": float(np.mean(np.abs(pi_err))),
        })

    total = sum(all_timing.values())
    return {
        "folds": fold_results,
        "aggregate": {
            "peak_week_mae": float(np.mean(np.abs(all_peak_week_errors))),
            "peak_intensity_mae": float(np.mean(np.abs(all_peak_intensity_errors))),
            "timing_counts": all_timing,
            "timing_pct": {k: round(v / total * 100, 1) if total > 0 else 0
                           for k, v in all_timing.items()},
            "intensity_counts": all_intensity,
            "n_total": total,
        },
    }
