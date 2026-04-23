# MEx Thesis Adaptation
# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis for EpiGNN WIS when the untrained calibration seed
# (seed 456, epoch 4) is excluded from the calibration set.
#
# Seed 456's cal model converged at epoch 4 vs. 30–133 for seeds 42/123,
# suggesting it failed to learn meaningful residuals.  This script tests
# whether excluding it materially changes the reported WIS.
#
# Operates entirely on cached JSON predictions — no model loading or training.
#
# Usage:
#   python src/eval/wis_sensitivity.py
#
# Prerequisites: run notebooks/flusight_compare.ipynb to populate
#   results/flusight_checkpoints/preds_epignn_*.json
# ─────────────────────────────────────────────────────────────────────────────
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.data.loader import _load_date_index, assign_seasons
from src.eval.quantile import (
    FLUSIGHT_LEVELS,
    compute_wis,
    fit_quantile_model,
    predict_quantiles,
)

# FIPS → state abbreviation (standard US 51 jurisdictions, matching the notebook)
FIPS_TO_ABBR: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}
ABBR_TO_FIPS: dict[str, str] = {v: k for k, v in FIPS_TO_ABBR.items()}

_WINDOW = 20
_HORIZON = 1
_MIN_IDX = _WINDOW + _HORIZON - 1  # = 20


def _load_json(path: Path) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f), dtype=float)


def _derive_test_split(project_root: Path) -> tuple[list[int], pd.DatetimeIndex]:
    """Return (test_indices, test_dates) for the 2024/25 season.

    test_indices are row positions in the raw data arrays (state_flu_admissions.txt).
    test_dates is aligned to test_indices, one date per row.
    """
    date_idx_path = project_root / "src" / "data" / "date_index.csv"
    date_df = pd.read_csv(
        date_idx_path, header=None, names=["idx", "date"], parse_dates=["date"]
    )
    idx_to_date = dict(zip(date_df["idx"], date_df["date"]))

    iso_years, iso_weeks = _load_date_index(date_idx_path, "date")
    seasons = assign_seasons(iso_years, iso_weeks)

    n = len(seasons)
    test_indices = [i for i in range(_MIN_IDX, n) if seasons[i] == "2024/25"]
    test_dates = pd.DatetimeIndex([idx_to_date[i] for i in test_indices])
    return test_indices, test_dates


def _build_gt_overlap(
    project_root: Path,
    states: list[str],
    overlap_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Build ground-truth matrix (T_overlap, 51) from the authoritative NHSN target CSV."""
    target_path = (
        project_root / "data" / "flusight_2024_25" / "target-hospital-admissions.csv"
    )
    target_df = pd.read_csv(target_path, parse_dates=["date"])
    target_df["location"] = target_df["location"].astype(str).str.zfill(2)

    gt = np.full((len(overlap_dates), len(states)), np.nan)
    for t, d in enumerate(overlap_dates):
        day_data = target_df[target_df["date"] == d]
        for r, abbr in enumerate(states):
            row = day_data[day_data["location"] == ABBR_TO_FIPS[abbr]]
            if not row.empty:
                gt[t, r] = float(row["value"].values[0])
    return gt


def run_epignn_sensitivity(project_root: Path) -> dict:
    """Sensitivity analysis: exclude untrained cal seed 456 from EpiGNN WIS.

    Uses the 66-row calibration set (seeds 42 + 123) instead of the full
    99-row set.  The QR model is then applied to the same seed-averaged test
    predictions and the same overlap window used in the primary results.

    Parameters
    ----------
    project_root : Path
        Root of the MEx project (parent of src/, results/, data/).

    Returns
    -------
    dict with keys:
        wis_primary     -- float: WIS (seed-avg preds, 2-seed cal)
        wis_std         -- float: std of per-seed WIS scores
        wis_per_seed    -- dict {seed: wis} for seeds 42, 123, 456
        cal_n           -- int: calibration rows used (66)
        comparison_note -- str: gap interpretation relative to primary WIS
    """
    ckpt_dir = project_root / "results" / "flusight_checkpoints"
    seeds = [42, 123, 456]
    required_files = [
        ckpt_dir / "preds_epignn_cal_pred.json",
        ckpt_dir / "preds_epignn_cal_true.json",
        *[ckpt_dir / f"preds_epignn_test_seed{s}.json" for s in seeds],
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cached EpiGNN predictions not found:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun notebooks/flusight_compare.ipynb to generate them."
        )

    # 1. Load cal predictions; slice to rows 0:66 (seeds 42+123 only)
    cal_pred_66 = _load_json(ckpt_dir / "preds_epignn_cal_pred.json")[:66]  # (66, 51)
    cal_true_66 = _load_json(ckpt_dir / "preds_epignn_cal_true.json")[:66]  # (66, 51)

    # 2. Load per-seed test predictions and compute ensemble average
    test_per_seed: dict[int, np.ndarray] = {
        s: _load_json(ckpt_dir / f"preds_epignn_test_seed{s}.json") for s in seeds
    }  # each (33, 51)
    test_avg = np.mean([test_per_seed[s] for s in seeds], axis=0)  # (33, 51)

    # 3. Derive overlap_test_rows: positions in the (33, 51) test array that fall
    #    in the FluSight window (dates >= 2024-11-30 within 2024/25 test season).
    #    Corresponds to TEST weeks with indices 9–32 inclusive (24 weeks).
    _, test_dates = _derive_test_split(project_root)
    flusight_start = pd.Timestamp("2024-11-30")
    overlap_test_rows = [
        pos for pos, d in enumerate(test_dates) if d >= flusight_start
    ]
    overlap_dates = test_dates[overlap_test_rows]

    # 4. Build gt_overlap (T_overlap, 51) from authoritative NHSN target CSV
    states_df = pd.read_csv(
        project_root / "src" / "data" / "state_index.csv",
        header=None, names=["idx", "abbr"],
    )
    states = list(states_df["abbr"])  # ['AK', 'AL', ..., 'WY'], len=51
    gt_overlap = _build_gt_overlap(project_root, states, overlap_dates)

    # Restrict to fully-observed rows for WIS scoring (mirrors the notebook logic)
    valid_local = np.where(~np.isnan(gt_overlap).any(axis=1))[0]
    valid_test = [overlap_test_rows[i] for i in valid_local]
    gt_valid = gt_overlap[valid_local]

    # 5. Fit QR on the 66-row cal set (seeds 42+123 only)
    qmodel = fit_quantile_model(cal_pred_66, cal_true_66, FLUSIGHT_LEVELS, alpha=0.0)

    # 6. WIS on overlap window using ensemble-averaged test predictions
    qpred_avg = predict_quantiles(qmodel, test_avg[valid_test], enforce_monotone=True)
    wis_result = compute_wis(qpred_avg, gt_valid, FLUSIGHT_LEVELS)
    wis_primary = wis_result["wis_mean"]

    # 7. Per-seed WIS using the same QR model (fitted on seeds 42+123 cal)
    wis_per_seed: dict[int, float] = {}
    for s in seeds:
        qp = predict_quantiles(
            qmodel, test_per_seed[s][valid_test], enforce_monotone=True
        )
        wis_per_seed[s] = float(compute_wis(qp, gt_valid, FLUSIGHT_LEVELS)["wis_mean"])
    wis_std = float(np.std(list(wis_per_seed.values())))

    original_wis = 165.01
    gap = abs(wis_primary - original_wis)
    if gap > 10:
        note = (
            f"Gap = {gap:.2f} WIS units (> 10 threshold): seed-sensitivity driven. "
            "Excluding the untrained seed 456 meaningfully changes EpiGNN WIS."
        )
    else:
        note = (
            f"Gap = {gap:.2f} WIS units (< 10 threshold): architecture-driven. "
            "EpiGNN underperformance is not primarily attributable to seed 456."
        )

    return {
        "wis_primary":     wis_primary,
        "wis_std":         wis_std,
        "wis_per_seed":    wis_per_seed,
        "cal_n":           66,
        "comparison_note": note,
    }


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    result = run_epignn_sensitivity(_root)

    ORIGINAL_WIS = 165.01
    ORIGINAL_STD = 27.09

    print("=" * 65)
    print("EpiGNN WIS Sensitivity: Excluding Untrained Cal Seed (456)")
    print("=" * 65)
    print(
        f"Original  EpiGNN WIS (3 seeds pooled): "
        f"{ORIGINAL_WIS:.2f} \u00b1 {ORIGINAL_STD:.2f}"
    )
    print(
        f"Sensitivity EpiGNN WIS (seeds 42+123): "
        f"{result['wis_primary']:.2f} \u00b1 {result['wis_std']:.2f}"
    )
    print()
    print("Per-seed WIS (sensitivity QR model, cal = seeds 42+123):")
    for seed, wis in result["wis_per_seed"].items():
        marker = "  \u2190 excluded from cal" if seed == 456 else ""
        print(f"  Seed {seed:3d}: {wis:.2f}{marker}")
    print()
    print(f"Cal rows : {result['cal_n']}  (2 seeds \u00d7 33 cal weeks)")
    print(f"\nInterpretation: {result['comparison_note']}")
    print("=" * 65)
