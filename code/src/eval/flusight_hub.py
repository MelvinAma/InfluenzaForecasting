"""Evaluate all FluSight-forecast-hub submissions on a common overlap window.

Parses raw CSV submissions from the cloned FluSight-forecast-hub repo, computes
WIS for each model on the specified horizons, and produces a ranking table with
Relative WIS following the CDC evaluation protocol.

Usage:
    from src.eval.flusight_hub import evaluate_all_hub_models
    results = evaluate_all_hub_models(
        hub_dir=Path("data/FluSight-forecast-hub"),
        target_dates=TEST_DATES,
        gt=gt_matrix,
        states=STATES,
        horizons=[1],
    )
"""
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

from src.eval.quantile import compute_wis, compute_relative_wis, FLUSIGHT_LEVELS

FIPS_TO_ABBR = {
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


def _parse_submission(csv_path: Path, horizon: int, states: list[str],
                      quantile_levels: list[float]) -> np.ndarray | None:
    """Parse a single FluSight submission CSV into a (m, Q) quantile array.

    Returns None if the file lacks sufficient data for this horizon.
    """
    Q = len(quantile_levels)
    level_to_q = {round(lvl, 4): i for i, lvl in enumerate(quantile_levels)}

    try:
        df = pd.read_csv(csv_path, dtype={"location": str})
    except Exception:
        return None

    df.columns = df.columns.str.lower()
    if "location" not in df.columns:
        return None

    df["location"] = df["location"].str.zfill(2)

    mask = (
        (df["horizon"] == horizon)
        & (df["output_type"].str.lower() == "quantile")
        & (df["location"].isin(FIPS_TO_ABBR))
    )
    df_filt = df[mask].copy()
    if df_filt.empty:
        return None

    df_filt["q_level"] = df_filt["output_type_id"].apply(lambda x: round(float(x), 4))
    df_filt = df_filt[df_filt["q_level"].isin(set(level_to_q))]

    frame = np.full((len(states), Q), np.nan)
    for _, row in df_filt.iterrows():
        abbr = FIPS_TO_ABBR.get(row["location"])
        q_idx = level_to_q.get(row["q_level"])
        if abbr in states and q_idx is not None:
            frame[states.index(abbr), q_idx] = float(row["value"])

    if np.sum(~np.isnan(frame[:, 0])) < len(states) // 2:
        return None

    for r in range(len(states)):
        q_row = frame[r]
        if np.isnan(q_row).any() and not np.isnan(q_row).all():
            known = np.where(~np.isnan(q_row))[0]
            frame[r] = np.interp(np.arange(Q), known, q_row[known])

    frame.sort(axis=1)
    return frame


def evaluate_hub_model(
    model_dir: Path,
    model_name: str,
    target_dates: pd.DatetimeIndex,
    gt: np.ndarray,
    states: list[str],
    horizons: list[int] = None,
    quantile_levels: list[float] = None,
) -> dict | None:
    """Evaluate a single FluSight hub model across specified horizons.

    For each horizon, target_end_date = reference_date + 7 * horizon.
    We look for the submission file at reference_date = target_date - 7 * horizon.

    Returns dict with per-horizon and aggregate WIS, or None if insufficient data.
    """
    if horizons is None:
        horizons = [1]
    if quantile_levels is None:
        quantile_levels = FLUSIGHT_LEVELS

    all_qpred = []
    all_gt = []
    all_scored_keys = []  # MEx Thesis Adaptation: track (horizon, date_idx) per row
    weeks_by_horizon = {}

    for h in horizons:
        qpred_rows, gt_rows, h_keys = [], [], []  # MEx Thesis Adaptation: h_keys

        for t, target_date in enumerate(target_dates):
            if np.isnan(gt[t]).any():
                continue

            ref_date = target_date - timedelta(days=7 * h)
            ref_str = ref_date.strftime("%Y-%m-%d")
            csv_path = model_dir / f"{ref_str}-{model_name}.csv"

            if not csv_path.exists():
                continue

            frame = _parse_submission(csv_path, h, states, quantile_levels)
            if frame is None:
                continue

            qpred_rows.append(frame)
            gt_rows.append(gt[t])
            h_keys.append((h, t))  # MEx Thesis Adaptation

        if not qpred_rows:
            continue

        qpred = np.stack(qpred_rows)
        gt_arr = np.stack(gt_rows)
        all_qpred.append(qpred)
        all_gt.append(gt_arr)
        all_scored_keys.extend(h_keys)  # MEx Thesis Adaptation
        weeks_by_horizon[h] = len(qpred_rows)

    if not all_qpred:
        return None

    combined_qpred = np.concatenate(all_qpred, axis=0)
    combined_gt = np.concatenate(all_gt, axis=0)
    combined_qpred.sort(axis=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wis_result = compute_wis(combined_qpred, combined_gt, quantile_levels)

    return {
        "model": model_name,
        "wis_mean": wis_result["wis_mean"],
        "wis_per_region": wis_result["wis_per_region"],
        "sharpness": wis_result["sharpness"],
        "calibration": wis_result["calibration"],
        "n_weeks_total": combined_qpred.shape[0],
        "weeks_by_horizon": weeks_by_horizon,
        "scored_keys": all_scored_keys,          # MEx Thesis Adaptation
        "combined_qpred": combined_qpred,        # MEx Thesis Adaptation
        "combined_gt": combined_gt,              # MEx Thesis Adaptation
    }


def evaluate_all_hub_models(
    hub_dir: Path,
    target_dates: pd.DatetimeIndex,
    gt: np.ndarray,
    states: list[str],
    horizons: list[int] = None,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Evaluate all FluSight hub models and return a ranked DataFrame.

    Parameters
    ----------
    hub_dir : Path
        Root of the cloned FluSight-forecast-hub repo.
    target_dates : pd.DatetimeIndex
        Week-ending dates for the test period (e.g. 2024/25 season).
    gt : np.ndarray, shape (T, m)
        Ground truth matrix aligned to target_dates and states.
    states : list of str
        State abbreviations in column order of gt.
    horizons : list of int
        Horizons to include (default [1]).
    min_coverage : float
        Minimum fraction of the baseline's evaluation weeks a model must
        cover to be included in ranking (default 0.5).  Prevents models
        with sparse submissions from appearing alongside full-season ones.

    Returns
    -------
    pd.DataFrame sorted by Rel. WIS ascending (best first).
    """
    if horizons is None:
        horizons = [1]

    model_output_dir = hub_dir / "model-output"
    results = []
    baseline_result = None

    model_dirs = sorted(model_output_dir.iterdir())
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        result = evaluate_hub_model(
            model_dir, model_name, target_dates, gt, states, horizons,
        )
        if result is None:
            continue

        results.append(result)
        if model_name == "FluSight-baseline":
            baseline_result = result

    if baseline_result is None:
        raise RuntimeError("FluSight-baseline not found or has insufficient data")

    bl_weeks = baseline_result["n_weeks_total"]
    min_weeks = max(1, int(bl_weeks * min_coverage))

    bl_key_to_idx = {k: i for i, k in enumerate(baseline_result["scored_keys"])}
    bl_qpred = baseline_result["combined_qpred"]
    bl_gt = baseline_result["combined_gt"]
    quantile_levels = FLUSIGHT_LEVELS

    rows = []
    for r in results:
        if r["n_weeks_total"] < min_weeks:
            continue

        # MEx Thesis Adaptation: pairwise overlap for Rel. WIS
        overlap = [
            (mi, bl_key_to_idx[k])
            for mi, k in enumerate(r["scored_keys"])
            if k in bl_key_to_idx
        ]
        if not overlap:
            continue
        model_idx, bl_idx = zip(*overlap)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl_wis_overlap = compute_wis(
                bl_qpred[list(bl_idx)], bl_gt[list(bl_idx)], quantile_levels,
            )
            model_wis_overlap = compute_wis(
                r["combined_qpred"][list(model_idx)],
                r["combined_gt"][list(model_idx)],
                quantile_levels,
            )
        rel_wis = compute_relative_wis(
            model_wis_overlap["wis_per_region"],
            bl_wis_overlap["wis_per_region"],
        )
        # End MEx Thesis Adaptation

        if np.isnan(rel_wis):
            continue
        rows.append({
            "Model": r["model"],
            "WIS": round(r["wis_mean"], 2),
            "Rel. WIS": round(rel_wis, 3),
            "Sharpness": round(r["sharpness"], 2),
            "Calibration": round(r["calibration"], 2),
            "N weeks": r["n_weeks_total"],
            "Overlap": len(overlap),  # MEx Thesis Adaptation
            "Horizons": str(r["weeks_by_horizon"]),
        })

    df = pd.DataFrame(rows).sort_values("Rel. WIS").reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df
