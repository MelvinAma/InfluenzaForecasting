"""Score saved pseudo-submissions against actual outcomes and FluSight models.

Reads all saved CSVs from results/weekly_submissions/, compares against
ground truth from the FluSight target data, and ranks against FluSight models
that submitted for the same reference dates.

Usage:
    python scripts/score_submissions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.quantile import compute_wis, compute_relative_wis, FLUSIGHT_LEVELS
from src.eval.flusight_hub import evaluate_hub_model, _parse_submission

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
ABBR_TO_FIPS = {v: k for k, v in FIPS_TO_ABBR.items()}

state_idx = pd.read_csv(
    PROJECT_ROOT / "src" / "data" / "state_index.csv",
    header=None, names=["idx", "abbr"],
)
STATES = list(state_idx["abbr"])

SUBMISSION_DIR = PROJECT_ROOT / "results" / "weekly_submissions"
HUB_DIR = PROJECT_ROOT / "data" / "FluSight-forecast-hub"
BL_DIR = HUB_DIR / "model-output" / "FluSight-baseline"


def load_ground_truth():
    """Load ground truth from FluSight target data."""
    target = pd.read_csv(
        HUB_DIR / "target-data" / "target-hospital-admissions.csv",
        parse_dates=["date"],
    )
    target["location"] = target["location"].astype(str).str.zfill(2)
    return target


def score_submission(csv_path, target_df):
    """Score a single pseudo-submission against ground truth."""
    sub = pd.read_csv(csv_path)
    sub["location"] = sub["location"].astype(str).str.zfill(2)
    ref_date = sub["reference_date"].iloc[0]

    results = []
    for h in sorted(sub["horizon"].unique()):
        h_sub = sub[sub["horizon"] == h]
        target_end = h_sub["target_end_date"].iloc[0]

        gt_row = target_df[target_df["date"] == target_end]
        if gt_row.empty:
            results.append({
                "reference_date": ref_date, "horizon": h,
                "target_end_date": target_end, "status": "no_ground_truth",
            })
            continue

        qpred = np.zeros((1, len(STATES), len(FLUSIGHT_LEVELS)))
        gt = np.zeros((1, len(STATES)))
        has_gt = True

        for r, abbr in enumerate(STATES):
            fips = ABBR_TO_FIPS[abbr]
            loc_gt = gt_row[gt_row["location"] == fips]
            if loc_gt.empty:
                has_gt = False
                break
            gt[0, r] = float(loc_gt["value"].values[0])

            loc_sub = h_sub[h_sub["location"] == fips].sort_values("output_type_id")
            if len(loc_sub) == len(FLUSIGHT_LEVELS):
                qpred[0, r, :] = loc_sub["value"].values
            else:
                has_gt = False
                break

        if not has_gt:
            results.append({
                "reference_date": ref_date, "horizon": h,
                "target_end_date": target_end, "status": "incomplete_data",
            })
            continue

        qpred.sort(axis=2)
        wis = compute_wis(qpred, gt, FLUSIGHT_LEVELS)

        bl_csv = BL_DIR / f"{ref_date}-FluSight-baseline.csv"
        bl_wis = None
        bl_rel = None
        if bl_csv.exists():
            bl_frame = _parse_submission(bl_csv, h, STATES, FLUSIGHT_LEVELS)
            if bl_frame is not None:
                bl_qpred = bl_frame[np.newaxis, :, :]
                bl_result = compute_wis(bl_qpred, gt, FLUSIGHT_LEVELS)
                bl_wis = bl_result["wis_mean"]
                bl_rel = compute_relative_wis(wis["wis_per_region"], bl_result["wis_per_region"])

        median_idx = len(FLUSIGHT_LEVELS) // 2
        mae = float(np.mean(np.abs(gt[0] - qpred[0, :, median_idx])))

        results.append({
            "reference_date": ref_date,
            "horizon": h,
            "target_end_date": target_end,
            "status": "scored",
            "wis": round(wis["wis_mean"], 2),
            "sharpness": round(wis["sharpness"], 2),
            "calibration": round(wis["calibration"], 2),
            "median_mae": round(mae, 2),
            "baseline_wis": round(bl_wis, 2) if bl_wis else None,
            "rel_wis": round(bl_rel, 3) if bl_rel else None,
        })

    return results


def main():
    submissions = sorted(SUBMISSION_DIR.glob("*-MEx-ColaGNN.csv"))
    if not submissions:
        print("No submissions found in", SUBMISSION_DIR)
        return

    print(f"Found {len(submissions)} pseudo-submission(s)")
    target_df = load_ground_truth()

    all_results = []
    for csv_path in submissions:
        print(f"\nScoring: {csv_path.name}")
        results = score_submission(csv_path, target_df)
        for r in results:
            status = r["status"]
            if status == "scored":
                bl_str = f", rel.WIS={r['rel_wis']}" if r["rel_wis"] else ""
                print(f"  h={r['horizon']}: WIS={r['wis']}, MAE={r['median_mae']}{bl_str}")
            else:
                print(f"  h={r['horizon']}: {status}")
        all_results.extend(results)

    scored = [r for r in all_results if r["status"] == "scored"]
    if scored:
        df = pd.DataFrame(scored)
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print(f"Weeks scored: {df['reference_date'].nunique()}")
        print(f"Mean WIS: {df['wis'].mean():.2f}")
        print(f"Mean MAE: {df['median_mae'].mean():.2f}")
        rel_scored = df[df["rel_wis"].notna()]
        if not rel_scored.empty:
            print(f"Mean Rel. WIS: {rel_scored['rel_wis'].mean():.3f}")

        out_path = SUBMISSION_DIR / "scoring_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nDetailed results: {out_path}")


if __name__ == "__main__":
    main()
