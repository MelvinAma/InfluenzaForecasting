"""Weekly pseudo-submission pipeline for FluSight 2025/26.

Mimics the exact process a FluSight participant follows each Wednesday:
1. Download latest NHSN hospitalization data
2. Update local data files
3. Run inference with pre-trained models (11 seeds x 3 horizons)
4. Format predictions as FluSight-compliant CSV
5. Save locally (not submitted to hub)

The predictions can later be scored against actual outcomes to evaluate
real-time performance without data vintage advantage.

Usage:
    python scripts/weekly_predict.py                    # auto-detect reference date
    python scripts/weekly_predict.py --ref 2026-03-07   # specific reference date
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tuning.runner import FLUSIGHT_LEVELS
from src.eval.evaluate import inference_colagnn
from src.data.loader import assign_seasons, _load_date_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456, 777, 1, 2, 3, 4, 5, 6, 7]

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

CKPT_DIR = PROJECT_ROOT / "results" / "prospective_2526"
SUBMISSION_DIR = PROJECT_ROOT / "results" / "weekly_submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "results" / "colagnn_us_best_config_v2.json") as f:
    MODEL_CFG = json.load(f)["config"]

QUANTILE_LEVELS = FLUSIGHT_LEVELS
ALIGNMENT = {2: 1, 3: 2, 4: 3}


def update_data_from_hub():
    """Update local data files from FluSight target data.

    Returns the date of the most recent data point added.
    """
    target_path = PROJECT_ROOT / "data" / "FluSight-forecast-hub" / "target-data" / "target-hospital-admissions.csv"
    target = pd.read_csv(target_path, parse_dates=["date"])
    target["location"] = target["location"].astype(str).str.zfill(2)

    date_idx_path = PROJECT_ROOT / "src" / "data" / "date_index.csv"
    di = pd.read_csv(date_idx_path, header=None, names=["idx", "date"], parse_dates=["date"])
    existing_dates = set(di["date"])
    last_idx = di["idx"].max()

    target_dates = sorted(target["date"].unique())
    new_dates = [d for d in target_dates if d not in existing_dates]

    if not new_dates:
        print(f"Data already up to date (through {di['date'].max().date()})")
        return di["date"].max()

    data_path = PROJECT_ROOT / "src" / "data" / "state_flu_admissions.txt"
    with open(data_path, "a") as f_data, open(date_idx_path, "a") as f_idx:
        for i, d in enumerate(sorted(new_dates)):
            day_data = target[target["date"] == d]
            row = []
            for abbr in STATES:
                fips = ABBR_TO_FIPS[abbr]
                match = day_data[day_data["location"] == fips]
                if not match.empty:
                    row.append(f"{float(match['value'].values[0]):.1f}")
                else:
                    row.append("0.0")
            f_data.write(",".join(row) + "\n")
            new_idx = last_idx + 1 + i
            f_idx.write(f"{new_idx},{d.strftime('%Y-%m-%d')}\n")
            print(f"  Added index {new_idx}: {d.strftime('%Y-%m-%d')}")

    print(f"Updated data with {len(new_dates)} new weeks")
    return max(new_dates)


def generate_predictions(our_h, cfg_path):
    """Run inference for all seeds at a given horizon, return averaged quantile predictions."""
    all_preds = []
    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"colagnn_2526_h{our_h}_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: missing checkpoint {ckpt_path.name}, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pred, _ = inference_colagnn(
            ckpt["best_state"], MODEL_CFG, cfg_path, split="test", device=DEVICE,
        )
        all_preds.append(pred)

    if not all_preds:
        raise RuntimeError(f"No checkpoints found for h={our_h}")

    avg = np.mean(all_preds, axis=0)
    avg.sort(axis=2)
    return avg


def format_submission(reference_date, predictions_by_horizon, test_dates_by_horizon):
    """Format predictions as a FluSight-compliant CSV DataFrame."""
    rows = []

    for our_h, fs_h in ALIGNMENT.items():
        test_dates = test_dates_by_horizon[our_h]
        preds = predictions_by_horizon[our_h]

        target_end = reference_date + timedelta(days=7 * fs_h)

        t_idx = None
        for t, d in enumerate(test_dates):
            if d.date() == target_end.date():
                t_idx = t
                break

        if t_idx is None:
            print(f"  h={our_h}: target {target_end.date()} not in test dates, skipping")
            continue

        for r, abbr in enumerate(STATES):
            fips = ABBR_TO_FIPS[abbr]
            for q, level in enumerate(QUANTILE_LEVELS):
                rows.append({
                    "reference_date": reference_date.strftime("%Y-%m-%d"),
                    "horizon": fs_h,
                    "target": "wk inc flu hosp",
                    "target_end_date": target_end.strftime("%Y-%m-%d"),
                    "location": fips,
                    "output_type": "quantile",
                    "output_type_id": round(level, 3),
                    "value": max(0, round(float(preds[t_idx, r, q]))),
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default=None,
                        help="Reference date (YYYY-MM-DD). Default: most recent Wednesday.")
    parser.add_argument("--skip-update", action="store_true",
                        help="Skip data update step")
    args = parser.parse_args()

    if args.ref:
        reference_date = datetime.strptime(args.ref, "%Y-%m-%d")
    else:
        today = datetime.now()
        days_since_sat = (today.weekday() - 5) % 7
        reference_date = today - timedelta(days=days_since_sat)
        reference_date = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"Reference date: {reference_date.date()}")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {len(SEEDS)}")
    print()

    if not args.skip_update:
        print("Step 1: Updating data from FluSight hub...")
        latest = update_data_from_hub()
        print(f"Data through: {latest.date() if hasattr(latest, 'date') else latest}")
    else:
        print("Step 1: Skipping data update")
    print()

    date_idx_path = PROJECT_ROOT / "src" / "data" / "date_index.csv"
    date_df = pd.read_csv(date_idx_path, header=None, names=["idx", "date"], parse_dates=["date"])
    idx_to_date = dict(zip(date_df["idx"], date_df["date"]))
    iso_years, iso_weeks = _load_date_index(date_idx_path, "date")
    seasons = assign_seasons(iso_years, iso_weeks)

    print("Step 2: Generating predictions...")
    predictions_by_horizon = {}
    test_dates_by_horizon = {}

    for our_h, fs_h in ALIGNMENT.items():
        cfg_path = str(PROJECT_ROOT / "src" / "configs" / f"us_colagnn_2526_h{our_h}.json")
        print(f"  h={our_h} (FluSight h={fs_h})...")

        P = 20
        min_idx = P + our_h - 1
        test_indices = [i for i in range(min_idx, len(seasons)) if seasons[i] == "2025/26"]
        test_dates = pd.DatetimeIndex([idx_to_date[i] for i in test_indices])

        preds = generate_predictions(our_h, cfg_path)
        predictions_by_horizon[our_h] = preds
        test_dates_by_horizon[our_h] = test_dates
        print(f"    {preds.shape[0]} test weeks, latest: {test_dates[-1].date()}")

    print()
    print("Step 3: Formatting submission...")
    submission = format_submission(reference_date, predictions_by_horizon, test_dates_by_horizon)

    if submission.empty:
        print("ERROR: No predictions generated for this reference date.")
        print("Target dates may be outside the available data range.")
        return

    out_path = SUBMISSION_DIR / f"{reference_date.strftime('%Y-%m-%d')}-MEx-ColaGNN.csv"
    submission.to_csv(out_path, index=False)

    n_horizons = submission["horizon"].nunique()
    n_locations = submission["location"].nunique()
    n_quantiles = submission["output_type_id"].nunique()
    print(f"  Horizons: {sorted(submission['horizon'].unique())}")
    print(f"  Locations: {n_locations}")
    print(f"  Quantiles: {n_quantiles}")
    print(f"  Total rows: {len(submission)}")
    print(f"  Saved to: {out_path}")

    print()
    print("Step 4: Validation...")
    for h in sorted(submission["horizon"].unique()):
        h_df = submission[submission["horizon"] == h]
        target_date = h_df["target_end_date"].iloc[0]
        medians = h_df[h_df["output_type_id"] == 0.5]["value"]
        print(f"  h={h} (target {target_date}): median pred range [{medians.min():.0f}, {medians.max():.0f}], "
              f"mean={medians.mean():.0f}")

    print(f"\nDone. Submission saved to {out_path}")


if __name__ == "__main__":
    main()
