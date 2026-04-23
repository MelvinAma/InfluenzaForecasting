"""Generate retroactive pseudo-submissions for all valid 2025/26 reference dates.

Runs inference once per horizon (11 seeds x 3 horizons = 33 model runs),
then formats FluSight-compliant CSVs for every valid Saturday reference date.

Usage:
    python scripts/generate_retroactive.py
"""
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

ALIGNMENT = {2: 1, 3: 2, 4: 3}


def run_inference_all(our_h, cfg_path):
    all_preds = []
    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"colagnn_2526_h{our_h}_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: missing {ckpt_path.name}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pred, _ = inference_colagnn(
            ckpt["best_state"], MODEL_CFG, cfg_path, split="test", device=DEVICE,
        )
        all_preds.append(pred)

    if not all_preds:
        raise RuntimeError(f"No checkpoints for h={our_h}")

    avg = np.mean(all_preds, axis=0)
    avg.sort(axis=2)
    return avg


def format_single_submission(reference_date, predictions_by_horizon, test_dates_by_horizon):
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
            continue

        for r, abbr in enumerate(STATES):
            fips = ABBR_TO_FIPS[abbr]
            for q, level in enumerate(FLUSIGHT_LEVELS):
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
    date_idx_path = PROJECT_ROOT / "src" / "data" / "date_index.csv"
    date_df = pd.read_csv(date_idx_path, header=None, names=["idx", "date"], parse_dates=["date"])
    idx_to_date = dict(zip(date_df["idx"], date_df["date"]))
    max_data_date = date_df["date"].max()
    iso_years, iso_weeks = _load_date_index(date_idx_path, "date")
    seasons = assign_seasons(iso_years, iso_weeks)

    print(f"Data through: {max_data_date.date()}")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {len(SEEDS)}")
    print()

    print("Step 1: Running inference (once per horizon)...")
    predictions_by_horizon = {}
    test_dates_by_horizon = {}

    P = 20
    for our_h, fs_h in ALIGNMENT.items():
        cfg_path = str(PROJECT_ROOT / "src" / "configs" / f"us_colagnn_2526_h{our_h}.json")
        print(f"  h={our_h} (FluSight h={fs_h})...")

        min_idx = P + our_h - 1
        test_indices = [i for i in range(min_idx, len(seasons)) if seasons[i] == "2025/26"]
        test_dates = pd.DatetimeIndex([idx_to_date[i] for i in test_indices])

        preds = run_inference_all(our_h, cfg_path)
        predictions_by_horizon[our_h] = preds
        test_dates_by_horizon[our_h] = test_dates
        print(f"    {preds.shape[0]} test weeks, range: {test_dates[0].date()} to {test_dates[-1].date()}")

    test_date_set = set()
    for td in test_dates_by_horizon.values():
        test_date_set.update(d.date() for d in td)

    print()
    print("Step 2: Finding valid reference dates...")
    bl_dir = PROJECT_ROOT / "data" / "FluSight-forecast-hub" / "model-output" / "FluSight-baseline"
    bl_files = sorted(bl_dir.glob("*.csv"))
    bl_dates = set()
    for f in bl_files:
        try:
            d = datetime.strptime(f.stem.split("-FluSight")[0], "%Y-%m-%d").date()
            bl_dates.add(d)
        except ValueError:
            pass

    valid_refs = []
    for bl_date in sorted(bl_dates):
        ref = datetime(bl_date.year, bl_date.month, bl_date.day)
        horizons_available = 0
        for fs_h in [1, 2, 3]:
            target = (ref + timedelta(days=7 * fs_h)).date()
            if target in test_date_set and target <= max_data_date.date():
                horizons_available += 1
        if horizons_available > 0 and bl_date >= datetime(2025, 11, 1).date():
            valid_refs.append((ref, horizons_available))

    print(f"  Found {len(valid_refs)} valid reference dates")
    for ref, n_h in valid_refs:
        print(f"    {ref.date()} ({n_h} horizon{'s' if n_h > 1 else ''})")

    print()
    print("Step 3: Generating submissions...")
    generated = 0
    for ref, n_h in valid_refs:
        sub = format_single_submission(ref, predictions_by_horizon, test_dates_by_horizon)
        if sub.empty:
            print(f"  {ref.date()}: no predictions (skipped)")
            continue

        out_path = SUBMISSION_DIR / f"{ref.strftime('%Y-%m-%d')}-MEx-ColaGNN.csv"
        sub.to_csv(out_path, index=False)

        n_horizons = sub["horizon"].nunique()
        medians = sub[sub["output_type_id"] == 0.5]["value"]
        print(f"  {ref.date()}: {n_horizons} horizon(s), {len(sub)} rows, "
              f"median range [{medians.min():.0f}, {medians.max():.0f}]")
        generated += 1

    print(f"\nGenerated {generated} pseudo-submissions in {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()
