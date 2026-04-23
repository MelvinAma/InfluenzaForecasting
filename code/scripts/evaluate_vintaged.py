"""Evaluate ColaGNN using preliminary (vintaged) data as model input.

For each reference date, creates a hybrid data file where recent weeks use
preliminary values (as they were reported at the time) while earlier weeks
use final values. This simulates what a FluSight participant would experience.

Compares resulting predictions against the final ground truth to measure
the impact of data vintage on forecast quality.

AGGREGATION NOTE: This script computes Rel.WIS per (week, horizon) pair and
then averages the per-week values (mean-of-ratios). The main FluSight ranking
comparison pools all weeks first, then computes a single Rel.WIS (ratio-of-means
with geometric mean over regions — CDC protocol). These give different absolute
values. The DEGRADATION estimate is valid (same method applied to both "final"
and "preliminary"), but absolute Rel.WIS from this script should NOT be compared
against the main ranking result.

Usage:
    python scripts/evaluate_vintaged.py
"""
import io
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tuning.runner import FLUSIGHT_LEVELS
from src.eval.evaluate import inference_colagnn
from src.eval.quantile import compute_wis, compute_relative_wis
from src.eval.flusight_hub import _parse_submission
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
VINTAGE_DIR = PROJECT_ROOT / "data" / "vintages"

with open(PROJECT_ROOT / "results" / "colagnn_us_best_config_v2.json") as f:
    MODEL_CFG = json.load(f)["config"]

ALIGNMENT = {2: 1, 3: 2, 4: 3}

BL_DIR = PROJECT_ROOT / "data" / "FluSight-forecast-hub" / "model-output" / "FluSight-baseline"


def load_vintage_target(ref_date):
    path = VINTAGE_DIR / f"target_{ref_date}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["location"] = df["location"].astype(str).str.zfill(2)
    return df


def create_hybrid_data(ref_date, final_data, final_dates, vintage_target):
    """Create a data file with preliminary values for dates in the vintage."""
    vintage_dates = set(vintage_target["date"].unique())
    hybrid = final_data.copy()

    replaced = 0
    for row_idx, date in enumerate(final_dates):
        if date not in vintage_dates:
            continue
        day_data = vintage_target[vintage_target["date"] == date]
        if day_data.empty:
            continue
        for col_idx, abbr in enumerate(STATES):
            fips = ABBR_TO_FIPS[abbr]
            match = day_data[day_data["location"] == fips]
            if not match.empty:
                new_val = float(match["value"].values[0])
                if np.isnan(new_val):
                    continue
                old_val = hybrid[row_idx, col_idx]
                if old_val != new_val:
                    hybrid[row_idx, col_idx] = new_val
                    replaced += 1

    return hybrid, replaced


def run_vintaged_inference(hybrid_data, our_h, date_index_path, tmp_dir):
    """Run inference using hybrid data file."""
    data_path = Path(tmp_dir) / f"hybrid_h{our_h}.txt"
    np.savetxt(data_path, hybrid_data, delimiter=",", fmt="%.1f")

    cfg = {
        "data_path": str(data_path),
        "adj_path": "src/data/state-adj-51.txt",
        "date_index_path": "src/data/date_index.csv",
        "date_format": "date",
        "window": 20,
        "horizon": our_h,
        "split_mode": "season",
        "test_seasons": ["2025/26"],
        "val_seasons": ["2024/25"],
        "exclude_seasons": ["2020/21"],
        "train_ratio": 0.6,
        "val_ratio": 0.2,
    }
    cfg_path = Path(tmp_dir) / f"cfg_h{our_h}.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    all_preds = []
    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"colagnn_2526_h{our_h}_seed{seed}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pred, _ = inference_colagnn(
            ckpt["best_state"], MODEL_CFG, str(cfg_path), split="test", device=DEVICE,
        )
        all_preds.append(pred)

    avg = np.mean(all_preds, axis=0)
    avg.sort(axis=2)
    return avg


def main():
    date_df = pd.read_csv(
        PROJECT_ROOT / "src" / "data" / "date_index.csv",
        header=None, names=["idx", "date"], parse_dates=["date"],
    )
    final_dates = list(date_df["date"])
    idx_to_date = dict(zip(date_df["idx"], date_df["date"]))

    iso_years, iso_weeks = _load_date_index(
        PROJECT_ROOT / "src" / "data" / "date_index.csv", "date"
    )
    seasons = assign_seasons(iso_years, iso_weeks)

    final_data = np.loadtxt(
        PROJECT_ROOT / "src" / "data" / "state_flu_admissions.txt", delimiter=","
    )

    final_target = pd.read_csv(
        PROJECT_ROOT / "data" / "FluSight-forecast-hub" / "target-data" / "target-hospital-admissions.csv",
        parse_dates=["date"],
    )
    final_target["location"] = final_target["location"].astype(str).str.zfill(2)

    vintage_files = sorted(VINTAGE_DIR.glob("target_*.csv"))
    ref_dates = [f.stem.replace("target_", "") for f in vintage_files]

    print(f"Device: {DEVICE}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Vintages available: {len(ref_dates)}")
    print()

    all_results = []

    for ref_date in ref_dates:
        ref_dt = pd.Timestamp(ref_date)
        vintage_target = load_vintage_target(ref_date)
        if vintage_target is None:
            continue

        hybrid, n_replaced = create_hybrid_data(ref_date, final_data, final_dates, vintage_target)
        vintage_max = vintage_target["date"].max()

        print(f"\n{'='*60}")
        print(f"Reference date: {ref_date} (vintage through {vintage_max.date()})")
        print(f"  Values replaced: {n_replaced}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            for our_h, fs_h in ALIGNMENT.items():
                target_end = ref_dt + timedelta(days=7 * fs_h)
                target_end_str = target_end.strftime("%Y-%m-%d")

                target_idx = None
                for i, d in enumerate(final_dates):
                    if d.date() == target_end.date():
                        target_idx = i
                        break

                if target_idx is None or target_idx >= len(final_data):
                    continue

                P = 20
                min_idx = P + our_h - 1
                test_indices = [i for i in range(min_idx, len(seasons)) if seasons[i] == "2025/26"]
                test_dates_list = [idx_to_date[i] for i in test_indices]

                t_idx = None
                for t, d in enumerate(test_dates_list):
                    if d.date() == target_end.date():
                        t_idx = t
                        break

                if t_idx is None:
                    continue

                preds = run_vintaged_inference(hybrid, our_h, "src/data/date_index.csv", tmp_dir)

                gt = np.zeros((1, len(STATES)))
                for r, abbr in enumerate(STATES):
                    fips = ABBR_TO_FIPS[abbr]
                    loc_gt = final_target[(final_target["date"] == target_end_str) &
                                          (final_target["location"] == fips)]
                    if loc_gt.empty:
                        break
                    gt[0, r] = float(loc_gt["value"].values[0])
                else:
                    qpred = preds[t_idx:t_idx+1, :, :]
                    wis = compute_wis(qpred, gt, FLUSIGHT_LEVELS)

                    bl_csv = BL_DIR / f"{ref_date}-FluSight-baseline.csv"
                    bl_rel = None
                    if bl_csv.exists():
                        bl_frame = _parse_submission(bl_csv, fs_h, STATES, FLUSIGHT_LEVELS)
                        if bl_frame is not None:
                            bl_qpred = bl_frame[np.newaxis, :, :]
                            bl_result = compute_wis(bl_qpred, gt, FLUSIGHT_LEVELS)
                            bl_rel = compute_relative_wis(wis["wis_per_region"], bl_result["wis_per_region"])

                    median_idx = len(FLUSIGHT_LEVELS) // 2
                    mae = float(np.mean(np.abs(gt[0] - qpred[0, :, median_idx])))

                    result = {
                        "reference_date": ref_date,
                        "horizon": fs_h,
                        "target_end_date": target_end_str,
                        "wis": round(wis["wis_mean"], 2),
                        "rel_wis": round(bl_rel, 3) if bl_rel else None,
                        "median_mae": round(mae, 2),
                        "data_type": "preliminary",
                    }
                    all_results.append(result)
                    rel_str = f", rel.WIS={result['rel_wis']}" if result["rel_wis"] else ""
                    print(f"  h={fs_h}: WIS={result['wis']}, MAE={result['median_mae']}{rel_str}")

    if not all_results:
        print("\nNo results generated.")
        return

    vintaged_df = pd.DataFrame(all_results)
    final_df = pd.read_csv(PROJECT_ROOT / "results" / "weekly_submissions" / "scoring_results.csv")

    print(f"\n{'='*60}")
    print("COMPARISON: FINAL vs PRELIMINARY DATA")
    print(f"{'='*60}")

    for h in sorted(vintaged_df["horizon"].unique()):
        v_h = vintaged_df[vintaged_df["horizon"] == h]
        f_h = final_df[final_df["horizon"] == h]

        common_refs = set(v_h["reference_date"]) & set(f_h["reference_date"])
        if not common_refs:
            continue

        v_common = v_h[v_h["reference_date"].isin(common_refs)]
        f_common = f_h[f_h["reference_date"].isin(common_refs)]

        v_wis = v_common["wis"].mean()
        f_wis = f_common["wis"].mean()
        v_rel = v_common["rel_wis"].dropna().mean()
        f_rel = f_common["rel_wis"].dropna().mean()

        print(f"\n  h={h} ({len(common_refs)} weeks):")
        print(f"    Final data:       WIS={f_wis:.1f}, Rel.WIS={f_rel:.3f}")
        print(f"    Preliminary data: WIS={v_wis:.1f}, Rel.WIS={v_rel:.3f}")
        print(f"    Degradation:      WIS +{((v_wis-f_wis)/f_wis*100):.1f}%, "
              f"Rel.WIS +{((v_rel-f_rel)/f_rel*100):.1f}%")

    v_all = vintaged_df["rel_wis"].dropna()
    f_common_all = final_df[final_df["reference_date"].isin(set(vintaged_df["reference_date"]))]
    f_all = f_common_all["rel_wis"].dropna()

    print(f"\n  Overall:")
    print(f"    Final data Rel.WIS:       {f_all.mean():.3f}")
    print(f"    Preliminary data Rel.WIS: {v_all.mean():.3f}")
    print(f"    Degradation:              +{((v_all.mean()-f_all.mean())/f_all.mean()*100):.1f}%")

    out_path = VINTAGE_DIR / "vintaged_scoring_results.csv"
    vintaged_df.to_csv(out_path, index=False)
    print(f"\nDetailed results: {out_path}")

    summary = {
        "final_mean_rel_wis": round(f_all.mean(), 3),
        "prelim_mean_rel_wis": round(v_all.mean(), 3),
        "degradation_pct": round((v_all.mean() - f_all.mean()) / f_all.mean() * 100, 1),
        "n_weeks": len(set(vintaged_df["reference_date"])),
        "per_horizon": {},
    }
    for h in sorted(vintaged_df["horizon"].unique()):
        v_h = vintaged_df[(vintaged_df["horizon"] == h) & vintaged_df["rel_wis"].notna()]
        f_h = final_df[(final_df["horizon"] == h) &
                       final_df["reference_date"].isin(set(v_h["reference_date"])) &
                       final_df["rel_wis"].notna()]
        summary["per_horizon"][f"h{h}"] = {
            "final_rel_wis": round(f_h["rel_wis"].mean(), 3),
            "prelim_rel_wis": round(v_h["rel_wis"].mean(), 3),
        }

    with open(VINTAGE_DIR / "vintage_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
