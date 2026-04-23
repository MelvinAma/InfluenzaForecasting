"""Evaluate ColaGNN 2024/25 with preliminary vs final NHSN data.

Quantifies the data vintage advantage in our Stage 2b FluSight comparison.
Real FluSight participants only had preliminary data at submission time;
we used final/revised data.

AGGREGATION NOTE: This script computes Rel.WIS per (week, horizon) pair and
then averages the per-week values (mean-of-ratios). The main FluSight ranking
comparison (flusight_compare.ipynb, run_prospective_2526.py) pools all weeks
first, then computes a single Rel.WIS (ratio-of-means with geometric mean
over regions). These give different absolute values:
  - Main (pooled, CDC protocol): Rel.WIS = 0.654
  - This script (per-week mean):  Rel.WIS = 0.621
The DEGRADATION estimate (+8.2%) is valid because both "final" and
"preliminary" use the same per-week method. But the absolute Rel.WIS
from this script should NOT be compared against the main ranking result.

Strategy:
  Phase 1 — Precompute final-data predictions for the full 2024/25 test season
             once per horizon (3 horizons × 11 seeds = 33 forward passes).
  Phase 2 — For each vintage ref_date, replace recent weeks with preliminary
             values and re-run inference (28 ref_dates × 3 horizons × 11 seeds).
  Phase 3 — Compute Rel.WIS for final and preliminary at each ref_date, then
             report degradation.

Usage:
    python scripts/evaluate_vintaged_2425.py
"""
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

CKPT_DIR = PROJECT_ROOT / "results" / "native_quantile_v3"
VINTAGE_DIR = PROJECT_ROOT / "data" / "vintages"
BL_DIR = (PROJECT_ROOT / "data" / "FluSight-forecast-hub"
          / "model-output" / "FluSight-baseline")

ALIGNMENT = {2: 1, 3: 2, 4: 3}   # our_h -> flusight_h

with open(PROJECT_ROOT / "results" / "colagnn_us_best_config_v2.json") as f:
    MODEL_CFG = json.load(f)["config"]

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


def _cfg_path(our_h: int) -> str:
    return str(PROJECT_ROOT / "src" / "configs" / f"us_colagnn_h{our_h}.json")


def _load_ckpt(our_h: int, seed: int):
    path = CKPT_DIR / f"colagnn_nq_h{our_h}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def precompute_final_predictions(date_df) -> dict:
    """Run inference with final NHSN data for all 2024/25 test weeks.

    Returns dict: our_h -> (avg_pred, test_dates)
      avg_pred shape: (T, 51, 23) — ensemble average, quantiles sorted ascending
      test_dates: list of pd.Timestamp, length T
    """
    idx_to_date = dict(zip(date_df["idx"], date_df["date"]))
    iso_years, iso_weeks = _load_date_index(
        PROJECT_ROOT / "src" / "data" / "date_index.csv", "date"
    )
    seasons = assign_seasons(iso_years, iso_weeks)
    P = 20

    result = {}
    for our_h, fs_h in ALIGNMENT.items():
        cfg = _cfg_path(our_h)
        print(f"  h={our_h}: loading {len(SEEDS)} seeds ...", end=" ", flush=True)

        all_preds = []
        for seed in SEEDS:
            ckpt = _load_ckpt(our_h, seed)
            pred, _ = inference_colagnn(
                ckpt["best_state"], MODEL_CFG, cfg, split="test", device=DEVICE,
            )
            all_preds.append(pred)

        avg = np.mean(all_preds, axis=0)
        avg.sort(axis=2)

        min_idx = P + our_h - 1
        test_indices = [i for i in range(min_idx, len(seasons))
                        if seasons[i] == "2024/25"]
        test_dates = [idx_to_date[i] for i in test_indices]

        result[our_h] = {"avg_pred": avg, "test_dates": test_dates}
        print(f"shape={avg.shape}")

    return result


def create_hybrid_data(final_data, final_dates, vintage_target):
    """Replace rows in final_data with preliminary values from vintage_target."""
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
                if not np.isnan(new_val) and hybrid[row_idx, col_idx] != new_val:
                    hybrid[row_idx, col_idx] = new_val
                    replaced += 1
    return hybrid, replaced


def run_hybrid_inference(hybrid_data, our_h, tmp_dir) -> np.ndarray:
    """Run ensemble inference with hybrid data. Returns avg_pred (T, 51, 23)."""
    data_path = Path(tmp_dir) / f"hybrid_h{our_h}.txt"
    np.savetxt(data_path, hybrid_data, delimiter=",", fmt="%.1f")

    cfg_override = {
        "data_path": str(data_path),
        "adj_path": "src/data/state-adj-51.txt",
        "date_index_path": "src/data/date_index.csv",
        "date_format": "date",
        "window": 20,
        "horizon": our_h,
        "split_mode": "season",
        "test_seasons": ["2024/25"],
        "val_seasons": ["2023/24"],
        "exclude_seasons": ["2020/21", "2025/26"],
    }
    cfg_path = Path(tmp_dir) / f"cfg_h{our_h}.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg_override, f)

    all_preds = []
    for seed in SEEDS:
        ckpt = _load_ckpt(our_h, seed)
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

    final_data = np.loadtxt(
        PROJECT_ROOT / "src" / "data" / "state_flu_admissions.txt", delimiter=","
    )

    final_target = pd.read_csv(
        PROJECT_ROOT / "data" / "FluSight-forecast-hub"
        / "target-data" / "target-hospital-admissions.csv",
        parse_dates=["date"],
    )
    final_target["location"] = final_target["location"].astype(str).str.zfill(2)

    # 2024/25 vintage ref dates only
    all_vintage_files = sorted(VINTAGE_DIR.glob("target_*.csv"))
    vintage_ref_dates = []
    for f in all_vintage_files:
        ref_date = f.stem.replace("target_", "")
        ref_dt = pd.Timestamp(ref_date)
        if pd.Timestamp("2024-09-01") <= ref_dt <= pd.Timestamp("2025-06-30"):
            vintage_ref_dates.append(ref_date)

    print(f"Device:          {DEVICE}")
    print(f"Seeds:           {len(SEEDS)}")
    print(f"Vintage weeks:   {len(vintage_ref_dates)}")
    print(f"  ({vintage_ref_dates[0]} to {vintage_ref_dates[-1]})")
    print()

    print("Phase 1: Precomputing final-data predictions ...")
    final_preds = precompute_final_predictions(date_df)
    print()

    print("Phase 2: Running preliminary-data inference per reference date ...")
    all_results = []

    for ref_date in vintage_ref_dates:
        ref_dt = pd.Timestamp(ref_date)

        vintage_path = VINTAGE_DIR / f"target_{ref_date}.csv"
        vintage_target = pd.read_csv(vintage_path, parse_dates=["date"])
        vintage_target["location"] = vintage_target["location"].astype(str).str.zfill(2)
        vintage_max = vintage_target["date"].max()

        hybrid, n_replaced = create_hybrid_data(final_data, final_dates, vintage_target)

        print(f"\n{'='*60}")
        print(f"  {ref_date}  (vintage through {vintage_max.date()}, "
              f"{n_replaced} values replaced)")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prelim_preds = {}
            for our_h in ALIGNMENT:
                prelim_preds[our_h] = run_hybrid_inference(hybrid, our_h, tmp_dir)

            for our_h, fs_h in ALIGNMENT.items():
                target_end = ref_dt + timedelta(days=7 * fs_h)
                target_end_str = target_end.strftime("%Y-%m-%d")

                test_dates = final_preds[our_h]["test_dates"]
                t_idx = next(
                    (i for i, d in enumerate(test_dates)
                     if d.date() == target_end.date()),
                    None,
                )
                if t_idx is None:
                    continue

                gt = np.zeros((1, len(STATES)))
                row_complete = True
                for r, abbr in enumerate(STATES):
                    fips = ABBR_TO_FIPS[abbr]
                    loc_gt = final_target[
                        (final_target["date"] == target_end_str) &
                        (final_target["location"] == fips)
                    ]
                    if loc_gt.empty:
                        row_complete = False
                        break
                    gt[0, r] = float(loc_gt["value"].values[0])
                if not row_complete:
                    continue

                bl_csv = BL_DIR / f"{ref_date}-FluSight-baseline.csv"
                if not bl_csv.exists():
                    continue
                bl_frame = _parse_submission(bl_csv, fs_h, STATES, FLUSIGHT_LEVELS)
                if bl_frame is None:
                    continue
                bl_wis_result = compute_wis(
                    bl_frame[np.newaxis, :, :], gt, FLUSIGHT_LEVELS
                )
                bl_wis_per_region = bl_wis_result["wis_per_region"]

                # Final-data WIS
                qpred_final = final_preds[our_h]["avg_pred"][t_idx:t_idx+1]
                wis_final = compute_wis(qpred_final, gt, FLUSIGHT_LEVELS)
                rel_final = compute_relative_wis(
                    wis_final["wis_per_region"], bl_wis_per_region
                )

                # Preliminary-data WIS
                qpred_prelim = prelim_preds[our_h][t_idx:t_idx+1]
                wis_prelim = compute_wis(qpred_prelim, gt, FLUSIGHT_LEVELS)
                rel_prelim = compute_relative_wis(
                    wis_prelim["wis_per_region"], bl_wis_per_region
                )

                median_idx = len(FLUSIGHT_LEVELS) // 2
                mae_final = float(np.mean(
                    np.abs(gt[0] - qpred_final[0, :, median_idx])
                ))
                mae_prelim = float(np.mean(
                    np.abs(gt[0] - qpred_prelim[0, :, median_idx])
                ))

                for dtype, wis_val, rel_wis_val, mae_val in [
                    ("final",       wis_final["wis_mean"],  rel_final,  mae_final),
                    ("preliminary", wis_prelim["wis_mean"], rel_prelim, mae_prelim),
                ]:
                    all_results.append({
                        "reference_date": ref_date,
                        "horizon":        fs_h,
                        "target_end_date": target_end_str,
                        "data_type":      dtype,
                        "wis":            round(wis_val, 2),
                        "rel_wis":        round(rel_wis_val, 3),
                        "median_mae":     round(mae_val, 2),
                    })

                print(f"    h={fs_h}: "
                      f"final Rel.WIS={rel_final:.3f}  "
                      f"prelim Rel.WIS={rel_prelim:.3f}  "
                      f"(Δ={rel_prelim - rel_final:+.3f})")

    if not all_results:
        print("\nNo results generated.")
        return

    df = pd.DataFrame(all_results)
    final_df = df[df["data_type"] == "final"]
    prelim_df = df[df["data_type"] == "preliminary"]

    print(f"\n{'='*60}")
    print("COMPARISON: FINAL vs PRELIMINARY DATA (2024/25 season)")
    print(f"{'='*60}")

    per_horizon = {}
    for h in sorted(final_df["horizon"].unique()):
        f_h = final_df[final_df["horizon"] == h]
        p_h = prelim_df[prelim_df["horizon"] == h]
        common = set(f_h["reference_date"]) & set(p_h["reference_date"])
        f_c = f_h[f_h["reference_date"].isin(common)]
        p_c = p_h[p_h["reference_date"].isin(common)]

        f_rel = f_c["rel_wis"].mean()
        p_rel = p_c["rel_wis"].mean()
        deg = (p_rel - f_rel) / f_rel * 100

        print(f"\n  h={h} ({len(common)} weeks):")
        print(f"    Final data Rel.WIS:       {f_rel:.3f}")
        print(f"    Preliminary data Rel.WIS: {p_rel:.3f}")
        print(f"    Degradation:              +{deg:.1f}%")

        per_horizon[f"h{h}"] = {
            "n_weeks":        len(common),
            "final_rel_wis":  round(f_rel, 3),
            "prelim_rel_wis": round(p_rel, 3),
            "degradation_pct": round(deg, 1),
        }

    all_refs = set(df["reference_date"])
    f_all = final_df[final_df["reference_date"].isin(all_refs)]["rel_wis"]
    p_all = prelim_df[prelim_df["reference_date"].isin(all_refs)]["rel_wis"]
    overall_deg = (p_all.mean() - f_all.mean()) / f_all.mean() * 100

    print(f"\n  Overall ({len(all_refs)} weeks):")
    print(f"    Final data Rel.WIS:       {f_all.mean():.3f}")
    print(f"    Preliminary data Rel.WIS: {p_all.mean():.3f}")
    print(f"    Degradation:              +{overall_deg:.1f}%")

    out_csv = VINTAGE_DIR / "vintaged_scoring_results_2425.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nDetailed results: {out_csv}")

    summary = {
        "season":               "2024/25",
        "n_weeks":              len(all_refs),
        "final_mean_rel_wis":   round(f_all.mean(), 3),
        "prelim_mean_rel_wis":  round(p_all.mean(), 3),
        "degradation_pct":      round(overall_deg, 1),
        "per_horizon":          per_horizon,
    }
    out_json = VINTAGE_DIR / "vintage_comparison_2425.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:          {out_json}")

    print(f"\nConclusion: Using preliminary data instead of final data")
    print(f"degrades ColaGNN Rel.WIS by {overall_deg:.1f}% on 2024/25.")
    print(f"Revised headline: Rel.WIS ≈ {p_all.mean():.3f} on a level playing field.")


if __name__ == "__main__":
    main()
