"""Quick validation: native quantile ColaGNN vs post-hoc QR.

Trains ColaGNN with pinball loss (23 FluSight quantile levels) on US NHSN h=1.
Compares resulting WIS against FluSight-baseline to check if Rel. WIS is plausible.

Previous post-hoc QR result: ColaGNN Rel. WIS = 0.467 (implausibly good).
Best-ever FluSight model: ~0.61. If native quantile gives a more realistic number,
the post-hoc QR was artificially advantaged.

Usage:
    python scripts/validate_native_quantile.py [--seeds 42 123 456] [--epochs 500]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tuning.runner import train_colagnn, FLUSIGHT_LEVELS
from src.eval.evaluate import inference_colagnn
from src.eval.quantile import compute_wis, compute_relative_wis
from src.data.loader import assign_seasons, _load_date_index

STATE_INDEX = pd.read_csv(
    PROJECT_ROOT / "src" / "data" / "state_index.csv",
    header=None, names=["idx", "abbr"],
)
STATES = list(STATE_INDEX["abbr"])
FIPS_TO_ABBR_MAP = {
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
ABBR_TO_FIPS_MAP = {v: k for k, v in FIPS_TO_ABBR_MAP.items()}


def load_flusight_baseline_wis(overlap_tidx, gt, test_dates):
    """Load FluSight-baseline forecasts and compute WIS on overlap window."""
    from datetime import timedelta

    flusight_dir = PROJECT_ROOT / "data" / "flusight_2024_25" / "model-output" / "FluSight-baseline"
    Q = len(FLUSIGHT_LEVELS)
    level_to_q = {round(lvl, 4): i for i, lvl in enumerate(FLUSIGHT_LEVELS)}

    qpred_rows, gt_rows = [], []
    for t in overlap_tidx:
        target_date = test_dates[t]
        ref_date = target_date - timedelta(days=7)
        fpath = flusight_dir / f"{ref_date.strftime('%Y-%m-%d')}-FluSight-baseline.csv"
        if not fpath.exists():
            continue

        df = pd.read_csv(fpath)
        df.columns = df.columns.str.lower()
        df["location"] = df["location"].astype(str).str.zfill(2)
        mask = (
            (df["horizon"] == 1)
            & (df["output_type"].str.lower() == "quantile")
            & (df["location"].isin(FIPS_TO_ABBR_MAP))
        )
        df_h1 = df[mask].copy()
        if df_h1.empty:
            continue

        df_h1["q_level"] = df_h1["output_type_id"].apply(lambda x: round(float(x), 4))
        df_h1 = df_h1[df_h1["q_level"].isin(set(level_to_q))]

        frame = np.full((len(STATES), Q), np.nan)
        for _, row in df_h1.iterrows():
            abbr = FIPS_TO_ABBR_MAP.get(row["location"])
            q_idx = level_to_q.get(row["q_level"])
            if abbr in STATES and q_idx is not None:
                frame[STATES.index(abbr), q_idx] = float(row["value"])

        if np.sum(~np.isnan(frame[:, 0])) < 26:
            continue

        for r in range(len(STATES)):
            q_row = frame[r]
            if np.isnan(q_row).any() and not np.isnan(q_row).all():
                known = np.where(~np.isnan(q_row))[0]
                frame[r] = np.interp(np.arange(Q), known, q_row[known])

        qpred_rows.append(frame)
        gt_rows.append(gt[t])

    qpred = np.stack(qpred_rows)
    gt_arr = np.stack(gt_rows)
    return compute_wis(qpred, gt_arr, FLUSIGHT_LEVELS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds:  {args.seeds}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")

    cfg_path = str(PROJECT_ROOT / "src" / "configs" / "us_colagnn.json")
    with open(PROJECT_ROOT / "results" / "colagnn_us_best_config_v2.json") as f:
        cfg = json.load(f)["config"]

    ckpt_dir = PROJECT_ROOT / "results" / "native_quantile_v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Build ground truth and test dates ---
    date_idx_path = PROJECT_ROOT / "src" / "data" / "date_index.csv"
    date_df = pd.read_csv(date_idx_path, header=None, names=["idx", "date"], parse_dates=["date"])
    idx_to_date = dict(zip(date_df["idx"], date_df["date"]))
    iso_years, iso_weeks = _load_date_index(date_idx_path, "date")
    seasons = assign_seasons(iso_years, iso_weeks)

    P, h = 20, 1
    min_idx = P + h - 1
    test_indices = [i for i in range(min_idx, len(seasons)) if seasons[i] == "2024/25"]
    test_dates = pd.DatetimeIndex([idx_to_date[i] for i in test_indices])

    target_path = PROJECT_ROOT / "data" / "flusight_2024_25" / "target-hospital-admissions.csv"
    target_df = pd.read_csv(target_path, parse_dates=["date"])
    target_df["location"] = target_df["location"].astype(str).str.zfill(2)

    gt = np.full((len(test_indices), len(STATES)), np.nan)
    for t, d in enumerate(test_dates):
        day_data = target_df[target_df["date"] == d]
        for r, abbr in enumerate(STATES):
            row = day_data[day_data["location"] == ABBR_TO_FIPS_MAP[abbr]]
            if not row.empty:
                gt[t, r] = float(row["value"].values[0])

    # --- Determine overlap with FluSight (weeks where baseline exists) ---
    from datetime import timedelta
    flusight_dir = PROJECT_ROOT / "data" / "flusight_2024_25" / "model-output" / "FluSight-baseline"
    overlap_tidx = []
    for t, td in enumerate(test_dates):
        ref = td - timedelta(days=7)
        if (flusight_dir / f"{ref.strftime('%Y-%m-%d')}-FluSight-baseline.csv").exists():
            if not np.isnan(gt[t]).any():
                overlap_tidx.append(t)
    print(f"\nOverlap weeks with FluSight: {len(overlap_tidx)}")
    print(f"  {test_dates[overlap_tidx[0]].date()} to {test_dates[overlap_tidx[-1]].date()}")

    # --- Train with native quantile loss ---
    all_preds = []
    all_true = None
    per_seed_wis = []

    for seed in args.seeds:
        ckpt_path = ckpt_dir / f"colagnn_nq_seed{seed}.pt"

        if ckpt_path.exists():
            print(f"\nSeed {seed}: loading checkpoint")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"  epoch={ckpt['best_epoch']}, val_loss={ckpt['best_val_loss']:.4e}")
        else:
            print(f"\nSeed {seed}: training with native quantile loss...")
            history = train_colagnn(
                cfg, cfg_path,
                num_epochs=args.epochs, patience=args.patience,
                seed=seed, device=device,
                quantile_levels=FLUSIGHT_LEVELS,
            )
            ckpt = {
                "best_state": history["best_state"],
                "best_epoch": history["best_epoch"],
                "best_val_loss": history["best_val_loss"],
            }
            torch.save(ckpt, ckpt_path)
            print(f"  epoch={ckpt['best_epoch']}, val_loss={ckpt['best_val_loss']:.4e}")

        pred, true = inference_colagnn(
            ckpt["best_state"], cfg, cfg_path, split="test", device=device,
        )
        print(f"  pred shape: {pred.shape}")  # expect (T, 51, 23)
        all_preds.append(pred)
        if all_true is None:
            all_true = true

        # Per-seed WIS on overlap window
        pred_ov = pred[overlap_tidx]
        gt_ov = gt[overlap_tidx]
        seed_wis = compute_wis(pred_ov, gt_ov, FLUSIGHT_LEVELS)
        per_seed_wis.append(seed_wis["wis_mean"])
        print(f"  seed {seed} WIS (overlap): {seed_wis['wis_mean']:.2f}")

    # --- Seed-averaged predictions ---
    avg_pred = np.mean(all_preds, axis=0)
    avg_pred.sort(axis=2)  # re-enforce monotonicity after averaging

    # WIS on overlap window
    avg_pred_ov = avg_pred[overlap_tidx]
    gt_ov = gt[overlap_tidx]
    primary_wis = compute_wis(avg_pred_ov, gt_ov, FLUSIGHT_LEVELS)

    print("\n" + "=" * 60)
    print("NATIVE QUANTILE ColaGNN — RESULTS")
    print("=" * 60)
    print(f"Primary WIS (seed-averaged): {primary_wis['wis_mean']:.2f}")
    print(f"  Sharpness:   {primary_wis['sharpness']:.2f}")
    print(f"  Calibration: {primary_wis['calibration']:.2f}")
    cal_ratio = primary_wis['calibration'] / primary_wis['wis_mean']
    print(f"  Cal/WIS:     {cal_ratio:.1%}")
    print(f"Per-seed WIS:  {[round(w, 2) for w in per_seed_wis]}")
    print(f"  mean ± std:  {np.mean(per_seed_wis):.2f} ± {np.std(per_seed_wis):.2f}")

    # --- FluSight-baseline WIS on same overlap ---
    print("\nComputing FluSight-baseline WIS on same overlap window...")
    bl_wis = load_flusight_baseline_wis(overlap_tidx, gt, test_dates)
    print(f"FluSight-baseline WIS: {bl_wis['wis_mean']:.2f}")

    # --- Relative WIS ---
    rel_wis = compute_relative_wis(primary_wis["wis_per_region"], bl_wis["wis_per_region"])
    print(f"\nRelative WIS (native quantile): {rel_wis:.3f}")
    print(f"Relative WIS (old post-hoc QR): 0.467")
    print(f"FluSight-ensemble Rel. WIS:     0.727")
    print(f"Best FluSight model ever:       ~0.61")

    if rel_wis > 0.55:
        print("\n→ PLAUSIBLE: Native quantile gives a realistic Rel. WIS.")
        print("  Post-hoc QR was likely artificially advantaged.")
    elif rel_wis < 0.45:
        print("\n→ STILL IMPLAUSIBLE: Check for data leakage or computation errors.")
    else:
        print("\n→ BORDERLINE: Needs further investigation.")

    # --- Also compute WIS on full test season ---
    gt_full_rows = [t for t in range(len(test_dates)) if not np.isnan(gt[t]).any()]
    avg_pred_full = avg_pred[gt_full_rows]
    gt_full = gt[gt_full_rows]
    full_wis = compute_wis(avg_pred_full, gt_full, FLUSIGHT_LEVELS)
    print(f"\nFull season WIS ({len(gt_full_rows)} weeks): {full_wis['wis_mean']:.2f}")

    # --- Save results ---
    results = {
        "method": "native_quantile",
        "model": "colagnn",
        "horizon": 1,
        "seeds": args.seeds,
        "n_overlap_weeks": len(overlap_tidx),
        "wis_primary": round(primary_wis["wis_mean"], 2),
        "wis_sharpness": round(primary_wis["sharpness"], 2),
        "wis_calibration": round(primary_wis["calibration"], 2),
        "wis_per_seed": [round(w, 2) for w in per_seed_wis],
        "relative_wis": round(rel_wis, 3),
        "baseline_wis": round(bl_wis["wis_mean"], 2),
        "wis_full_season": round(full_wis["wis_mean"], 2),
    }
    out_path = ckpt_dir / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
