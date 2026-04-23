"""[HISTORICAL REFERENCE - NOT THE PUBLIC BENCHMARK AUTHORITY]

This script documents an earlier EpiGNN aligned-horizon FluSight rerun.
Like `run_aligned_flusight.py`, it is kept for transparency, not as the
authoritative thesis ranking path. The public benchmark authority is the
stored `results/vintage_matched_*` artifacts together with the shared scorer
in `src/eval/flusight_hub.py`.

Same basic methodology as the historical ColaGNN helper: native quantile loss,
11 seeds, and the `h=2,3,4 -> FluSight h=1,2,3` horizon mapping.
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tuning.runner import train_epignn, FLUSIGHT_LEVELS
from src.eval.evaluate import inference_epignn
from src.eval.quantile import compute_wis, compute_relative_wis
from src.data.loader import assign_seasons, _load_date_index
from src.eval.flusight_hub import evaluate_hub_model

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

with open(PROJECT_ROOT / "results" / "epignn_us_best_config_v2.json") as f:
    cfg = json.load(f)["config"]

ckpt_dir = PROJECT_ROOT / "results" / "native_quantile_epignn"
ckpt_dir.mkdir(parents=True, exist_ok=True)
hub_dir = PROJECT_ROOT / "data" / "FluSight-forecast-hub"
bl_dir = hub_dir / "model-output" / "FluSight-baseline"

date_idx_path = PROJECT_ROOT / "src" / "data" / "date_index.csv"
date_df = pd.read_csv(date_idx_path, header=None, names=["idx", "date"], parse_dates=["date"])
idx_to_date = dict(zip(date_df["idx"], date_df["date"]))
iso_years, iso_weeks = _load_date_index(date_idx_path, "date")
seasons = assign_seasons(iso_years, iso_weeks)

target_path = hub_dir / "target-data" / "target-hospital-admissions.csv"
target_df = pd.read_csv(target_path, parse_dates=["date"])
target_df["location"] = target_df["location"].astype(str).str.zfill(2)

print(f"Device: {DEVICE}")
print(f"Seeds:  {SEEDS}")
print(f"EpiGNN cfg: {cfg}")

ALIGNMENT = {2: 1, 3: 2, 4: 3}

results_all = {}
all_qpred_combined = []
all_gt_combined = []

for our_h, fs_h in ALIGNMENT.items():
    print(f"\n{'='*60}")
    print(f"EPIGNN h={our_h} (aligned with FluSight h={fs_h})")
    print(f"{'='*60}")

    cfg_path = str(PROJECT_ROOT / "src" / "configs" / f"us_epignn_h{our_h}.json")

    P = 20
    min_idx = P + our_h - 1
    test_indices_h = [i for i in range(min_idx, len(seasons)) if seasons[i] == "2024/25"]
    test_dates_h = pd.DatetimeIndex([idx_to_date[i] for i in test_indices_h])

    gt_h = np.full((len(test_indices_h), len(STATES)), np.nan)
    for t, d in enumerate(test_dates_h):
        day_data = target_df[target_df["date"] == d]
        for r, abbr in enumerate(STATES):
            row = day_data[day_data["location"] == ABBR_TO_FIPS[abbr]]
            if not row.empty:
                gt_h[t, r] = float(row["value"].values[0])

    overlap_tidx = []
    for t, td in enumerate(test_dates_h):
        ref = td - timedelta(days=7 * fs_h)
        fname = f"{ref.strftime('%Y-%m-%d')}-FluSight-baseline.csv"
        if (bl_dir / fname).exists() and not np.isnan(gt_h[t]).any():
            overlap_tidx.append(t)
    print(f"Overlap weeks: {len(overlap_tidx)}")

    all_preds_h = []
    for seed in SEEDS:
        ckpt_path = ckpt_dir / f"epignn_nq_h{our_h}_seed{seed}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"  Seed {seed}: loaded, epoch={ckpt['best_epoch']}")
        else:
            print(f"  Seed {seed}: training h={our_h}...", flush=True)
            history = train_epignn(
                cfg, cfg_path,
                num_epochs=500, patience=20,
                seed=seed, device=DEVICE,
                quantile_levels=FLUSIGHT_LEVELS,
            )
            ckpt = {
                "best_state": history["best_state"],
                "best_epoch": history["best_epoch"],
                "best_val_loss": history["best_val_loss"],
            }
            torch.save(ckpt, ckpt_path)
            print(f"  Seed {seed}: epoch={ckpt['best_epoch']}, val={ckpt['best_val_loss']:.4e}")

        pred, true = inference_epignn(
            ckpt["best_state"], cfg, cfg_path, split="test", device=DEVICE,
        )
        all_preds_h.append(pred)

    avg_pred = np.mean(all_preds_h, axis=0)
    avg_pred.sort(axis=2)

    pred_ov = avg_pred[overlap_tidx]
    gt_ov = gt_h[overlap_tidx]
    wis_h = compute_wis(pred_ov, gt_ov, FLUSIGHT_LEVELS)

    bl_result = evaluate_hub_model(
        bl_dir, "FluSight-baseline", test_dates_h, gt_h, STATES, horizons=[fs_h],
    )
    rel_h = compute_relative_wis(wis_h["wis_per_region"], bl_result["wis_per_region"])

    per_seed_wis = []
    for p in all_preds_h:
        ps = p[overlap_tidx].copy()
        ps.sort(axis=2)
        sw = compute_wis(ps, gt_ov, FLUSIGHT_LEVELS)
        per_seed_wis.append(sw["wis_mean"])

    print(f"  EpiGNN WIS:     {wis_h['wis_mean']:.2f} (sharp={wis_h['sharpness']:.2f}, cal={wis_h['calibration']:.2f})")
    print(f"  Baseline WIS:   {bl_result['wis_mean']:.2f}")
    print(f"  Rel. WIS:       {rel_h:.3f}")
    print(f"  Per-seed WIS:   {[round(w, 2) for w in per_seed_wis]}")
    print(f"  Seed std:       {np.std(per_seed_wis):.2f}")

    results_all[f"our_h{our_h}_vs_fs_h{fs_h}"] = {
        "our_horizon": our_h,
        "flusight_horizon": fs_h,
        "wis": round(wis_h["wis_mean"], 2),
        "rel_wis": round(rel_h, 3),
        "sharpness": round(wis_h["sharpness"], 2),
        "calibration": round(wis_h["calibration"], 2),
        "baseline_wis": round(bl_result["wis_mean"], 2),
        "n_weeks": len(overlap_tidx),
        "per_seed_wis": [round(w, 2) for w in per_seed_wis],
    }
    all_qpred_combined.append(pred_ov)
    all_gt_combined.append(gt_ov)

# Combined
combined_qpred = np.concatenate(all_qpred_combined, axis=0)
combined_gt = np.concatenate(all_gt_combined, axis=0)
combined_wis = compute_wis(combined_qpred, combined_gt, FLUSIGHT_LEVELS)

P = 20
min_idx_2 = P + 2 - 1
test_indices_ref = [i for i in range(min_idx_2, len(seasons)) if seasons[i] == "2024/25"]
test_dates_ref = pd.DatetimeIndex([idx_to_date[i] for i in test_indices_ref])
gt_ref = np.full((len(test_indices_ref), len(STATES)), np.nan)
for t, d in enumerate(test_dates_ref):
    day_data = target_df[target_df["date"] == d]
    for r, abbr in enumerate(STATES):
        row = day_data[day_data["location"] == ABBR_TO_FIPS[abbr]]
        if not row.empty:
            gt_ref[t, r] = float(row["value"].values[0])

bl_combined = evaluate_hub_model(
    bl_dir, "FluSight-baseline", test_dates_ref, gt_ref, STATES, horizons=[1, 2, 3],
)
combined_rel = compute_relative_wis(
    combined_wis["wis_per_region"], bl_combined["wis_per_region"],
)

print(f"\n{'='*60}")
print(f"COMBINED (EpiGNN our h=2,3,4 vs FluSight h=1,2,3)")
print(f"{'='*60}")
print(f"EpiGNN WIS:     {combined_wis['wis_mean']:.2f}")
print(f"  Sharpness:    {combined_wis['sharpness']:.2f}")
print(f"  Calibration:  {combined_wis['calibration']:.2f}")
print(f"Baseline WIS:   {bl_combined['wis_mean']:.2f}")
print(f"Rel. WIS:       {combined_rel:.3f}")

results_all["combined_aligned"] = {
    "our_horizons": [2, 3, 4],
    "flusight_horizons": [1, 2, 3],
    "wis": round(combined_wis["wis_mean"], 2),
    "rel_wis": round(combined_rel, 3),
    "sharpness": round(combined_wis["sharpness"], 2),
    "calibration": round(combined_wis["calibration"], 2),
    "baseline_wis": round(bl_combined["wis_mean"], 2),
}

print(f"\n{'='*60}")
print("COMPARISON: EPIGNN vs COLAGNN vs FLUSIGHT")
print(f"{'='*60}")
print(f"  EpiGNN (aligned):        {combined_rel:.3f}")

colagnn_path = PROJECT_ROOT / "results" / "native_quantile_v3" / "aligned_flusight_results.json"
if colagnn_path.exists():
    with open(colagnn_path) as f:
        colagnn_rel = json.load(f)["combined_aligned"]["rel_wis"]
    print(f"  ColaGNN (aligned):       {colagnn_rel:.3f}")

ranking_path = PROJECT_ROOT / "results" / "flusight_hub_ranking_hcombined.csv"
if ranking_path.exists():
    ranking_df = pd.read_csv(ranking_path)
    for model_name, label in [
        ("FluSight-ensemble", "FluSight-ensemble"),
        ("FluSight-trained_med", "FluSight-trained_med"),
        ("UVAFluX-CESGCN", "UVAFluX-CESGCN (GNN)"),
    ]:
        row = ranking_df[ranking_df["Model"] == model_name]
        if not row.empty:
            print(f"  {label:<23s} {float(row.iloc[0]['Rel. WIS']):.3f}")

out_path = ckpt_dir / "epignn_aligned_results.json"
with open(out_path, "w") as f:
    json.dump(results_all, f, indent=2)
print(f"\nResults saved to: {out_path}")
