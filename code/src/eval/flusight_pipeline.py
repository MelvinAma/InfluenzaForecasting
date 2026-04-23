"""FluSight comparison pipeline with native quantile loss.

Trains models with pinball loss across multiple horizons and computes WIS
directly from model outputs (no post-hoc quantile regression).

Usage:
    from src.eval.flusight_pipeline import run_flusight_pipeline
    results = run_flusight_pipeline(
        model_type="colagnn",
        hpo_config=cfg,
        horizons=[1, 2, 3],
        seeds=[42, 123, 456],
        device="cuda",
    )
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parents[2]

from src.tuning.runner import train_epignn, train_colagnn, FLUSIGHT_LEVELS
from src.eval.evaluate import inference_epignn, inference_colagnn
from src.eval.quantile import compute_wis, compute_relative_wis


def _get_config_path(model_type: str, horizon: int) -> str:
    if horizon == 1:
        return str(PROJ_ROOT / "src" / "configs" / f"us_{model_type}.json")
    return str(PROJ_ROOT / "src" / "configs" / f"us_{model_type}_h{horizon}.json")


def train_native_quantile(
    model_type: str,
    hpo_config: dict,
    horizon: int,
    seed: int,
    ckpt_dir: Path,
    num_epochs: int = 500,
    patience: int = 20,
    device: str = "cpu",
) -> dict:
    """Train a single model with native quantile loss, with checkpointing."""
    ckpt_path = ckpt_dir / f"{model_type}_nq_h{horizon}_seed{seed}.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt

    config_path = _get_config_path(model_type, horizon)
    train_fn = train_epignn if model_type == "epignn" else train_colagnn

    history = train_fn(
        hpo_config, config_path,
        num_epochs=num_epochs, patience=patience,
        seed=seed, device=device,
        quantile_levels=FLUSIGHT_LEVELS,
    )

    ckpt = {
        "best_state": history["best_state"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "horizon": horizon,
        "seed": seed,
        "model_type": model_type,
    }
    torch.save(ckpt, ckpt_path)
    return ckpt


def inference_native_quantile(
    model_type: str,
    hpo_config: dict,
    horizon: int,
    best_state: dict,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference for a native quantile model. Returns (pred, true).

    pred shape: (T, m, Q) — quantile predictions in original scale.
    true shape: (T, m) — ground truth in original scale.
    """
    config_path = _get_config_path(model_type, horizon)
    infer_fn = inference_epignn if model_type == "epignn" else inference_colagnn
    return infer_fn(best_state, hpo_config, config_path, split="test", device=device)


def run_flusight_pipeline(
    model_type: str,
    hpo_config: dict,
    horizons: list[int] = None,
    seeds: list[int] = None,
    gt_per_horizon: dict[int, np.ndarray] = None,
    overlap_rows_per_horizon: dict[int, list[int]] = None,
    ckpt_dir: Path = None,
    num_epochs: int = 500,
    patience: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Full native quantile pipeline: train, infer, compute WIS.

    Parameters
    ----------
    model_type : {"epignn", "colagnn"}
    hpo_config : dict
        Best HPO config for this model.
    horizons : list of int
        Forecast horizons to evaluate (default [1, 2, 3]).
    seeds : list of int
        Seeds for training (default [42, 123, 456]).
    gt_per_horizon : dict mapping horizon -> np.ndarray (T, m)
        Authoritative ground truth for each horizon's test split.
    overlap_rows_per_horizon : dict mapping horizon -> list of int
        Test row indices that overlap with FluSight submissions.
    ckpt_dir : Path
        Directory for checkpoints.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with:
        per_horizon : dict[int, dict] — per-horizon results
        combined_wis : float — aggregate WIS across horizons
        combined_wis_per_region : np.ndarray (m,) — for Rel. WIS computation
        per_seed_combined_wis : list[float]
    """
    if horizons is None:
        horizons = [1, 2, 3]
    if seeds is None:
        seeds = [42, 123, 456]
    if ckpt_dir is None:
        ckpt_dir = PROJ_ROOT / "results" / f"native_quantile_{model_type}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    per_horizon = {}
    all_qpred_combined = []
    all_gt_combined = []
    all_preds_by_horizon = {}

    for h in horizons:
        if verbose:
            print(f"\n{'='*40}")
            print(f"{model_type.upper()} h={h}")
            print(f"{'='*40}")

        all_preds_h = []
        for seed in seeds:
            if verbose:
                print(f"  Seed {seed}...", end=" ", flush=True)

            ckpt = train_native_quantile(
                model_type, hpo_config, h, seed, ckpt_dir,
                num_epochs, patience, device,
            )
            if verbose:
                print(f"epoch={ckpt['best_epoch']}, val_loss={ckpt['best_val_loss']:.4e}")

            pred, true = inference_native_quantile(
                model_type, hpo_config, h, ckpt["best_state"], device,
            )
            all_preds_h.append(pred)

        avg_pred_h = np.mean(all_preds_h, axis=0)
        avg_pred_h.sort(axis=2)

        gt_h = gt_per_horizon[h] if gt_per_horizon else true
        overlap_rows = overlap_rows_per_horizon.get(h, list(range(len(gt_h)))) \
            if overlap_rows_per_horizon else list(range(len(gt_h)))

        pred_ov = avg_pred_h[overlap_rows]
        gt_ov = gt_h[overlap_rows]
        wis_h = compute_wis(pred_ov, gt_ov, FLUSIGHT_LEVELS)

        per_seed_wis_h = []
        for p in all_preds_h:
            p_sorted = p[overlap_rows].copy()
            p_sorted.sort(axis=2)
            sw = compute_wis(p_sorted, gt_ov, FLUSIGHT_LEVELS)
            per_seed_wis_h.append(sw["wis_mean"])

        per_horizon[h] = {
            "wis": wis_h,
            "avg_pred": avg_pred_h,
            "per_seed_wis": per_seed_wis_h,
            "n_overlap_weeks": len(overlap_rows),
        }

        all_preds_by_horizon[h] = all_preds_h
        all_qpred_combined.append(pred_ov)
        all_gt_combined.append(gt_ov)

        if verbose:
            print(f"  WIS (overlap): {wis_h['wis_mean']:.2f} "
                  f"(sharp={wis_h['sharpness']:.2f}, cal={wis_h['calibration']:.2f})")
            print(f"  Per-seed: {[round(w, 2) for w in per_seed_wis_h]}")

    combined_qpred = np.concatenate(all_qpred_combined, axis=0)
    combined_gt = np.concatenate(all_gt_combined, axis=0)
    combined_wis = compute_wis(combined_qpred, combined_gt, FLUSIGHT_LEVELS)

    per_seed_combined = []
    for s_idx in range(len(seeds)):
        seed_qpreds = []
        seed_gts = []
        for h in horizons:
            gt_h = gt_per_horizon[h] if gt_per_horizon else true
            overlap_rows = overlap_rows_per_horizon.get(h, list(range(len(gt_h)))) \
                if overlap_rows_per_horizon else list(range(len(gt_h)))
            p = all_preds_by_horizon[h][s_idx][overlap_rows].copy()
            p.sort(axis=2)
            seed_qpreds.append(p)
            seed_gts.append(gt_h[overlap_rows])
        sq = np.concatenate(seed_qpreds, axis=0)
        sg = np.concatenate(seed_gts, axis=0)
        sw = compute_wis(sq, sg, FLUSIGHT_LEVELS)
        per_seed_combined.append(sw["wis_mean"])

    if verbose:
        print(f"\n{'='*40}")
        print(f"COMBINED (h={horizons})")
        print(f"{'='*40}")
        print(f"WIS: {combined_wis['wis_mean']:.2f}")
        print(f"  Sharpness:   {combined_wis['sharpness']:.2f}")
        print(f"  Calibration: {combined_wis['calibration']:.2f}")

    return {
        "per_horizon": per_horizon,
        "combined_wis": combined_wis["wis_mean"],
        "combined_wis_result": combined_wis,
        "combined_wis_per_region": combined_wis["wis_per_region"],
        "per_seed_combined_wis": per_seed_combined,
        "horizons": horizons,
        "seeds": seeds,
    }
