"""Leave-one-season-out cross-validation for EpiGNN and ColaGNN.

For each test season S (starting from index min_train_seasons + 1):
  - val   = season S-1
  - train = all seasons before S-1 (excluding exclude_seasons from base config)
  - model is trained from scratch on this split, then scored on S

Aggregate statistics (mean and std across folds) summarise model
performance across multiple epidemic seasons.

Usage:
    from src.eval.cross_validate import cv_epignn, cv_colagnn

    results = cv_epignn(
        "src/configs/us_epignn.json",
        hparam_config={"lr": 1e-3, "k": 8, ...},
        min_train_seasons=2,
        num_epochs=200,
        onset_threshold=1600.0,
    )
    print(results["aggregate"]["mae_mean"], "+/-", results["aggregate"]["mae_std"])
    for fold in results["folds"]:
        print(fold["test_season"], fold["metrics"]["mae_mean"])
"""
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
# MEx Thesis Adaptation: per-fold checkpoint persistence for capacity-planning reuse
import torch
# End MEx Thesis Adaptation

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.data.loader import FluDataLoader
from src.tuning.runner import train_epignn, train_colagnn
from src.eval.evaluate import evaluate_epignn, evaluate_colagnn


# MEx Thesis Adaptation: per-fold checkpoint persistence
def _save_fold_checkpoint(ckpt_dir: str, model_family: str, test_season: str,
                          val_season: str, seed: int, best_state: dict) -> None:
    """Persist best_state for one (fold, seed) so notebooks can re-run inference later."""
    out_dir = Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{test_season.replace('/', '-')}_seed{seed}.pt"
    torch.save(
        {
            "best_state":  best_state,
            "model_family": model_family,
            "test_season":  test_season,
            "val_season":   val_season,
            "seed":         seed,
        },
        out_dir / fname,
    )
# End MEx Thesis Adaptation


def _probe_loader(base_config_path: str) -> tuple[dict, FluDataLoader]:
    with open(base_config_path) as f:
        base_cfg = json.load(f)

    probe_cfg = {
        **base_cfg,
        "split_mode": "ratio",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_seasons": [],
        "val_seasons": [],
    }
    return base_cfg, FluDataLoader(probe_cfg)


def _available_seasons(base_config_path: str) -> list[str]:
    """Return chronologically sorted influenza seasons present in the dataset.

    Seasons excluded by the base config (exclude_seasons) are omitted.
    Off-season weeks (epi weeks 21-39) are never assigned a season and are
    not included.
    """
    base_cfg, fl = _probe_loader(base_config_path)

    excluded = set(base_cfg.get("exclude_seasons", []))
    unique = {s for s in fl.seasons if s is not None} - excluded
    return sorted(unique, key=lambda s: int(s.split("/")[0]))


def _season_start_rows(base_config_path: str) -> dict[str, int]:
    base_cfg, fl = _probe_loader(base_config_path)
    excluded = set(base_cfg.get("exclude_seasons", []))
    starts: dict[str, int] = {}
    for idx, season in enumerate(fl.seasons):
        if season is None or season in excluded or season in starts:
            continue
        starts[season] = idx
    return starts


def _write_fold_config(base_config_path: str, test_season: str,
                       val_season: str) -> str:
    """Write a temporary JSON config for one CV fold. Returns temp file path."""
    with open(base_config_path) as f:
        cfg = json.load(f)

    cfg["split_mode"] = "season"
    cfg["test_seasons"] = [test_season]
    cfg["val_seasons"] = [val_season]
    cfg["exclude_seasons"] = cfg.get("exclude_seasons", [])
    cfg["train_cutoff_row"] = _season_start_rows(base_config_path)[val_season]

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
    )
    json.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _aggregate(fold_metrics: list[dict], onset_threshold: Optional[float],
               *, fold_seasons: Optional[list[str]] = None) -> dict:
    """Compute mean and std of scalar metrics across folds.

    When *fold_seasons* is provided and contains duplicate season labels
    (multi-seed CV), folds are grouped by season to separate season-level
    variability from seed-level variability.
    """
    def _collect(key):
        return np.array([f[key] for f in fold_metrics], dtype=float)

    multi_seed = (fold_seasons is not None
                  and len(set(fold_seasons)) < len(fold_seasons))

    if multi_seed:
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        for season, fm in zip(fold_seasons, fold_metrics):
            groups[season].append(fm)

        season_keys = sorted(groups, key=lambda s: int(s.split("/")[0]))
        agg: dict = {}

        for metric in ("mae_mean", "rmse_mean", "mape_mean", "pcc_mean"):
            short = metric.replace("_mean", "")
            use_nan = short in ("mape", "pcc")
            s_means, s_stds = [], []
            for sk in season_keys:
                vals = [fm[metric] for fm in groups[sk]]
                s_means.append(float(np.nanmean(vals) if use_nan else np.mean(vals)))
                if len(vals) > 1:
                    s_stds.append(float(np.nanstd(vals, ddof=1) if use_nan else np.std(vals, ddof=1)))

            fn = np.nanmean if use_nan else np.mean
            sfn = np.nanstd if use_nan else np.std
            agg[f"{short}_mean"] = float(fn(s_means))
            agg[f"{short}_std"] = float(sfn(s_means, ddof=1) if len(s_means) > 1 else 0.0)
            if s_stds:
                agg[f"{short}_seed_std"] = float(np.mean(s_stds))

        agg["n_seasons"] = len(season_keys)
        agg["n_runs"] = len(fold_metrics)
        agg["n_seeds"] = len(fold_metrics) // len(season_keys)
    else:
        agg = {
            "mae_mean":   float(np.mean(_collect("mae_mean"))),
            "mae_std":    float(np.std(_collect("mae_mean"), ddof=1) if len(fold_metrics) > 1 else 0.0),
            "rmse_mean":  float(np.mean(_collect("rmse_mean"))),
            "rmse_std":   float(np.std(_collect("rmse_mean"), ddof=1) if len(fold_metrics) > 1 else 0.0),
            "mape_mean":  float(np.nanmean(_collect("mape_mean"))),
            "mape_std":   float(np.nanstd(_collect("mape_mean"), ddof=1) if len(fold_metrics) > 1 else 0.0),
            "pcc_mean":   float(np.nanmean(_collect("pcc_mean"))),
            "pcc_std":    float(np.nanstd(_collect("pcc_mean"), ddof=1) if len(fold_metrics) > 1 else 0.0),
            "n_seasons":  len(fold_metrics),
            "n_runs":     len(fold_metrics),
        }

    peak_int_errs = np.concatenate([f["peak_intensity_error"] for f in fold_metrics])
    peak_wk_errs  = np.concatenate([f["peak_week_error"] for f in fold_metrics])
    agg["peak_intensity_mae"] = float(np.mean(np.abs(peak_int_errs)))
    agg["peak_week_mae"]      = float(np.mean(np.abs(peak_wk_errs)))

    if onset_threshold is not None:
        onset_errs = np.concatenate([
            f["onset_week_error"] for f in fold_metrics
            if f["onset_week_error"] is not None
        ])
        valid = onset_errs[~np.isnan(onset_errs)]
        agg["onset_week_mae"] = float(np.mean(np.abs(valid))) if len(valid) > 0 else float("nan")
    else:
        agg["onset_week_mae"] = None

    agg["n_folds"] = len(fold_metrics)
    return agg


def cv_epignn(base_config_path: str, hparam_config: dict,
              min_train_seasons: int = 2, num_epochs: int = 200,
              patience: int = 20, onset_threshold: Optional[float] = None,
              device: str = "cpu", seed: int = 42,
              ckpt_dir: Optional[str] = None) -> dict:
    """Leave-one-season-out cross-validation for EpiGNN.

    Parameters
    ----------
    base_config_path : str
        Path to a MEx JSON config. The split fields (test_seasons, val_seasons)
        are overridden per fold; all other fields (data path, window, etc.) are
        preserved. Seasons in exclude_seasons are never used as folds.
    hparam_config : dict
        Hyperparameter config passed unchanged to train_epignn each fold.
    min_train_seasons : int
        Minimum number of training seasons before the val season. The first
        valid fold index is min_train_seasons + 1.
    num_epochs, patience : int
        Training budget per fold.
    onset_threshold : float, optional
        Epidemic baseline in original scale (US ~1600). Passed to evaluate_epignn.
    device : str
        ``"cpu"`` or ``"cuda"``.
    seed : int
        Random seed passed to train_epignn each fold (same seed across folds
        for reproducibility of architecture initialisation).

    Returns
    -------
    dict with keys:
        folds     -- list of dicts: {test_season, val_season, metrics}
        aggregate -- dict of mean/std over folds (see _aggregate)
    """
    seasons = _available_seasons(base_config_path)
    start_idx = min_train_seasons + 1

    if start_idx >= len(seasons):
        raise ValueError(
            f"Not enough seasons for min_train_seasons={min_train_seasons}. "
            f"Found {len(seasons)} seasons: {seasons}"
        )

    folds = []
    for i in range(start_idx, len(seasons)):
        test_season = seasons[i]
        val_season  = seasons[i - 1]
        fold_cfg    = _write_fold_config(base_config_path, test_season, val_season)
        try:
            history = train_epignn(
                hparam_config, fold_cfg,
                num_epochs=num_epochs, patience=patience,
                seed=seed, device=device,
            )
            metrics = evaluate_epignn(
                history["best_state"], hparam_config, fold_cfg,
                onset_threshold=onset_threshold, device=device,
            )
            # MEx Thesis Adaptation: persist best_state per fold/seed
            if ckpt_dir is not None:
                _save_fold_checkpoint(ckpt_dir, "epignn", test_season, val_season,
                                      seed, history["best_state"])
            # End MEx Thesis Adaptation
        finally:
            Path(fold_cfg).unlink(missing_ok=True)

        folds.append({
            "test_season": test_season,
            "val_season":  val_season,
            "metrics":     metrics,
        })

    return {
        "folds":     folds,
        "aggregate": _aggregate(
            [f["metrics"] for f in folds], onset_threshold,
            fold_seasons=[f["test_season"] for f in folds]),
    }


def cv_colagnn(base_config_path: str, hparam_config: dict,
               min_train_seasons: int = 2, num_epochs: int = 200,
               patience: int = 20, onset_threshold: Optional[float] = None,
               device: str = "cpu", seed: int = 42,
               ckpt_dir: Optional[str] = None) -> dict:
    """Leave-one-season-out cross-validation for ColaGNN.

    Parameters
    ----------
    base_config_path : str
        Path to a MEx JSON config. Split fields are overridden per fold.
    hparam_config : dict
        Hyperparameter config passed unchanged to train_colagnn each fold.
    min_train_seasons : int
        Minimum number of training seasons before the val season.
    num_epochs, patience : int
        Training budget per fold.
    onset_threshold : float, optional
        Epidemic baseline in original scale (US ~1600, Sweden ~66).
    device : str
        ``"cpu"`` or ``"cuda"``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        folds     -- list of dicts: {test_season, val_season, metrics}
        aggregate -- dict of mean/std over folds (see _aggregate)
    """
    seasons = _available_seasons(base_config_path)
    start_idx = min_train_seasons + 1

    if start_idx >= len(seasons):
        raise ValueError(
            f"Not enough seasons for min_train_seasons={min_train_seasons}. "
            f"Found {len(seasons)} seasons: {seasons}"
        )

    folds = []
    for i in range(start_idx, len(seasons)):
        test_season = seasons[i]
        val_season  = seasons[i - 1]
        fold_cfg    = _write_fold_config(base_config_path, test_season, val_season)
        try:
            history = train_colagnn(
                hparam_config, fold_cfg,
                num_epochs=num_epochs, patience=patience,
                seed=seed, device=device,
            )
            metrics = evaluate_colagnn(
                history["best_state"], hparam_config, fold_cfg,
                onset_threshold=onset_threshold, device=device,
            )
            # MEx Thesis Adaptation: persist best_state per fold/seed
            if ckpt_dir is not None:
                _save_fold_checkpoint(ckpt_dir, "colagnn", test_season, val_season,
                                      seed, history["best_state"])
            # End MEx Thesis Adaptation
        finally:
            Path(fold_cfg).unlink(missing_ok=True)

        folds.append({
            "test_season": test_season,
            "val_season":  val_season,
            "metrics":     metrics,
        })

    return {
        "folds":     folds,
        "aggregate": _aggregate(
            [f["metrics"] for f in folds], onset_threshold,
            fold_seasons=[f["test_season"] for f in folds]),
    }
