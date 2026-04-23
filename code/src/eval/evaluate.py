"""Evaluation utilities for EpiGNN and ColaGNN forecasts.

Metrics computed on the held-out test split (original scale):
  - MAE, RMSE, MAPE, PCC per region and national aggregate (mean over regions)
  - Peak intensity error: predicted_peak - actual_peak
  - Peak week error: predicted_peak_week_idx - actual_peak_week_idx
  - Onset week error: predicted_onset_idx - actual_onset_idx, where onset is
    the first week of 3 consecutive weeks at or above onset_threshold;
    NaN when either series never reaches that run.

Usage:
    from src.tuning.runner import train_epignn
    from src.eval.evaluate import evaluate_epignn

    history = train_epignn(config, config_path, num_epochs=200)
    metrics = evaluate_epignn(
        history["best_state"], config, config_path, onset_threshold=1600.0
    )
    print(metrics["mae_mean"], metrics["rmse_mean"])
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import pearsonr

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.tuning.runner import _activate_epignn, _activate_colagnn, _replace_output_layer


def compute_onset_threshold(loader, config: dict) -> np.ndarray:
    """Compute per-region onset thresholds from off-season weeks in training years.

    Threshold per region = mean + 2 * std of raw admissions during off-season weeks
    (epi weeks 21–39, i.e. loader.seasons[i] is None) that are *not* adjacent to any
    val, test, or excluded season.  This prevents threshold leakage from held-out data.

    Parameters
    ----------
    loader : FluDataLoader
        An instantiated loader.  rawdat and seasons are read directly.
    config : dict
        The same config dict used to create the loader.  Used to determine which
        seasons are blocked (val_seasons, test_seasons, exclude_seasons).

    Returns
    -------
    np.ndarray, shape (m,)
        Per-region onset threshold in original admissions scale.
        Returns np.full(m, np.nan) if no qualifying off-season weeks exist.

    Notes
    -----
    US hard-coded value ~1600 and Sweden ~66 should be close to the national mean of
    the returned array.  Differences reflect per-state heterogeneity — using a single
    scalar uniformly over 51 states is an approximation.
    """
    blocked = (
        set(config.get("val_seasons", []))
        | set(config.get("test_seasons", []))
        | set(config.get("exclude_seasons", []))
    )

    n = len(loader.seasons)
    qualifying = []
    for i in range(n):
        if loader.seasons[i] is not None:
            continue
        # Find the nearest non-None season on each side
        prev_season = next(
            (loader.seasons[j] for j in range(i - 1, -1, -1)
             if loader.seasons[j] is not None),
            None,
        )
        next_season = next(
            (loader.seasons[j] for j in range(i + 1, n)
             if loader.seasons[j] is not None),
            None,
        )
        if prev_season in blocked or next_season in blocked:
            continue
        qualifying.append(i)

    if not qualifying:
        return np.full(loader.m, np.nan)

    offseason_data = loader.rawdat[qualifying]  # shape (k, m)
    return np.mean(offseason_data, axis=0) + 2.0 * np.std(offseason_data, axis=0)


def _onset_week(series: np.ndarray, threshold: float) -> float:
    """Index of first week that starts a 3-consecutive-week run at or above threshold."""
    for t in range(len(series) - 2):
        if series[t] >= threshold and series[t + 1] >= threshold and series[t + 2] >= threshold:
            return float(t)
    return float("nan")


def compute_metrics(pred: np.ndarray, true: np.ndarray,
                    onset_threshold: Optional[float] = None) -> dict:
    """Compute standard influenza forecast metrics from arrays in original scale.

    Parameters
    ----------
    pred : np.ndarray, shape (T, m)
        Model predictions in original (denormalized) scale.
    true : np.ndarray, shape (T, m)
        Ground-truth values in original (denormalized) scale.
    onset_threshold : float, optional
        Epidemic baseline in original scale (US ~1600, Sweden ~66).
        When provided, onset_week_error[r] = predicted_onset[r] - actual_onset[r].
        NaN when either series never completes a 3-week run above threshold.

    Returns
    -------
    dict with keys:
        mae, rmse, mape, pcc            -- np.ndarray shape (m,), per region
        mae_mean, rmse_mean, mape_mean, pcc_mean  -- float, mean over regions
        rmse_global, pcc_global         -- float, computed over all (T*m) predictions
                                           (matches the "RMSE"/"PCC" columns in
                                           original train.py output and paper tables)
        peak_intensity_error            -- np.ndarray shape (m,), pred_peak - true_peak
        peak_week_error                 -- np.ndarray shape (m,), signed week offset
        onset_week_error                -- np.ndarray shape (m,) or None
        n_weeks, n_regions              -- int
    """
    err = pred - true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    rmse_global = float(np.sqrt(np.mean(err ** 2)))
    pcc_global = float(pearsonr(true.flatten(), pred.flatten())[0])

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(true) > 1e-6, np.abs(err / true) * 100.0, np.nan)
    mape = np.nanmean(rel, axis=0)

    m = pred.shape[1]
    region_idx = np.arange(m)
    true_peak_idx = np.argmax(true, axis=0)
    pred_peak_idx = np.argmax(pred, axis=0)
    true_peak_val = true[true_peak_idx, region_idx]
    pred_peak_val = pred[pred_peak_idx, region_idx]

    pcc = np.array([pearsonr(true[:, r], pred[:, r])[0] for r in range(m)])

    onset_err = None
    if onset_threshold is not None:
        onset_err = np.full(m, float("nan"))
        for r in range(m):
            t_onset = _onset_week(true[:, r], onset_threshold)
            p_onset = _onset_week(pred[:, r], onset_threshold)
            if not (np.isnan(t_onset) or np.isnan(p_onset)):
                onset_err[r] = p_onset - t_onset

    return {
        "mae":                  mae,
        "rmse":                 rmse,
        "mape":                 mape,
        "pcc":                  pcc,
        "mae_mean":             float(np.mean(mae)),
        "rmse_mean":            float(np.mean(rmse)),
        "mape_mean":            float(np.nanmean(mape)),
        "pcc_mean":             float(np.nanmean(pcc)),
        "rmse_global":          rmse_global,
        "pcc_global":           pcc_global,
        "peak_intensity_error": pred_peak_val - true_peak_val,
        "peak_week_error":      (pred_peak_idx - true_peak_idx).astype(float),
        "onset_week_error":     onset_err,
        "n_weeks":              int(pred.shape[0]),
        "n_regions":            int(pred.shape[1]),
    }


def _denormalize(arr: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    return arr * (data_max - data_min + 1e-12) + data_min


def _detect_n_quantiles(state_dict: dict, key: str) -> int:
    return state_dict[key].shape[0]


def _run_inference_epignn(best_state: dict, config: dict, config_path: str,
                          device: str = "cpu", split: str = "test") -> tuple:
    """Return (pred, true) in original scale.

    Quantile-trained models (output width > 1): pred is (T, m, Q), true is (T, m).
    Point models: pred is (T, m), true is (T, m).
    """
    _activate_epignn()
    from data import MExDataLoader
    from models import EpiGNN

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    loader = MExDataLoader(config_path, cuda=cuda)

    args = argparse.Namespace(
        window=loader.P, horizon=loader.h,
        n_layer=config.get("n_layer", 1), n_hidden=config.get("n_hidden", 20),
        dropout=config.get("dropout", 0.2), k=config.get("k", 8),
        hidR=config.get("hidR", 64), hidA=config.get("hidA", 64),
        hidP=config.get("hidP", 1), hw=config.get("hw", 0),
        extra="", label="", pcc="", n=config.get("n", 2),
        res=config.get("res", 0), s=config.get("s", 2),
        cuda=cuda, ablation=None,
    )
    model = EpiGNN(args, loader)

    n_q = _detect_n_quantiles(best_state, "output.weight")
    q_mode = n_q > 1
    if q_mode:
        _replace_output_layer(model, "output", n_q)

    model.load_state_dict(best_state)
    if cuda:
        model = model.cuda()

    model.train(False)
    batches = loader.test if split == "test" else loader.val
    preds, trues = [], []
    with torch.no_grad():
        for inputs in loader.get_batches(batches, batch_size=256, shuffle=False):
            X, Y, index = inputs[0], inputs[1], inputs[2]
            out, _ = model(X, index)
            if q_mode:
                preds.append(out.cpu().numpy())
            else:
                preds.append(out.view(-1, loader.m).cpu().numpy())
            trues.append(Y.view(-1, loader.m).cpu().numpy())

    if not preds:
        raise RuntimeError(f"{split} split is empty for config {config_path}")

    pred = np.concatenate(preds, axis=0)
    true = np.vstack(trues)

    if q_mode:
        for q in range(n_q):
            pred[:, :, q] = _denormalize(pred[:, :, q], loader.min, loader.max)
        pred = np.sort(pred, axis=2)
    else:
        pred = _denormalize(pred, loader.min, loader.max)
    true = _denormalize(true, loader.min, loader.max)
    return pred, true


def _run_inference_colagnn(best_state: dict, config: dict, config_path: str,
                           device: str = "cpu", split: str = "test") -> tuple:
    """Return (pred, true) in original scale.

    Quantile-trained models (output width > 1): pred is (T, m, Q), true is (T, m).
    Point models: pred is (T, m), true is (T, m).
    """
    _activate_colagnn()
    from data import MExDataLoader
    from models import cola_gnn

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    loader = MExDataLoader(config_path, cuda=cuda)

    args = argparse.Namespace(
        window=loader.P, horizon=loader.h,
        n_layer=config.get("n_layer", 1), n_hidden=config.get("n_hidden", 20),
        dropout=config.get("dropout", 0.2), k=config.get("k", 10),
        hidsp=config.get("hidsp", 10), bi=config.get("bi", False),
        rnn_model=config.get("rnn_model", "RNN"), cuda=cuda,
    )
    model = cola_gnn(args, loader)

    n_q = _detect_n_quantiles(best_state, "out.weight")
    q_mode = n_q > 1
    if q_mode:
        _replace_output_layer(model, "out", n_q)

    model.load_state_dict(best_state)
    if cuda:
        model = model.cuda()

    model.train(False)
    batches = loader.test if split == "test" else loader.val
    preds, trues = [], []
    with torch.no_grad():
        for inputs in loader.get_batches(batches, batch_size=256, shuffle=False):
            X, Y = inputs[0], inputs[1]
            out, _ = model(X)
            if q_mode:
                preds.append(out.cpu().numpy())
            else:
                preds.append(out.view(-1, loader.m).cpu().numpy())
            trues.append(Y.view(-1, loader.m).cpu().numpy())

    if not preds:
        raise RuntimeError(f"{split} split is empty for config {config_path}")

    pred = np.concatenate(preds, axis=0)
    true = np.vstack(trues)

    if q_mode:
        for q in range(n_q):
            pred[:, :, q] = _denormalize(pred[:, :, q], loader.min, loader.max)
        pred = np.sort(pred, axis=2)
    else:
        pred = _denormalize(pred, loader.min, loader.max)
    true = _denormalize(true, loader.min, loader.max)
    return pred, true


def evaluate_epignn(best_state: dict, config: dict, config_path: str,
                    onset_threshold: Optional[float] = None,
                    device: str = "cpu") -> dict:
    """Score a trained EpiGNN model on the held-out test split.

    Parameters
    ----------
    best_state : dict
        State dict from training: ``train_epignn(...)["best_state"]``.
    config : dict
        Hyperparameter config used during training.
    config_path : str
        Path to the MEx JSON config (controls dataset and split).
    onset_threshold : float, optional
        Epidemic baseline in original scale (US ~1600, Sweden ~66).
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict -- see ``compute_metrics`` for key descriptions.
    """
    pred, true = _run_inference_epignn(best_state, config, config_path, device)
    return compute_metrics(pred, true, onset_threshold)


def evaluate_colagnn(best_state: dict, config: dict, config_path: str,
                     onset_threshold: Optional[float] = None,
                     device: str = "cpu") -> dict:
    """Score a trained ColaGNN model on the held-out test split.

    Parameters
    ----------
    best_state : dict
        State dict from training: ``train_colagnn(...)["best_state"]``.
    config : dict
        Hyperparameter config used during training.
    config_path : str
        Path to the MEx JSON config (controls dataset and split).
    onset_threshold : float, optional
        Epidemic baseline in original scale (US ~1600, Sweden ~66).
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict -- see ``compute_metrics`` for key descriptions.
    """
    pred, true = _run_inference_colagnn(best_state, config, config_path, device)
    return compute_metrics(pred, true, onset_threshold)


def inference_epignn(best_state: dict, config: dict, config_path: str,
                     split: str = "test", device: str = "cpu") -> tuple:
    """Run EpiGNN inference and return (pred, true) in original scale.

    Use ``split="val"`` to get validation predictions for quantile calibration.

    Parameters
    ----------
    split : {"test", "val"}
        Which dataset split to run inference on.

    Returns
    -------
    pred : np.ndarray, shape (T, m)
    true : np.ndarray, shape (T, m)
    """
    return _run_inference_epignn(best_state, config, config_path, device, split)


def inference_colagnn(best_state: dict, config: dict, config_path: str,
                      split: str = "test", device: str = "cpu") -> tuple:
    """Run ColaGNN inference and return (pred, true) in original scale.

    Use ``split="val"`` to get validation predictions for quantile calibration.

    Parameters
    ----------
    split : {"test", "val"}
        Which dataset split to run inference on.

    Returns
    -------
    pred : np.ndarray, shape (T, m)
    true : np.ndarray, shape (T, m)
    """
    return _run_inference_colagnn(best_state, config, config_path, device, split)
