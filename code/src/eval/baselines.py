"""Persistence and seasonal naive baseline forecasts.

Both baselines produce predictions compatible with compute_metrics():
arrays of shape (n_test, m) in original (denormalized) scale.

Persistence: prediction for week t is the observed value at week t-1.
Seasonal naive: prediction for week t is the observed value from the
same epidemiological week in the previous season.

These are evaluated on the test split only, using the same season-aligned
splits as the GNN models, so results are directly comparable.

Usage:
    from src.eval.baselines import persistence_forecast, seasonal_naive_forecast
    from src.eval.evaluate import compute_metrics

    pred, true = persistence_forecast("src/configs/sweden_epignn.json")
    metrics = compute_metrics(pred, true, onset_threshold=66.0)
"""
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from src.data.loader import FluDataLoader


def _load_test_indices(config_path: str) -> tuple:
    """Load FluDataLoader and return (rawdat, test_set, seasons)."""
    with open(config_path) as f:
        cfg = json.load(f)
    fl = FluDataLoader(cfg)
    return fl.rawdat, fl.test_set, fl.seasons


def persistence_forecast(config_path: str,
                         horizon: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Persistence baseline: predict the value observed `horizon` weeks before the target.

    For each test index i, the prediction is rawdat[i - horizon, :].
    With horizon=1, this is rawdat[i-1, :], i.e. the most recently observed
    value at prediction time.

    Parameters
    ----------
    config_path : str
        Path to a MEx JSON config. Uses the test split as defined by the config.
    horizon : int, optional
        Forecast horizon override. If None, uses the value from the config.

    Returns
    -------
    pred : np.ndarray, shape (n_test, m) -- predictions in original scale
    true : np.ndarray, shape (n_test, m) -- observed values in original scale
    """
    with open(config_path) as f:
        cfg = json.load(f)
    h = horizon if horizon is not None else cfg.get("horizon", 1)
    fl = FluDataLoader(cfg)

    test_idx = fl.test_set
    pred = np.stack([fl.rawdat[i - h, :] for i in test_idx], axis=0)
    true = np.stack([fl.rawdat[i, :] for i in test_idx], axis=0)
    return pred, true


def seasonal_naive_forecast(config_path: str,
                            season_length: int = 52) -> tuple[np.ndarray, np.ndarray]:
    """Seasonal naive baseline: predict the value from the same week one year prior.

    For each test index i, the prediction is rawdat[i - season_length, :].
    When the lookback index is out of range (i.e., i < season_length), that
    test point is skipped and excluded from the returned arrays.

    Parameters
    ----------
    config_path : str
        Path to a MEx JSON config.
    season_length : int
        Number of weeks in one season. Default 52 (one calendar year).

    Returns
    -------
    pred : np.ndarray, shape (n_valid_test, m)
    true : np.ndarray, shape (n_valid_test, m)
    """
    with open(config_path) as f:
        cfg = json.load(f)
    fl = FluDataLoader(cfg)

    test_idx = fl.test_set
    preds, trues = [], []
    for i in test_idx:
        lookback = i - season_length
        if lookback < 0:
            continue
        preds.append(fl.rawdat[lookback, :])
        trues.append(fl.rawdat[i, :])

    if not preds:
        raise RuntimeError(
            f"No valid test points with season_length={season_length}. "
            f"First test index is {min(test_idx)}, need at least {season_length}."
        )
    return np.stack(preds, axis=0), np.stack(trues, axis=0)


def cv_persistence(config_path: str,
                   min_train_seasons: int = 2,
                   onset_threshold: Optional[float] = None,
                   horizon: Optional[int] = None) -> dict:
    """Leave-one-season-out cross-validation for the persistence baseline.

    Uses the same fold structure as cv_epignn/cv_colagnn, producing directly
    comparable aggregate metrics. No training is required.

    Parameters
    ----------
    config_path : str
        Path to a MEx JSON config.
    min_train_seasons : int
        Same semantics as in cv_epignn.
    onset_threshold : float, optional
        Passed to compute_metrics.
    horizon : int, optional
        Forecast horizon override. If None, uses the value from the config.

    Returns
    -------
    dict with keys: folds, aggregate -- same structure as cv_epignn output.
    """
    import json
    import tempfile
    from src.eval.cross_validate import _available_seasons, _write_fold_config
    from src.eval.evaluate import compute_metrics
    from src.eval.cross_validate import _aggregate

    seasons = _available_seasons(config_path)
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
        fold_cfg    = _write_fold_config(config_path, test_season, val_season)
        try:
            pred, true = persistence_forecast(fold_cfg, horizon=horizon)
            metrics = compute_metrics(pred, true, onset_threshold)
        finally:
            Path(fold_cfg).unlink(missing_ok=True)
        folds.append({"test_season": test_season, "val_season": val_season,
                      "metrics": metrics})

    return {"folds": folds,
            "aggregate": _aggregate(
                [f["metrics"] for f in folds], onset_threshold,
                fold_seasons=[f["test_season"] for f in folds])}


def cv_seasonal_naive(config_path: str,
                      min_train_seasons: int = 2,
                      season_length: int = 52,
                      onset_threshold: Optional[float] = None) -> dict:
    """Leave-one-season-out cross-validation for the seasonal naive baseline.

    Uses the same fold structure as cv_epignn/cv_colagnn.

    Parameters
    ----------
    config_path : str
        Path to a MEx JSON config.
    min_train_seasons, season_length, onset_threshold : see other functions.

    Returns
    -------
    dict with keys: folds, aggregate.
    """
    from src.eval.cross_validate import _available_seasons, _write_fold_config, _aggregate
    from src.eval.evaluate import compute_metrics

    seasons = _available_seasons(config_path)
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
        fold_cfg    = _write_fold_config(config_path, test_season, val_season)
        try:
            pred, true = seasonal_naive_forecast(fold_cfg, season_length=season_length)
            metrics = compute_metrics(pred, true, onset_threshold)
        finally:
            Path(fold_cfg).unlink(missing_ok=True)
        folds.append({"test_season": test_season, "val_season": val_season,
                      "metrics": metrics})

    return {"folds": folds,
            "aggregate": _aggregate(
                [f["metrics"] for f in folds], onset_threshold,
                fold_seasons=[f["test_season"] for f in folds])}
