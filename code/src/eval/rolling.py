"""Rolling (expanding-window) forecast evaluation — EXPERIMENTAL, NOT USED IN RESULTS.

Not imported by any result-producing script or notebook. Contains a known bug:
inference (line 123) uses the original config_path for normalization, but training
uses a rolling config with train_cutoff_row. This means the model sees data
normalized differently at inference time than during training.

If this module is to be used for thesis results, fix inference to use the rolling
config for normalization consistency.

Usage:
    from src.eval.rolling import rolling_forecast
    results = rolling_forecast(
        model_type="colagnn",
        config=config,
        config_path="src/configs/us_colagnn.json",
        retrain_every=4,
        seeds=[42, 123, 456],
        quantile_levels=FLUSIGHT_LEVELS,
    )
    # results["predictions"] shape: (n_test_weeks, n_regions, n_quantiles, n_seeds)
"""
import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]


def rolling_forecast(
    model_type: str,
    config: dict,
    config_path: str,
    retrain_every: int = 4,
    seeds: list[int] = None,
    num_epochs: int = 200,
    patience: int = 20,
    quantile_levels: Optional[list[float]] = None,
    device: str = "cpu",
) -> dict:
    """Expanding-window forecast with periodic retraining.

    Parameters
    ----------
    model_type : {"epignn", "colagnn"}
    config : dict
        Base config with test_seasons, val_seasons, etc.
    config_path : str
        Path to the JSON config file.
    retrain_every : int
        Retrain every N test weeks. 1 = every week (expensive), 4 = monthly.
    seeds : list of int
        Seeds to average over.
    quantile_levels : list of float or None
        If provided, train with pinball loss and produce quantile forecasts.

    Returns
    -------
    dict with:
        predictions: np.ndarray (n_test, m[, Q]) — per-seed averaged
        ground_truth: np.ndarray (n_test, m)
        per_seed_predictions: list of np.ndarray
        retrain_points: list of int — test indices where retraining occurred
    """
    import sys
    sys.path.insert(0, str(PROJ_ROOT / "src" / "data"))
    from loader import FluDataLoader

    if seeds is None:
        seeds = [42]

    base_loader = FluDataLoader(config)
    test_indices = sorted(base_loader.test_set)
    if not test_indices:
        raise RuntimeError("No test indices found")

    if model_type == "epignn":
        from src.tuning.runner import train_epignn as train_fn
        from src.eval.evaluate import _run_inference_epignn as infer_fn
        output_key = "output.weight"
    elif model_type == "colagnn":
        from src.tuning.runner import train_colagnn as train_fn
        from src.eval.evaluate import _run_inference_colagnn as infer_fn
        output_key = "out.weight"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    n_test = len(test_indices)
    m = base_loader.m

    all_seed_preds = []
    retrain_points = []

    for seed in seeds:
        seed_preds = []
        current_state = None

        for t_idx, test_row in enumerate(test_indices):
            need_retrain = (t_idx % retrain_every == 0) or current_state is None

            if need_retrain:
                if seed == seeds[0]:
                    retrain_points.append(t_idx)
                rolling_config = copy.deepcopy(config)
                rolling_config["train_cutoff_row"] = test_row

                import tempfile, os
                fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="rolling_")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(rolling_config, f)
                    history = train_fn(
                        rolling_config, tmp_path,
                        num_epochs=num_epochs, patience=patience,
                        seed=seed, device=device,
                        quantile_levels=quantile_levels,
                    )
                    current_state = history["best_state"]
                finally:
                    os.unlink(tmp_path)

            # For each test week, we need single-week inference.
            # Use the full test split and pick the right row.
            if t_idx == 0 or need_retrain:
                pred_full, true_full = infer_fn(
                    current_state, config, config_path, device=device, split="test"
                )

            seed_preds.append(pred_full[t_idx])

        all_seed_preds.append(np.array(seed_preds))

    ground_truth = np.array([
        base_loader.rawdat[idx] for idx in test_indices
    ])

    avg_pred = np.mean(all_seed_preds, axis=0)

    return {
        "predictions": avg_pred,
        "ground_truth": ground_truth,
        "per_seed_predictions": all_seed_preds,
        "retrain_points": sorted(set(retrain_points)),
        "test_indices": test_indices,
        "n_retrains": len(set(retrain_points)),
    }
