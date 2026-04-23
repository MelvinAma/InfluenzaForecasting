"""Ray Tune HPO for EpiGNN and ColaGNN.

Usage (standalone):
    python -m src.tuning.hpo --model epignn --config src/configs/us_epignn.json \
        --num-samples 50 --max-epochs 200 --gpus-per-trial 0.5

Usage (from notebook or Python):
    from src.tuning.hpo import run_epignn_hpo, run_colagnn_hpo
    results = run_epignn_hpo("src/configs/us_epignn.json", num_samples=50)
    best = results.get_best_result(metric="val_loss", mode="min")
    print(best.config, best.metrics)
"""
import os
import sys
import hashlib
import argparse
from pathlib import Path
from typing import Optional

os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")

if sys.platform == "win32":
    import ray._private.utils as _ray_utils
    _ray_utils.detect_fate_sharing_support_win32 = lambda: False

import ray
from ray import tune
from ray.tune import TuneConfig, Tuner, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

EPIGNN_SPACE = {
    "lr":           tune.loguniform(1e-4, 1e-2),
    "k":            tune.choice([4, 8, 12, 16]),
    "hidP":         tune.choice([1, 2, 4]),
    "hidA":         tune.choice([32, 64, 128]),
    "hidR":         64,
    "n_hidden":     tune.choice([16, 32, 64]),
    "n_layer":      tune.choice([1, 2]),
    "n":            tune.choice([1, 2, 3]),
    "dropout":      tune.uniform(0.1, 0.5),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "batch_size":   tune.choice([32, 64, 128]),
    "res":          0,
    "s":            tune.choice([2, 3]),
    "hw":           tune.choice([0, 1]),
}

COLAGNN_SPACE = {
    "lr":           tune.loguniform(1e-4, 1e-2),
    "k":            tune.choice([5, 10, 15, 20]),
    "hidsp":        tune.choice([8, 15, 32, 64]),
    "n_hidden":     tune.choice([16, 32, 64]),
    "n_layer":      tune.choice([1, 2]),
    "dropout":      tune.uniform(0.1, 0.5),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "batch_size":   tune.choice([16, 32, 64]),
    "rnn_model":    tune.choice(["RNN", "GRU", "LSTM"]),
    "bi":           False,  # bi=True breaks cola_gnn.forward (view resolves to 2b not b)
}


def _epignn_trial(config):
    config = dict(config)
    config_path = config.pop("_config_path")
    num_epochs   = config.pop("_num_epochs", 200)
    patience     = config.pop("_patience", 20)
    device       = config.pop("_device", "cpu")
    seed         = config.pop("_seed", 42)

    from src.tuning.runner import train_epignn
    history = train_epignn(config, config_path, num_epochs=num_epochs,
                           patience=patience, seed=seed, device=device)
    tune.report({"val_loss": history["best_val_loss"],
                 "best_epoch": history["best_epoch"]})


def _colagnn_trial(config):
    config = dict(config)
    config_path = config.pop("_config_path")
    num_epochs   = config.pop("_num_epochs", 200)
    patience     = config.pop("_patience", 20)
    device       = config.pop("_device", "cpu")
    seed         = config.pop("_seed", 42)

    from src.tuning.runner import train_colagnn
    history = train_colagnn(config, config_path, num_epochs=num_epochs,
                            patience=patience, seed=seed, device=device)
    tune.report({"val_loss": history["best_val_loss"],
                 "best_epoch": history["best_epoch"]})


def _short_dirname(trial):
    digest = hashlib.sha1(trial.trial_id.encode()).hexdigest()[:8]
    return f"{trial.trainable_name}_{digest}"


def _make_tuner(trial_fn, space, config_path, num_samples, max_epochs,
                patience, gpus_per_trial, seed, storage_path,
                max_concurrent_trials=2):
    config_path = str(Path(config_path).resolve())
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=min(max(1, max_epochs // 20), max_epochs),
        reduction_factor=2,
    )
    search_alg = OptunaSearch(metric="val_loss", mode="min", seed=seed)

    param_space = {
        **space,
        "_config_path": config_path,
        "_num_epochs":  max_epochs,
        "_patience":    patience,
        "_device":      "cuda" if gpus_per_trial > 0 else "cpu",
        "_seed":        seed,
    }

    resources = {"cpu": 2}
    if gpus_per_trial > 0:
        resources["gpu"] = gpus_per_trial

    storage = storage_path or str(PROJ_ROOT / "results" / "ray_results")
    os.makedirs(storage, exist_ok=True)

    return Tuner(
        trainable=tune.with_resources(trial_fn, resources),
        param_space=param_space,
        tune_config=TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            metric="val_loss",
            mode="min",
            trial_dirname_creator=_short_dirname,
        ),
        run_config=RunConfig(storage_path=storage),
    )


def run_epignn_hpo(config_path: str, num_samples: int = 150, max_epochs: int = 500,
                   patience: int = 20, gpus_per_trial: float = 0.0,
                   seed: int = 42, storage_path: Optional[str] = None,
                   max_concurrent_trials: int = 2):
    """Run hyperparameter search for EpiGNN. Returns ResultGrid."""
    if not ray.is_initialized():
        ray.init()
    tuner = _make_tuner(_epignn_trial, EPIGNN_SPACE, config_path,
                        num_samples, max_epochs, patience, gpus_per_trial,
                        seed, storage_path, max_concurrent_trials)
    return tuner.fit()


def run_colagnn_hpo(config_path: str, num_samples: int = 150, max_epochs: int = 500,
                    patience: int = 20, gpus_per_trial: float = 0.0,
                    seed: int = 42, storage_path: Optional[str] = None,
                    max_concurrent_trials: int = 2):
    """Run hyperparameter search for ColaGNN. Returns ResultGrid."""
    if not ray.is_initialized():
        ray.init()
    tuner = _make_tuner(_colagnn_trial, COLAGNN_SPACE, config_path,
                        num_samples, max_epochs, patience, gpus_per_trial,
                        seed, storage_path, max_concurrent_trials)
    return tuner.fit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       required=True, choices=["epignn", "colagnn"])
    ap.add_argument("--config",      required=True, help="Path to MEx JSON config")
    ap.add_argument("--num-samples", type=int,   default=150)
    ap.add_argument("--max-epochs",  type=int,   default=500)
    ap.add_argument("--patience",    type=int,   default=20)
    ap.add_argument("--gpus",        type=float, default=0.0,
                    help="GPU fraction per trial (0 = CPU only)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--storage",     type=str,   default=None)
    ap.add_argument("--concurrent",  type=int,   default=2,
                    help="Max concurrent trials")
    args = ap.parse_args()

    fn = run_epignn_hpo if args.model == "epignn" else run_colagnn_hpo
    results = fn(args.config, num_samples=args.num_samples, max_epochs=args.max_epochs,
                 patience=args.patience, gpus_per_trial=args.gpus,
                 seed=args.seed, storage_path=args.storage,
                 max_concurrent_trials=args.concurrent)

    best = results.get_best_result(metric="val_loss", mode="min")
    print("\nBest val_loss:", best.metrics["val_loss"])
    print("Best config:", best.config)
