"""Sweden CV launcher that persists per-fold checkpoints.

For each seed in --seeds, runs cv_colagnn or cv_epignn on the tuned HPO
config and saves best_state per fold to
results/checkpoints/cv_<model>_sweden_tuned/<test-season>_seed<n>.pt.

The checkpoint dicts are produced by cross_validate._save_fold_checkpoint
(added as a MEx Thesis Adaptation). A separate notebook is expected to
iterate over these checkpoints and produce the per-fold newsvendor
evaluation for Stage 3 capacity planning.
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.eval.cross_validate import cv_colagnn, cv_epignn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["colagnn", "epignn"])
    p.add_argument(
        "--seeds", type=int, nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 42, 123, 456, 777],
    )
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--onset_threshold", type=float, default=66.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--min_train_seasons", type=int, default=2)
    args = p.parse_args()

    cfg_path = str(PROJ_ROOT / "src" / "configs" / f"sweden_{args.model}.json")
    hpo_path = PROJ_ROOT / "results" / f"{args.model}_sweden_best_config.json"
    with open(hpo_path) as f:
        hpo = json.load(f)["config"]
    ckpt_dir = str(PROJ_ROOT / "results" / "checkpoints" /
                   f"cv_{args.model}_sweden_tuned")

    fn = cv_colagnn if args.model == "colagnn" else cv_epignn

    print(f"model={args.model}  seeds={args.seeds}  epochs={args.num_epochs}  "
          f"patience={args.patience}  device={args.device}", flush=True)
    print(f"cfg={cfg_path}", flush=True)
    print(f"ckpt_dir={ckpt_dir}", flush=True)
    print(f"hpo={hpo}", flush=True)
    print("", flush=True)

    per_seed = []
    overall_start = time.time()
    for i, seed in enumerate(args.seeds, 1):
        t0 = time.time()
        print(f"=== [{i}/{len(args.seeds)}] {args.model} seed={seed} ===",
              flush=True)
        res = fn(
            cfg_path, hpo,
            num_epochs=args.num_epochs, patience=args.patience,
            onset_threshold=args.onset_threshold,
            seed=seed, device=args.device,
            min_train_seasons=args.min_train_seasons,
            ckpt_dir=ckpt_dir,
        )
        elapsed = time.time() - t0
        agg = res["aggregate"]
        mae = agg.get("mae_mean")
        rmse = agg.get("rmse_mean")
        print(f"seed={seed}  elapsed={elapsed/60:.1f} min  "
              f"folds={len(res['folds'])}  "
              f"mae_mean={mae:.3f}  rmse_mean={rmse:.3f}", flush=True)
        per_seed.append({
            "seed": seed,
            "elapsed_sec": elapsed,
            "folds": res["folds"],
            "aggregate": agg,
        })

    out_json = PROJ_ROOT / "results" / f"cv_{args.model}_sweden_tuned_with_ckpt.json"
    with open(out_json, "w") as f:
        json.dump(per_seed, f, indent=2, default=str)

    total = time.time() - overall_start
    print(f"\nAll seeds complete. total={total/60:.1f} min ({total/3600:.2f} h)",
          flush=True)
    print(f"Results: {out_json}", flush=True)
    print(f"Checkpoints: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
