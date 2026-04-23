# Reproducibility

This document maps each thesis artifact (table, figure, and load-bearing numerical claim cited in the running text) to the file in this repository that supports it and to the code path that produced it.

The authoritative numerical artifacts live under `code/results/` and are indexed in `code/results/README.md`. The scripts index in `code/scripts/README.md` describes entry points and historical helpers. The release snapshot is artifact-first: if prose and helper scripts disagree, treat the stored outputs in `code/results/` as authoritative.

## How to install

```bash
# 1) create a clean Python 3.11+ environment
python -m venv .venv
. .venv/bin/activate         # on Windows: .venv\Scripts\activate

# 2) install PyTorch following the index URL comments in code/requirements.txt
#    for example, CUDA 12.1:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu121

# 3) install the rest of the pinned dependencies
pip install -r code/requirements.txt
```

The checked-in pins match the local pipeline-test environment. The heavy HPO and evaluation reruns reported in the thesis were executed in Google Colab on NVIDIA A100-SXM4-40GB GPUs; rerunning on a different device can produce small numerical differences due to training non-determinism.

## Thesis artifact -> file -> producing path

### Results: Reproduction Validation on wILI Data

| Thesis artifact | Primary source file | Producing path |
|---|---|---|
| Table 4.1 (wILI rerun vs paper) | `code/results/wili_base_eval.json`, `code/results/wili_tuned_eval.json` (`wili_reproduction_table.csv` is a convenience table) | `code/src/tuning/hpo.py` with `src/configs/us_wili_*.json`; evaluation via `code/src/eval/evaluate.py` |
| "rerun within about 10% of paper RMSE" narrative | `code/results/wili_base_eval.json`, `code/results/wili_tuned_eval.json` | same path |

### Results: Baseline and Tuned Model Performance

| Thesis artifact | Primary source file | Producing path |
|---|---|---|
| "base above 190 MAE" narrative | `code/results/us_base_eval.json` | evaluation of paper-default configs via `code/src/eval/evaluate.py` |
| "tuned ColaGNN MAE 89.1" | `code/results/colagnn_us_ensemble_metrics.json` (`ensemble.mae_mean`) | HPO via `code/src/tuning/hpo.py --model colagnn --config src/configs/us_colagnn.json`; ensemble via `code/src/eval/evaluate.py` |
| "tuned EpiGNN MAE 102.9" | `code/results/epignn_us_ensemble_metrics.json` (`ensemble.mae_mean`) | HPO via `code/src/tuning/hpo.py --model epignn --config src/configs/us_epignn.json`; ensemble via `code/src/eval/evaluate.py` |
| Swedish tuning paragraph | `code/results/sweden_base_vs_tuned.csv` | same pipeline with `src/configs/sweden_{colagnn,epignn}.json` |
| Figure 4.1 (bias-variance seed dispersion) | `code/results/us_base_eval.json`, `code/results/colagnn_us_ensemble_metrics.json`, `code/results/epignn_us_ensemble_metrics.json` | `code/src/visualization/bias_variance_seed_dispersion.py` |

Sample-standard-deviation note: the thesis recomputes sample standard deviation across the 11 stored per-seed values (for example `per_seed_mae`) rather than reusing stored population-standard-deviation fields. Both conventions are derivable from the JSON artifacts.

### Results: FluSight Benchmark Comparison

| Thesis artifact | Primary source file | Producing path |
|---|---|---|
| Table 4.2 (vintage-matched ranking) | `code/results/vintage_matched_hub_ranking.csv` | `code/src/eval/flusight_hub.py::evaluate_all_hub_models` |
| Supporting benchmark metadata | `code/results/vintage_matched_comparison.json` | same shared hub scoring path |
| Detailed per (week, horizon, region) WIS values | `code/results/vintage_matched_detail.csv` | same shared hub scoring path |
| FluSight quantile fan chart (if shown) | `code/results/figures/flusight_quantile_fan_chart.png` (+ metadata JSON, if present) | `code/src/visualization/flusight_quantile_fan_chart.py` |

The release snapshot does not bundle the original `retrain_vintage.ipynb` notebook or the intermediate `results/vintage_submissions/` directory used during the full benchmark build. Public readers should therefore treat the stored `code/results/vintage_matched_*` artifacts and the shared scorer in `code/src/eval/flusight_hub.py` as the authoritative benchmark record. Historical FluSight helper scripts remain under `code/scripts/` for transparency, but they are not the thesis-authoritative ranking source.

Horizon alignment in the benchmark contract is `MEx h=2 -> FluSight h=1`, `MEx h=3 -> FluSight h=2`, and `MEx h=4 -> FluSight h=3`. The stored ranking artifacts already reflect that contract.

### Results: Swedish Data Results

| Thesis artifact | Primary source file | Producing path |
|---|---|---|
| "MAE improves from 8.2 to 8.0" narrative | `code/results/cv_colagnn_sweden.json`, `code/results/cv_persistence_sweden.json` | `code/src/eval/cross_validate.py` aggregated via `code/src/eval/report.py` |
| Season-wise Swedish fold statements | `code/results/cv_colagnn_sweden.json`, `code/results/cv_epignn_sweden.json`, `code/results/cv_persistence_sweden.json` | same aggregate CV path; per-season winners are derived from the stored fold records |
| Per-horizon Swedish CV inputs to the newsvendor analysis | `code/results/cv_{colagnn,epignn}_sweden_tuned_h{1,2,3,4}_with_ckpt.json` | `code/scripts/run_sweden_cv_ckpt.py` repeated per horizon |
| Persistence baseline (CV) | `code/results/cv_persistence_sweden.json`, `code/results/cv_persistence_sweden_h{1,2,3}.json` | `code/src/eval/baselines.py` |
| Seasonal-naive baseline (CV) | `code/results/cv_seasonal_sweden.json` | `code/src/eval/baselines.py` |

Files with `_with_ckpt` in the name persist checkpoints or reinference inputs for later capacity-planning runs. They are supporting provenance, not a second metrics authority for the Swedish aggregate claims.

### Results: Capacity-Planning Results

| Thesis artifact | Primary source file | Producing path |
|---|---|---|
| Table 4.3 (Sweden multihorizon pooled) | `code/results/newsvendor_sweden_multihorizon.csv` (rows with `scope = pooled`) | `code/src/capacity/newsvendor.py::evaluate_newsvendor` applied to the fold-wise post-hoc-calibrated Swedish forecasts |
| "Persistence wins zero of six folds at h=3" | `code/results/sweden_fold_winners_h3.csv` (compact view) and `code/results/newsvendor_sweden_multihorizon.csv` (raw fold rows) | same pipeline |
| Figure 4.2 (Swedish long-horizon sweep) | `code/results/figures/newsvendor_sweden_multihorizon_panel_c.png` | `code/src/visualization/replot_sweden_multihorizon.py` |
| Table 4.4 (US newsvendor cross-check) | `code/results/newsvendor_initial.csv` (rows with `region = NATIONAL`) | `code/src/capacity/newsvendor.py` applied to the stored MEx and FluSight quantile outputs on the 2024/25 holdout |
| Figure 4.3 (US 3:1 under/over decomposition) | `code/results/newsvendor_initial.csv` (`ratio = 3:1`, `region = NATIONAL`) | `code/src/visualization/replot_us_newsvendor.py` |
| Pooled service-level claims | `code/results/newsvendor_sweden_multihorizon.csv` (`service_level` column, pooled rows) | derived in-place by the newsvendor pipeline |

### Other figures cited in the thesis

| Figure | File | Script |
|---|---|---|
| US season overlay | `code/results/figures/us_season_overlay.png` | `code/src/visualization/seasonal_trends.py` |
| Sweden season overlay | `code/results/figures/sweden_season_overlay.png` | `code/src/visualization/seasonal_trends.py` |

## Reproducing a headline result end to end

Example: reproducing the Table 4.3 entry "MEx-ColaGNN-SE at h=3 pooled 3:1 cost = 118,823".

1. Train the tuned ColaGNN model on Sweden CV with checkpoints:

   ```bash
   cd code
   python -m scripts.run_sweden_cv_ckpt --model colagnn \
       --config src/configs/sweden_colagnn_h3.json \
       --seeds 42 123 456 777 1 2 3 4 5 6 7
   ```

   Produces `results/cv_colagnn_sweden_tuned_h3_with_ckpt.json` and per-fold checkpoints in `results/checkpoints/`.

2. Run the newsvendor evaluation across horizons and cost ratios:

   ```bash
   python -m src.capacity.newsvendor
   ```

   Produces `results/newsvendor_sweden_multihorizon.csv`.

3. Read the pooled value for the target row:

   ```bash
   awk -F, '$1 == "Sweden-CV-multihorizon" && $2 == "MEx-ColaGNN-SE" \
         && $3 == 3 && $4 == "POOLED" && $5 == "3:1" { print $9 }' \
       results/newsvendor_sweden_multihorizon.csv
   ```

   Expected output: `118823.3636856416`.

## Tests

See `code/tests/README.md` for the test-suite scope and run instructions. In short:

```bash
cd code
pytest tests/
```

## Data dependencies and omitted intermediates

- FluSight hub archive. Clone from `https://github.com/cdcepi/FluSight-forecast-hub` and place the archive at `data/FluSight-forecast-hub/` before running the scripts that need it. The evaluation uses the hub's authoritative `target-hospital-admissions.csv` as ground truth.
- Raw NHSN CSV dumps. The repository ships the converted matrix (`code/src/data/state_flu_admissions.txt`) rather than the raw multi-metric CSVs. To refresh from source, download the most recent Weekly Hospital Respiratory Data CSV from the link in `code/src/data/README.md` and rerun `nhsn_converter.py`.
- Raw Folkhalsomyndigheten CSV. Same pattern: the repository ships the converted matrix; the raw wide-format CSV is fetched from the link in `code/src/data/README.md` and converted with `sweden_converter.py`.
- The release snapshot does not bundle the original vintage-matched retraining notebook or the intermediate submission directory used during the full FluSight rerun. For the public benchmark record, use `code/results/vintage_matched_*` plus `code/src/eval/flusight_hub.py`.

## Scope and limitations of the reproducibility claim

The repository reproduces:

- every numerical cell in Tables 4.1, 4.2, 4.3, and 4.4 of the thesis from the files cited above;
- the fold-level Swedish newsvendor statements, including the "persistence wins zero of six folds at h=3" claim;
- the FluSight 2024/25 ranking (Rel. WIS) at the pooled aggregation used in the thesis.

The repository does not reproduce:

- the interview-driven cost parameters or planning-lead-time mapping in the Swedish operational analysis; these are qualitative inputs that live outside the code pipeline;
- a live FluSight participation (the benchmark is retrospective).
