# Result Artifacts

Stored outputs behind the thesis tables, figures, and load-bearing numerical claims. For the thesis-to-artifact map, see `REPRODUCIBILITY.md` at the repository root.

All metric values are stored in their native scale: weekly influenza admissions for US NHSN, lab-confirmed cases for Sweden, and weighted ILI percentage for wILI. Files with `_with_ckpt` in the name were produced by runs that also persisted checkpoints or reinference inputs. Unless explicitly stated otherwise, the authoritative metrics source is the aggregate JSON/CSV artifact, not the `_with_ckpt` companion.

## Results Section 4.1: Reproduction Validation on wILI Data

- `wili_base_eval.json` - per-seed wILI rerun values for EpiGNN and ColaGNN, 11 seeds each.
- `wili_tuned_eval.json` - authoritative tuned wILI metrics, 11 seeds.
- `wili_reproduction_table.csv` - compact rerun-vs-paper convenience table used for Table 4.1.
- `wili_base_vs_tuned.csv` - JSON-backed convenience summary.

## Results Section 4.2: Baseline and Tuned Model Performance

- `us_base_eval.json` - paper-default US NHSN runs for both models, 11 seeds, with per-seed MAE, RMSE, MAPE, PCC, and global variants.
- `colagnn_us_ensemble_metrics.json` - tuned US ColaGNN, 11-seed seed-averaged ensemble.
- `epignn_us_ensemble_metrics.json` - tuned US EpiGNN, 11-seed seed-averaged ensemble.
- `colagnn_us_baseline_metrics.csv`, `epignn_us_baseline_metrics.csv` - flat per-seed base-run exports.
- `colagnn_us_tuned_metrics.csv`, `epignn_us_tuned_metrics.csv` - flat tuned per-seed exports, plus `_summary.csv` convenience variants.
- `sweden_base_vs_tuned.csv` - per-seed mean +/- std summary for the Swedish holdout tuning paragraph.
- `sweden_ensemble_metrics.json` - seed-averaged Swedish holdout ensemble metrics.

## Results Section 4.3: FluSight Benchmark Comparison

- `vintage_matched_hub_ranking.csv` - authoritative ranking used in Table 4.2. Scored through `code/src/eval/flusight_hub.py` with the shared hub aggregation path.
- `vintage_matched_comparison.json` - supporting benchmark metadata and model-level summary for the same comparison.
- `vintage_matched_detail.csv` - per (week, horizon, region) WIS values behind the aggregate ranking.
- `flusight_hub_ranking_h{1,2,3,123,combined}.csv` - alternate sensitivity rankings kept for diagnostics, not cited in the main text.
- `wis_comparison.csv`, `wis_our_models.csv`, `wis_per_seed.csv` - WIS diagnostics and supporting summaries.

The public FluSight authority is the stored `vintage_matched_*` output pair plus the shared scorer path, not the historical helper scripts under `code/scripts/`.

## Results Section 4.4: Swedish Data Results

- `cv_colagnn_sweden.json`, `cv_epignn_sweden.json` - authoritative 6-fold leave-one-season-out Swedish CV outputs, with stored fold records and aggregate blocks.
- `cv_persistence_sweden.json` - authoritative persistence baseline for aggregate and season-wise Swedish comparisons.
- `cv_seasonal_sweden.json` - Swedish seasonal-naive (52-week lag) baseline.
- `cv_colagnn_sweden_tuned_with_ckpt.json`, `cv_epignn_sweden_tuned_with_ckpt.json` - checkpointed companions to the aggregate CV runs. These retain per-fold checkpoints for later reinference and are supporting provenance, not a second metrics authority.
- `cv_colagnn_sweden_tuned_h{1,2,3,4}_with_ckpt.json`, `cv_epignn_sweden_tuned_h{1,2,3,4}_with_ckpt.json` - per-horizon checkpointed CV runs used by the Swedish multi-horizon newsvendor evaluation.
- `cv_persistence_sweden_h{1,2,3}.json` - persistence baselines for the horizon-specific operational runs.
- `sweden_per_seed_detail.csv`, `colagnn_sweden_tuned_metrics_*.csv`, `epignn_sweden_tuned_metrics_*.csv` - flat per-seed detail exports.

## Capacity-Planning Results (Section 4.5)

- `newsvendor_sweden_multihorizon.csv` - authoritative source for Table 4.3. Swedish 6-fold LOSO CV at `h in {1, 2, 3, 4}` with under-to-over cost ratios in `{2:1, 3:1, 5:1, 10:1}`, stored as fold-level and pooled rows across 198 weeks and 21 counties.
- `sweden_fold_winners_h3.csv` - compact fold-winner view for the `h=3`, `3:1` Swedish comparison.
- `newsvendor_initial.csv` - authoritative source for Table 4.4. US NHSN 2024/25 holdout newsvendor evaluation for MEx-ColaGNN, MEx-EpiGNN, FluSight-ensemble, and FluSight-baseline across the same cost ratios.
- `newsvendor_sweden_cv.csv`, `newsvendor_sweden_initial.csv` - intermediate Swedish newsvendor files retained for traceability.

## Best HPO Configurations

- `colagnn_{sweden,us,wili}_best_config*.json` - selected ColaGNN best hyperparameters per dataset.
- `epignn_{sweden,us,wili}_best_config*.json` - selected EpiGNN best hyperparameters per dataset.
- `{colagnn,epignn}_sweden_h{1,2,3,4}_best_config.json` - per-horizon best configurations used for the Swedish multi-horizon newsvendor evaluation.

A human-readable summary of the HPO search spaces and the selected best values is in `CONFIGS.md` at the repository root.

## Figures

The `figures/` subdirectory contains the PNGs cited in the thesis PDF (season overlays, bias-variance seed-dispersion plot, Sweden multihorizon panels, US newsvendor breakdown, FluSight quantile fan chart, per-region heatmaps, peak-onset summaries, WIS-per-week diagnostics). Regeneration scripts for the non-obvious panels are under `code/src/visualization/`.

## Stale or Superseded Files

- `colagnn_sweden_best_config.json`, `colagnn_us_best_config.json`, `epignn_sweden_best_config.json`, `epignn_us_best_config.json` - earlier tuning outputs. The `_v2.json` variants supersede these for the US and Sweden main experiments.
- Any `flusight_aligned_comparison.*` file, if present, is earlier and uses a different aggregation. The authoritative FluSight ranking is `vintage_matched_hub_ranking.csv`.
