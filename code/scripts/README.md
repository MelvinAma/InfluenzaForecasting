# Scripts

Entry-point scripts associated with the stored artifacts under `code/results/`. The release snapshot is artifact-first: cite the stored outputs in `code/results/` before you cite helper scripts.

Example:

```bash
cd code
python -m scripts.run_sweden_cv_ckpt --model colagnn --config src/configs/sweden_colagnn_h3.json
```

The authoritative thesis artifacts are indexed in `code/results/README.md` and mapped to thesis tables and figures in `REPRODUCIBILITY.md` at the repository root.

## Index

### FluSight benchmark helpers (historical/reference)

- `run_aligned_flusight.py` - historical ColaGNN horizon-alignment and pseudo-submission helper. Retained to document the `h=2,3,4 -> FluSight h=1,2,3` mapping, but not the thesis-authoritative ranking source in this public snapshot.
- `run_epignn_flusight.py` - historical EpiGNN counterpart to the aligned FluSight helper. Same caveat as above.
- `run_unaligned_flusight.py` - diagnostic variant without horizon alignment. Retained to show why omitting the offset is invalid.
- `fetch_data_vintages.py` - fetches preliminary NHSN snapshots by FluSight reference date for the vintage analysis workflow.
- `evaluate_vintaged.py` - evaluates a ColaGNN run against per-reference-date preliminary inputs.
- `evaluate_vintaged_2425.py` - quantifies the data-vintage advantage in the 2024/25 FluSight comparison.
- `score_submissions.py` - scores saved pseudo-submission CSVs through the shared hub path. The stored `results/vintage_matched_*` files remain the authoritative public benchmark record.
- `run_multi_horizon.py` - older multi-horizon native-quantile experiment retained for reference. Superseded by the later vintage-matched benchmark workflow.

### Prospective and Weekly Operation

- `weekly_predict.py` - weekly pseudo-submission pipeline for FluSight 2025/26.
- `generate_retroactive.py` - generates retroactive pseudo-submissions for valid 2025/26 reference dates from fixed weights.
- `run_prospective_2526.py` - prospective 2025/26 season evaluation using the stored data snapshot dated 2026-03-05.

### Sweden Evaluation

- `run_sweden_cv_ckpt.py` - launches Swedish leave-one-season-out cross-validation with per-fold checkpoints persisted to `results/checkpoints/cv_<model>_sweden_tuned/`. These checkpointed runs feed the later Swedish newsvendor reinference artifacts; the thesis's aggregate Swedish metrics are still cited from `results/cv_*_sweden.json`.

### Sanity Checks

- `validate_native_quantile.py` - short validation run comparing native-quantile ColaGNN against FluSight-baseline WIS before launching a longer rerun.

## Notes

- The FluSight hub archive (`data/FluSight-forecast-hub/`) is not bundled with this repository. Clone it from `https://github.com/cdcepi/FluSight-forecast-hub` before running the scripts that need it.
- The public release snapshot does not bundle the original vintage-matched retraining notebook or the intermediate submission directory used during the full FluSight rerun. For the benchmark record, use `results/vintage_matched_*` together with `src/eval/flusight_hub.py`.
- Result filenames use the convention `<model>_<dataset>_<metric>.{json,csv}` for per-run artifacts and `cv_<model>_<dataset>[_tuned][_h<k>]_with_ckpt.json` for cross-validation runs that also persisted checkpoints.
- The thesis-to-artifact map is summarized in `REPRODUCIBILITY.md` at the repository root.
