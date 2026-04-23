# Tests

Automated tests covering the data pipeline, model loaders, quantile math, evaluation, newsvendor capacity planning, and training adapters.

## Running the tests

From the repository root (`PublicRepo/`):

```bash
pip install -r code/requirements.txt
# install the correct PyTorch wheel separately, see the comments in code/requirements.txt
cd code
pytest tests/
```

A few tests import the vendored model code under `code/models/ColaGNN/src/` and `code/models/EpiGNN/src/`. If PyTorch is not installed or if scipy is missing, those tests are skipped and the data and evaluation tests still pass.

## What each file covers

### `test_data_pipeline.py`

End-to-end validation of Phase 1 data conversion. Covers the US and Swedish converters, adjacency construction, and format validators. Includes factual geographic checks (for example, California borders Nevada, Oregon, and Arizona but not Texas; New York has exactly five neighbors; Alaska, Hawaii, and Gotland are isolated; Stockholm borders Uppsala).

### `test_loader.py`

Tests for the MEx data loader and the shared season-aware split logic. Confirms that train/val/test splits respect season boundaries, that off-season weeks are excluded from all splits, and that normalization statistics are computed from training data only.

### `test_quantile.py`

Tests for `src/eval/quantile.py`. Covers:

- `compute_wis`: known-value checks, decomposition into sharpness and calibration components, shape handling, non-negativity
- `fit_quantile_model`: shape and coverage checks
- Monotonicity enforcement after sorting along the quantile axis

### `test_eval.py`

Tests for the broader evaluation stack: per-horizon pooling, relative WIS against a baseline, MAE / RMSE / PCC computation with both per-region-mean and globally pooled aggregation, horizon alignment helpers.

### `test_newsvendor.py`

Tests for `src/capacity/newsvendor.py`. Covers the critical-ratio formula, the interpolated-quantile stocking decision, the realized-cost decomposition into under and over components, and the pooled evaluation across weeks and regions.

### `test_colagnn.py`

Smoke test for ColaGNN model construction and forward pass, using the vendored upstream code in `code/models/ColaGNN/src/`. Confirms that the adjacency normalization path and the MEx data loader wrapper produce tensors of the expected shape.

### `test_epignn.py`

Smoke test for EpiGNN model construction and forward pass, using the vendored upstream code in `code/models/EpiGNN/src/`. Confirms shape compatibility with the MEx data loader wrapper and exercises the scipy compatibility fix.

### `test_training.py`

Short integration test that runs a few training steps end to end on a tiny input to confirm that the tuning runner, optimizer, loss, and early-stopping wiring all fit together without erroring.

## Scope

These tests are oriented at catching regressions in the MEx pipeline rather than re-testing upstream model behavior. They validate:

- the data converters and adjacency builders against factual geography
- the MEx data loader and split logic
- the evaluation math (WIS, MAE, RMSE, PCC, relative WIS)
- the newsvendor decision rule and cost decomposition
- that the vendored models still construct and run after the adaptation block

They do not reproduce the published HPO or FluSight benchmark runs; those are reproduced by the scripts in `code/scripts/` and the entry points listed in `REPRODUCIBILITY.md` at the repository root.
