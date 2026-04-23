# HPO search spaces and selected best configurations

This document summarizes the hyperparameter search spaces used during tuning and the selected best values per model and dataset. Raw machine-readable configurations sit next to the result artifacts:

- Dataset-level configs: `code/src/configs/*.json` (one per model / dataset / horizon)
- Selected best values: `code/results/*_best_config*.json`

All searches were run with Ray Tune 2.54.0 using the Optuna TPE sampler for exploration and the ASHA scheduler for early stopping of underperforming trials. The trial budget was 150 samples per model per dataset; the ASHA grace period was 25 epochs out of a 500-epoch maximum.

## Dataset-level fixed settings

The fixed part of the configuration — data paths, adjacency paths, split policy, window, horizon — is shared by base and tuned runs and is not part of the search space. The checked-in values are:

| Setting | US NHSN | Sweden |
|---|---|---|
| `data_path` | `src/data/state_flu_admissions.txt` | `src/data/sweden_flu_cases.txt` |
| `adj_path` | `src/data/state-adj-51.txt` | `src/data/sweden-adj-21.txt` |
| `date_index_path` | `src/data/date_index.csv` | `src/data/sweden_week_index.csv` |
| `date_format` | `date` | `week` |
| `window` | 20 | 20 |
| `horizon` (default) | 1 | 1 |
| `split_mode` | `season` | `season` |
| `test_seasons` | `["2024/25"]` | `["2024/25"]` |
| `val_seasons` | `["2023/24"]` | `["2023/24"]` |
| `exclude_seasons` | `["2020/21", "2025/26"]` | `["2020/21", "2025/26"]` |

Horizon extensions for the FluSight benchmark and the Swedish multihorizon newsvendor evaluation reuse the same config files with the horizon parameter overridden to 2, 3, or 4. The per-horizon best configs are stored in `code/results/{model}_sweden_h{k}_best_config.json`.

## EpiGNN search space

Defined in `code/src/tuning/hpo.py` (`EPIGNN_SPACE`).

| Hyperparameter | Type | Search domain | Notes |
|---|---|---|---|
| `lr` | log-uniform | `[1e-4, 1e-2]` | base learning rate |
| `k` | categorical | `{4, 8, 12, 16}` | graph multiplicity used inside EpiGNN |
| `hidP` | categorical | `{1, 2, 4}` | fed into EpiGNN's hidR computation |
| `hidA` | categorical | `{32, 64, 128}` | graph-attention hidden width |
| `hidR` | fixed | `64` | always overwritten inside EpiGNN as `k * 4 * hidP + k`; the kept value is a safe default |
| `n_hidden` | categorical | `{16, 32, 64}` | recurrent hidden width |
| `n_layer` | categorical | `{1, 2}` | number of recurrent layers |
| `n` | categorical | `{1, 2, 3}` | multi-scale convolution depth |
| `dropout` | uniform | `[0.1, 0.5]` | |
| `weight_decay` | log-uniform | `[1e-5, 1e-2]` | |
| `batch_size` | categorical | `{32, 64, 128}` | |
| `res` | fixed | `0` | `res=1` triggers a shape mismatch under the MEx data loader |
| `s` | categorical | `{2, 3}` | scale exponent in multi-scale block |
| `hw` | categorical | `{0, 1}` | highway connection flag |

## ColaGNN search space

Defined in `code/src/tuning/hpo.py` (`COLAGNN_SPACE`).

| Hyperparameter | Type | Search domain | Notes |
|---|---|---|---|
| `lr` | log-uniform | `[1e-4, 1e-2]` | |
| `k` | categorical | `{5, 10, 15, 20}` | neighborhood parameter |
| `hidsp` | categorical | `{8, 15, 32, 64}` | spatial hidden width; reproduction baselines pin `hidsp=10` |
| `n_hidden` | categorical | `{16, 32, 64}` | recurrent hidden width |
| `n_layer` | categorical | `{1, 2}` | number of recurrent layers |
| `dropout` | uniform | `[0.1, 0.5]` | |
| `weight_decay` | log-uniform | `[1e-5, 1e-2]` | |
| `batch_size` | categorical | `{16, 32, 64}` | |
| `rnn_model` | categorical | `{"RNN", "GRU", "LSTM"}` | |
| `bi` | fixed | `False` | `bi=True` causes a tensor-shape mismatch in the ColaGNN forward pass |

## Best configurations selected by HPO

Numeric values are taken directly from the `_best_config*.json` files under `code/results/`. Dashes indicate hyperparameters that are model-specific and not applicable.

### ColaGNN

| Hyperparameter | Sweden | US NHSN | wILI |
|---|---|---|---|
| `lr` | 0.00588 | 0.00748 | 0.00380 |
| `k` | 10 | 20 | 10 |
| `hidsp` | 32 | 32 | 64 |
| `n_hidden` | 16 | 16 | 16 |
| `n_layer` | 2 | 1 | 2 |
| `dropout` | 0.253 | 0.310 | 0.137 |
| `weight_decay` | 1.09e-4 | 1.20e-5 | 1.09e-5 |
| `batch_size` | 32 | 32 | 16 |
| `rnn_model` | GRU | GRU | LSTM |
| `bi` | False | False | False |
| Best val loss | 0.0232 | 0.0766 | - |
| Best epoch | 76 | 61 | - |

Source: `code/results/colagnn_sweden_best_config_v2.json`, `code/results/colagnn_us_best_config_v2.json`, `code/results/colagnn_wili_best_config.json`.

### EpiGNN

| Hyperparameter | Sweden | US NHSN | wILI |
|---|---|---|---|
| `lr` | 0.00835 | 0.00511 | 0.00560 |
| `k` | 8 | 16 | 8 |
| `hidP` | 2 | 2 | 2 |
| `hidA` | 64 | 32 | 128 |
| `hidR` | 64 | 64 | 64 |
| `n_hidden` | 64 | 16 | 32 |
| `n_layer` | 1 | 2 | 2 |
| `n` | 2 | 3 | 3 |
| `dropout` | 0.235 | 0.325 | 0.416 |
| `weight_decay` | 1.20e-5 | 1.94e-3 | 1.38e-5 |
| `batch_size` | 32 | 32 | 64 |
| `res` | 0 | 0 | 0 |
| `s` | 3 | 2 | 3 |
| `hw` | 1 | 1 | 0 |
| Best val loss | 0.00217 | 0.00648 | 0.0148 |
| Best epoch | 85 | 99 | 58 |

Source: `code/results/epignn_sweden_best_config_v2.json`, `code/results/epignn_us_best_config_v2.json`, `code/results/epignn_wili_best_config.json`.

## Per-horizon Swedish configurations

The Swedish multihorizon newsvendor evaluation (Table 4.3 of the thesis) uses horizon-specific configurations stored in `code/results/{model}_sweden_h{1,2,3,4}_best_config.json`. These were produced by rerunning HPO on the Swedish data at each horizon so the reported cost numbers are not extrapolated from a single-horizon fit.

## Notes and divergences from paper defaults

- The released EpiGNN implementation binarizes positive learned edges before graph normalization even though the paper equation implies weighted propagation. The MEx reproduction follows the released code and documents this divergence. `res` is pinned to zero because `res=1` triggers a shape mismatch under the MEx data loader.
- The released ColaGNN implementation uses spatial width 10. The MEx wrapper exposes this as `hidsp`. Reproduction baselines pin `hidsp=10`; tuned runs search over `hidsp` and land at 32 (Sweden, US NHSN) or 64 (wILI).
- The bidirectional-RNN flag `bi` is fixed at `False` because `bi=True` breaks the ColaGNN forward pass.

## How to rerun HPO

From the repository root:

```bash
cd code
python -m src.tuning.hpo --model epignn  --config src/configs/us_epignn.json     --num-samples 150 --max-epochs 500 --gpus 1.0
python -m src.tuning.hpo --model colagnn --config src/configs/us_colagnn.json    --num-samples 150 --max-epochs 500 --gpus 1.0
python -m src.tuning.hpo --model epignn  --config src/configs/sweden_epignn.json  --num-samples 150 --max-epochs 500 --gpus 1.0
python -m src.tuning.hpo --model colagnn --config src/configs/sweden_colagnn.json --num-samples 150 --max-epochs 500 --gpus 1.0
```

The authoritative runs were executed on an NVIDIA A100 in Google Colab. Rerunning on a different device or with different concurrency can produce slightly different selected configurations because of the TPE sampler's sequential dependency and ASHA's early-stopping behavior; the resulting pooled evaluation metrics remain close in practice.
