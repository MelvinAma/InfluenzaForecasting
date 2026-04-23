# MEx Influenza Forecasting

Public snapshot for the KTH dual-degree master's thesis:

**Using Spatio-Temporal Influenza Forecasts to Support Regional Healthcare Capacity Planning**
Hyperparameter-Tuned Graph Neural Networks for Influenza Prediction and Capacity Analysis.

Prepared as a KTH dual-degree master's thesis project in computer science and industrial engineering/management.

This repository accompanies the thesis PDF and contains the code, configurations, stored result artifacts, and tests that support the tables, figures, and numerical claims in the text. It is intended to let an external reviewer inspect the implementation, trace every reported number back to the artifact that produced it, and rerun the pipeline end to end if desired.

## Package scope

Included:

- the forecasting, evaluation, tuning, and capacity-planning code;
- the stored result files (JSON and CSV) cited by the thesis;
- the figures referenced in the thesis PDF;
- automated tests covering the data pipeline, model loaders, quantile math, evaluation, and newsvendor capacity planning.

Not included:

- the LaTeX thesis source (the thesis PDF is distributed separately);
- private planning notes and internal drafts;
- raw multi-gigabyte training checkpoints;
- the cloned FluSight hub archive (this is fetched separately; see `REPRODUCIBILITY.md`).

This is a one-time submission snapshot. It is not maintained as an active project repository.

## Top-level layout

```
PublicRepo/
  README.md                 <- this file
  REPRODUCIBILITY.md        <- thesis artifact to file mapping, how to rerun
  CONFIGS.md                <- HPO search spaces and selected best configurations
  code/
    requirements.txt        <- pinned dependencies (install PyTorch separately)
    src/
      data/                 <- data converters, adjacency matrices, matrices in EpiGNN/ColaGNN format
      configs/              <- per-model per-dataset training configurations
      eval/                 <- evaluation pipeline (WIS, MAE, RMSE, PCC, FluSight hub scoring)
      tuning/               <- Ray Tune HPO runner (hpo.py) and training wrapper (runner.py)
      capacity/             <- newsvendor decision-rule evaluation
      visualization/        <- scripts that regenerate the figures cited in the thesis
    scripts/                <- entry-point scripts (FluSight reruns, Sweden CV launcher, ...)
    results/                <- stored outputs cited in the thesis (JSON and CSV) plus figures/
    tests/                  <- pytest test suite
    models/
      ColaGNN/              <- vendored upstream ColaGNN code (Deng et al., CIKM 2020)
      EpiGNN/               <- vendored upstream EpiGNN  code (Xie et al., IJCAI 2022)
```

## Where to start

- Read the thesis PDF first. It is the primary document; this repository supports it.
- To see how thesis tables and figures map to specific files here, open `REPRODUCIBILITY.md`.
- To see which hyperparameters were searched and which were selected, open `CONFIGS.md`.
- To browse the result artifacts by thesis section, open `code/results/README.md`.
- To see what each entry-point script does, open `code/scripts/README.md`.
- To run the tests, open `code/tests/README.md`.
- For data provenance and matrix shapes, open `code/src/data/README.md`.

## Install

```bash
# create a clean Python 3.11+ environment
python -m venv .venv
. .venv/bin/activate         # on Windows: .venv\Scripts\activate

# install PyTorch following the index URL comments in code/requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu121

# install the rest of the pinned dependencies
pip install -r code/requirements.txt
```

Notes:

- The pins in `code/requirements.txt` match the local development environment. The authoritative HPO and evaluation reruns described in the thesis Implementation chapter were executed in Google Colab on NVIDIA A100-SXM4-40GB GPUs. Rerunning on a different device or library version can produce small numerical differences from training non-determinism.
- `numpy` is pinned to `1.26.4` because the vendored model code depends on the pre-2.0 numpy API and because `scipy==1.10.1` breaks under numpy 2.x.
- `torch-geometric` is not required. Both EpiGNN and ColaGNN implement their graph operations in plain PyTorch inside `code/models/`.

## Run the tests

```bash
cd code
pytest tests/
```

See `code/tests/README.md` for test-suite coverage.

## Citing

If you cite any of this work, please cite the thesis itself rather than this repository. The repository snapshot is intended as supporting evidence for the thesis text, not as a standalone contribution.

## Attribution

The two model implementations under `code/models/` are vendored copies of published open-source research code:

- ColaGNN: Deng, Wang, Rangwala, Wang, Ning, "Cola-GNN: Cross-location Attention Based Graph Neural Networks for Long-term ILI Prediction", CIKM 2020. Upstream: `https://github.com/amy-deng/colagnn`.
- EpiGNN: Xie, Zhang, Li, Zhou, Zuo, "EpiGNN: Exploring Spatial Transmission with Graph Neural Network for Regional Epidemic Forecasting", IJCAI 2022. Upstream: `https://github.com/Xiefeng69/EpiGNN`.

Where the thesis adapts files inside these directories, the adaptations are marked inline with `MEx Thesis Adaptation` blocks. Both directories retain their original upstream READMEs; the MEx-specific adaptations are described in the thesis Implementation chapter, not in those READMEs.

FluSight benchmark submissions, the `target-hospital-admissions.csv` ground truth, and the scoring conventions are from the CDC FluSight-forecast-hub repository (`https://github.com/cdcepi/FluSight-forecast-hub`), used under the hub's public terms.

US surveillance data is from the CDC National Healthcare Safety Network; Swedish surveillance data is from Folkhalsomyndigheten. Links and column conventions are documented in `code/src/data/README.md`.
