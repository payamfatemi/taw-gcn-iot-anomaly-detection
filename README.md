# TAW-GCN: Topology-Aware Weighted Graph Convolutional Network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21365109.svg)](https://doi.org/10.5281/zenodo.21365109)

Reference implementation of the proposed **Topology-Aware Weighted Graph Convolutional Network (TAW-GCN)** for binary anomaly detection in IoT traffic. This repository implements only the proposed method; external baselines are intentionally excluded. Internal ablation variants are included.

## Implemented pipeline

1. Dataset-specific adapters for IoT-23 and TON_IoT.
2. Canonical schema creation and binary label mapping.
3. Leakage-aware train/validation/test splitting.
4. Train-only fitting of imputers, categorical encoders, and min-max scalers.
5. Sparse topology-aware graph construction from temporal proximity, communication dependency, and behavioral similarity.
6. Symmetrization, self-loops, and symmetric adjacency normalization.
7. Weighted GCN representation learning and binary anomaly classification.
8. Adam optimization, early stopping, checkpointing, and reproducible evaluation.
9. Accuracy, Precision, Recall, F1-score, confusion matrix, runtime, graph statistics, and prediction export.
10. Standard, data-volume, network-scale, and ablation experiment drivers.

## Important scientific note

The article defines the mathematical parameters but does not provide exact numerical values for all hyperparameters, including `alpha`, `beta`, `gamma`, `tau`, `delta`, hidden dimensions, split ratios, and maximum epochs. Values in `configs/` are therefore documented implementation defaults and must be tuned on the validation split. They are not presented as values reported by the article.

The raw IoT-23 and TON_IoT datasets are not redistributed in this repository. Place authorized copies under `data/raw/` and update the dataset configuration aliases when required by the downloaded release.

## Project layout

```text
configs/                    YAML experiment configurations
data/                       raw -> interim -> processed -> splits -> transformers -> graphs
src/taw_gcn/                reusable Python package
scripts/                    command-line entry points
tests/                      unit, integration, regression, and smoke tests
outputs/                    run-specific models, metrics, predictions, and reports
docs/                       research and reproducibility documentation
.zenodo.json                Zenodo deposition metadata
CITATION.cff                software citation metadata
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

## Quick smoke test

Generate a small canonical demo dataset, preprocess it, build graphs, train, and evaluate:

```bash
python scripts/generate_demo_data.py --output data/raw/demo/demo.csv --rows 1200
python scripts/run_pipeline.py \
  --config configs/base.yaml \
  --config configs/datasets/demo.yaml \
  --raw data/raw/demo/demo.csv \
  --run-name demo_smoke
```

Run tests:

```bash
pytest -q
ruff check .
mypy src
```

## Real datasets

### IoT-23

Place labeled Zeek connection logs or converted CSV/Parquet files under:

```text
data/                       raw -> interim -> processed -> splits -> transformers -> graphs
```

Then run:

```bash
python scripts/run_pipeline.py \
  --config configs/base.yaml \
  --config configs/datasets/iot23.yaml \
  --raw data/raw/iot23 \
  --run-name iot23_standard
```

### TON_IoT

Place the network-traffic subset under:

```text
data/                       raw -> interim -> processed -> splits -> transformers -> graphs
```

Then run:

```bash
python scripts/run_pipeline.py \
  --config configs/base.yaml \
  --config configs/datasets/ton_iot.yaml \
  --raw data/raw/ton_iot \
  --run-name ton_iot_standard
```

## Main outputs

Each run is isolated under `outputs/experiments/<run-name>/` and contains:

- resolved configuration and run manifest;
- processed split statistics and leakage report;
- sparse train, validation, and test graphs;
- best and final model checkpoints;
- training history;
- test metrics and confusion matrix;
- anomaly probabilities and predicted labels;
- runtime and graph statistics;
- SHA-256 checksums.

## Graph definition

For a candidate node pair `(i, j)`:

```text
S_temp = exp(-|t_i - t_j| / tau)
S_comm = 1 when an enabled communication field matches, otherwise 0
S_beh  = cosine_similarity(x_i, x_j)
w_ij   = alpha*S_temp + beta*S_comm + gamma*S_beh
```

The edge is retained when `w_ij >= delta`. Candidate generation prevents dense `O(N^2)` pair enumeration by combining a temporal window, shared communication fields, and behavioral top-k neighbors.

## Reproducibility

- deterministic seeds are set for Python, NumPy, and PyTorch;
- split IDs and hashes are saved;
- preprocessing is fit only on the training split;
- labels are excluded from graph construction;
- resolved configurations and environment information are saved per run;
- no article figure values are hard-coded.

See `docs/reproducibility.md` and `docs/paper_to_code_mapping.md`.

## Code and data availability

- Archived release `v0.1.0`: [10.5281/zenodo.21365109](https://doi.org/10.5281/zenodo.21365109)

The source code is maintained at:

https://github.com/payamfatemi/taw-gcn-iot-anomaly-detection

A versioned archival DOI will be added after the `v0.1.0` release is deposited in Zenodo.

The raw IoT-23 and TON_IoT datasets are not redistributed because of dataset size and licensing considerations. Users must obtain authorized copies from the official dataset providers. This repository includes dataset adapters, configuration files, preparation scripts, and reproducibility documentation.

Generated datasets, trained checkpoints, graphs, predictions, and experiment outputs are excluded from version control by default. Each local experiment stores its resolved configuration, manifests, metrics, runtime information, graph statistics, and SHA-256 checksums under `outputs/experiments/<run-name>/`.

## Citation

If you use this software in academic or research work, please cite the archived software release and the associated TAW-GCN article.

### Software citation

```text
Derkhshanfard, N., Fatemi, S. P., Sefati, S. S., and Mirzaei, A. (2026).
TAW-GCN: Topology-Aware Weighted Graph Convolutional Network for IoT Anomaly Detection
(Version 0.1.0) [Computer software].
https://github.com/payamfatemi/taw-gcn-iot-anomaly-detection
```

The repository also includes machine-readable citation metadata in `CITATION.cff`.

After the `v0.1.0` release is archived in Zenodo, the version-specific DOI and DOI-based citation will be added to this section.

### BibTeX

```bibtex
@software{taw_gcn_2026,
  author    = {Nahideh Derkhshanfard and Seyed Payam Fatemi and Seyed Salar Sefati and Abbas Mirzaei},
  title     = {TAW-GCN: Topology-Aware Weighted Graph Convolutional Network for IoT Anomaly Detection},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/payamfatemi/taw-gcn-iot-anomaly-detection},
  license   = {MIT},
  doi = {10.5281/zenodo.21365109}
}
```

### Associated article

The associated manuscript is titled:

```text
A Topology-Aware Weighted Graph Convolutional Framework for Scalable Anomaly Detection in IoT Networks
```

The final journal name, volume, issue, page range, article DOI, and publication status should be added after the article is officially published.

## License

MIT License. Dataset licenses remain with their original providers.
