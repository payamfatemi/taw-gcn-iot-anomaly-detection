# Changelog

All notable changes to this research software are documented in this file.

## 0.1.0 - 2026-07-15

### Added

- Initial research-software release of TAW-GCN.
- Dataset adapters for IoT-23, TON_IoT, and the synthetic demo dataset.
- Canonical traffic schema generation and binary label mapping.
- Leakage-aware train, validation, and test splitting.
- Train-only fitting of imputers, categorical encoders, and numerical scalers.
- Sparse topology-aware weighted graph construction.
- Temporal proximity, communication dependency, and behavioral similarity scoring.
- Weighted graph convolutional representation learning.
- Binary anomaly classification with configurable decision threshold.
- Adam optimization, early stopping, checkpointing, and training-history export.
- Accuracy, Precision, Recall, F1-score, confusion matrix, and prediction export.
- Runtime profiling, graph statistics, experiment manifests, and SHA-256 checksums.
- Internal ablation-study configurations and experiment drivers.
- Unit, integration, regression, and smoke tests.
- Reproducibility and paper-to-code mapping documentation.
- CITATION.cff and Zenodo deposition metadata.

### Validation

- All 10 automated tests passed.
- Ruff linting passed.
- Mypy validation passed for 115 source files.
- The end-to-end demo pipeline completed successfully.
- Generated artifact checksums were verified successfully.
- Runtime warnings were resolved.
