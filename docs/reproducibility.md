# Reproducibility

Each run stores:

- resolved YAML configuration;
- Python, operating-system, PyTorch, CUDA, and Git metadata;
- dataset and split manifests;
- graph statistics;
- best and final checkpoints;
- epoch-level training history;
- test probabilities and labels;
- runtime profile;
- SHA-256 checksums.

Use `scripts/verify_reproducibility.py <run_dir>` to validate artifact integrity. Deterministic operations can reduce performance and may not be available for every hardware/kernel combination; warnings are retained rather than hidden.
