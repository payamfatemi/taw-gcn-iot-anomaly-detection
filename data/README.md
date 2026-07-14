# Data lifecycle

- `raw/`: immutable source files obtained under the dataset provider's license.
- `interim/`: canonical but not model-ready tables.
- `processed/`: train/validation/test tables after train-only preprocessing.
- `splits/`: split IDs, statistics, leakage reports, and hashes.
- `transformers/`: fitted preprocessing artifacts.
- `graphs/`: sparse inductive or transductive graph artifacts.
- `manifests/`: dataset-level provenance metadata.

Raw datasets and generated artifacts are excluded from Git.
