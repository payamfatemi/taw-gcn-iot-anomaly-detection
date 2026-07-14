# Dataset preparation protocol

1. Obtain the dataset from its official provider.
2. Preserve source files unchanged under `data/raw/<dataset>/`.
3. Configure column aliases in `configs/datasets/<dataset>.yaml`.
4. Run the adapter and inspect `dataset_manifest.json`.
5. Reject missing labels, unparseable timestamps, duplicate record IDs, and single-class datasets.
6. Keep source/destination addresses as graph metadata; do not treat them as ordinary numerical magnitudes.
7. Fit preprocessing artifacts only on the training split.
8. Store split IDs and SHA-256 hashes.

Dataset licenses and redistribution restrictions remain applicable.
