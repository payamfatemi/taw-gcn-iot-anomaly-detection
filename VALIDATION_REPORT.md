# Validation report

The packaged reference implementation was validated before archive creation.

- Python package installed successfully in editable mode with `--no-deps`.
- Package import returned version `0.1.0`.
- Unit, integration, regression, and smoke suite: **10 tests passed**.
- Python source compilation with `compileall`: successful.
- End-to-end demo: data generation, preprocessing, grouped splitting, train-only transformation, graph construction, training, evaluation, checkpointing, prediction export, and manifests completed successfully.
- Run-artifact checksum verification: successful.
- `CITATION.cff`, `.zenodo.json`, and all YAML configurations parsed successfully.

The raw IoT-23 and TON_IoT datasets were not included and therefore dataset-specific numerical results were not claimed or validated in this archive.
