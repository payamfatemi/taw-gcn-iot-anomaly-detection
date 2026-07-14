# Zenodo Release Procedure

This document describes the release procedure for archiving TAW-GCN on Zenodo and obtaining a persistent DOI.

## 1. Verify release metadata

Confirm that the software version and release date are identical in the following files:

- `pyproject.toml`
- `CITATION.cff`
- `.zenodo.json`
- `CHANGELOG.md`
- `ARCHIVE_MANIFEST.json`

For the initial release, the expected version is `0.1.0` and the Git tag is `v0.1.0`.

## 2. Run final quality checks

Run the following commands from the repository root:

    python -m pytest -q
    python -m ruff check .
    python -m mypy src

Expected results:

- 10 tests passed
- Ruff checks passed
- Mypy checks passed for 115 source files

## 3. Validate the demo pipeline

Generate the demo dataset:

    python scripts/generate_demo_data.py --output data/raw/demo/demo.csv --rows 1200

Run the complete demo pipeline:

    python scripts/run_pipeline.py --config configs/base.yaml --config configs/datasets/demo.yaml --raw data/raw/demo/demo.csv --run-name demo_smoke

Verify the generated artifacts:

    python scripts/verify_reproducibility.py outputs/experiments/demo_smoke

The verification result must report:

    "verified": true

## 4. Real-dataset reproducibility

The raw IoT-23 and TON_IoT datasets are not included in the repository.

Before claiming complete reproduction of the article-level numerical results, authorized copies of both datasets must be obtained from their official providers and the corresponding experiment configurations must be executed.

The synthetic demo run validates the software pipeline but does not reproduce the performance values reported in the article.

## 5. Prepare the GitHub repository

Create or verify the following public repository:

    https://github.com/payamfatemi/taw-gcn-iot-anomaly-detection

Commit all release-ready source files and push the `main` branch.

Do not commit:

- raw datasets
- processed datasets
- trained checkpoints
- generated graphs
- experiment outputs
- virtual environments
- credentials or access tokens

## 6. Connect GitHub to Zenodo

Sign in to Zenodo and connect the GitHub account associated with `payamfatemi`.

Synchronize the repository list and enable:

    taw-gcn-iot-anomaly-detection

The repository must be enabled before publishing the GitHub Release.

## 7. Create the version tag

Create the semantic version tag:

    v0.1.0

Push the tag to GitHub.

## 8. Publish the GitHub Release

Create a GitHub Release using the `v0.1.0` tag.

Recommended release title:

    TAW-GCN v0.1.0

The release notes should summarize the initial implementation, validation results, dataset restrictions, and reproducibility scope.

## 9. Verify the Zenodo archive

After the GitHub Release is published, wait for Zenodo to archive it.

Verify the following Zenodo record fields:

- title
- creators
- software version
- publication date
- MIT license
- repository relationship
- uploaded source archive
- version-specific DOI
- concept DOI

## 10. Record the DOI

After Zenodo creates the DOI, add it to the current `main` branch in:

- `README.md`
- `CITATION.cff`
- the GitHub repository description or DOI badge
- the manuscript Code Availability section

Do not rewrite or move the already published `v0.1.0` Git tag.

The DOI can be added to the maintained branch and included in future releases.

## 11. Update the manuscript

Replace the manuscript placeholders:

    [GITHUB REPOSITORY URL]

with:

    https://github.com/payamfatemi/taw-gcn-iot-anomaly-detection

Replace:

    [ZENODO DOI]

with the DOI minted by Zenodo.

## 12. Dataset restriction

Do not upload or redistribute raw IoT-23 or TON_IoT files unless their respective licenses and official providers explicitly permit redistribution.
