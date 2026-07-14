from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from taw_gcn.constants import METADATA_COLUMNS
from taw_gcn.data.factory import create_adapter
from taw_gcn.data.integrity_checker import validate_canonical_frame
from taw_gcn.preprocessing.cleaner import clean_frame
from taw_gcn.preprocessing.feature_transformer import fit_transformer, transform_frame
from taw_gcn.preprocessing.leakage_checker import check_split_leakage
from taw_gcn.preprocessing.splitter import split_frame
from taw_gcn.types import PreparedSplits
from taw_gcn.utils.file_io import ensure_dir, write_json, write_table
from taw_gcn.utils.hashing import sha256_strings


def prepare_dataset(config: dict[str, Any], raw: str | Path, run_dir: str | Path) -> PreparedSplits:
    run_dir = ensure_dir(run_dir)
    adapter = create_adapter(config)
    canonical = adapter.adapt(raw)
    integrity = validate_canonical_frame(canonical)
    canonical, cleaning = clean_frame(canonical, config["preprocessing"])
    train_raw, val_raw, test_raw, split_manifest = split_frame(canonical, config["split"])
    leakage = check_split_leakage(
        train_raw,
        val_raw,
        test_raw,
        group_columns=split_manifest.get("group_columns_used", []),
    )

    fitted = fit_transformer(train_raw, config["preprocessing"])
    metadata_columns = list(METADATA_COLUMNS)
    train = transform_frame(train_raw, fitted, metadata_columns)
    validation = transform_frame(val_raw, fitted, metadata_columns)
    test = transform_frame(test_raw, fitted, metadata_columns)

    dataset_name = config["dataset"]["name"]
    processed_dir = ensure_dir(run_dir / "processed")
    output_format = config["preprocessing"].get("output_format", "csv")
    write_table(train, processed_dir / "train", output_format)
    write_table(validation, processed_dir / "validation", output_format)
    write_table(test, processed_dir / "test", output_format)

    transformer_dir = ensure_dir(run_dir / "transformers")
    joblib.dump(fitted, transformer_dir / "feature_transformer.joblib")
    write_json(fitted.output_feature_columns, transformer_dir / "feature_columns.json")
    write_json(
        {
            "dataset": dataset_name,
            "integrity": integrity,
            "cleaning": cleaning,
            "split": split_manifest,
            "leakage": leakage,
            "feature_columns": fitted.output_feature_columns,
            "source_numeric_columns": fitted.source_numeric_columns,
            "source_categorical_columns": fitted.source_categorical_columns,
            "split_hashes": {
                "train": sha256_strings(train["record_id"].astype(str)),
                "validation": sha256_strings(validation["record_id"].astype(str)),
                "test": sha256_strings(test["record_id"].astype(str)),
            },
        },
        run_dir / "dataset_manifest.json",
    )
    return PreparedSplits(train, validation, test, fitted.output_feature_columns, processed_dir)
