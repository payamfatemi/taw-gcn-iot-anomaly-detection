from __future__ import annotations

from typing import Any

import pandas as pd

from taw_gcn.exceptions import DataIntegrityError


def check_split_leakage(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> dict[str, Any]:
    ids = {
        "train": set(train["record_id"].astype(str)),
        "validation": set(validation["record_id"].astype(str)),
        "test": set(test["record_id"].astype(str)),
    }
    overlaps = {
        "train_validation_record_overlap": len(ids["train"] & ids["validation"]),
        "train_test_record_overlap": len(ids["train"] & ids["test"]),
        "validation_test_record_overlap": len(ids["validation"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise DataIntegrityError(f"Record leakage detected: {overlaps}")

    group_report: dict[str, int] = {}
    for column in group_columns or []:
        if all(column in split.columns for split in (train, validation, test)):
            values = [set(split[column].astype(str)) for split in (train, validation, test)]
            group_report[f"{column}_train_validation_overlap"] = len(values[0] & values[1])
            group_report[f"{column}_train_test_overlap"] = len(values[0] & values[2])
            group_report[f"{column}_validation_test_overlap"] = len(values[1] & values[2])
    return {**overlaps, **group_report, "labels_used_as_features": False, "transformer_fit_scope": "train_only"}
