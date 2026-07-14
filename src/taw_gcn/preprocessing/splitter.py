from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _distribution_error(frame: pd.DataFrame, indices: np.ndarray, target_ratio: float) -> float:
    full_rate = float(frame["label"].mean())
    split_rate = float(frame.iloc[indices]["label"].mean()) if len(indices) else 0.0
    size_error = abs(len(indices) / len(frame) - target_ratio)
    class_error = abs(split_rate - full_rate)
    return size_error + class_error


def _best_group_split(
    frame: pd.DataFrame,
    groups: pd.Series,
    train_ratio: float,
    random_state: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state + offset)
        train_idx, other_idx = next(splitter.split(frame, frame["label"], groups))
        error = _distribution_error(frame, train_idx, train_ratio)
        if best is None or error < best[0]:
            best = (error, train_idx, other_idx)
    assert best is not None
    return best[1], best[2]


def split_frame(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    strategy = config.get("strategy", "grouped_stratified")
    train_ratio = float(config["train_ratio"])
    val_ratio = float(config["validation_ratio"])
    test_ratio = float(config["test_ratio"])
    seed = int(config.get("random_state", 42))
    attempts = int(config.get("search_attempts", 32))

    used_strategy = strategy
    group_columns = [column for column in config.get("group_columns", []) if column in frame.columns]
    eligible_group_columns = [column for column in group_columns if frame[column].nunique(dropna=False) >= 3]
    # Use one highest-priority grouping key to guarantee that a capture/scenario
    # cannot be split across subsets through a more granular composite key.
    valid_group_columns = eligible_group_columns[:1]

    if strategy == "time":
        ordered = frame.sort_values(config.get("time_column", "timestamp")).reset_index(drop=True)
        train_end = int(len(ordered) * train_ratio)
        val_end = train_end + int(len(ordered) * val_ratio)
        train = ordered.iloc[:train_end]
        validation = ordered.iloc[train_end:val_end]
        test = ordered.iloc[val_end:]
    elif strategy == "grouped_stratified" and valid_group_columns:
        groups = frame[valid_group_columns].astype(str).agg("|".join, axis=1)
        train_idx, other_idx = _best_group_split(frame, groups, train_ratio, seed, attempts)
        train = frame.iloc[train_idx]
        remaining = frame.iloc[other_idx].reset_index(drop=True)
        remaining_groups = groups.iloc[other_idx].reset_index(drop=True)
        relative_val = val_ratio / (val_ratio + test_ratio)
        if remaining_groups.nunique() >= 2:
            val_idx, test_idx = _best_group_split(
                remaining, remaining_groups, relative_val, seed + 10_000, attempts
            )
            validation = remaining.iloc[val_idx]
            test = remaining.iloc[test_idx]
        else:
            used_strategy = "stratified_random_fallback"
            validation, test = train_test_split(
                remaining,
                train_size=relative_val,
                random_state=seed,
                stratify=remaining["label"],
            )
    else:
        used_strategy = "stratified_random"
        train, remaining = train_test_split(
            frame,
            train_size=train_ratio,
            random_state=seed,
            stratify=frame["label"],
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        validation, test = train_test_split(
            remaining,
            train_size=relative_val,
            random_state=seed,
            stratify=remaining["label"],
        )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    manifest = {
        "strategy_requested": strategy,
        "strategy_used": used_strategy,
        "random_state": seed,
        "group_columns_used": valid_group_columns,
        "train_records": len(train),
        "validation_records": len(validation),
        "test_records": len(test),
        "train_anomaly_rate": float(train["label"].mean()),
        "validation_anomaly_rate": float(validation["label"].mean()),
        "test_anomaly_rate": float(test["label"].mean()),
    }
    return train, validation, test, manifest
