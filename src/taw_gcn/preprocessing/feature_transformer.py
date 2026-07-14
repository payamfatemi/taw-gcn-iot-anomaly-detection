from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


@dataclass(slots=True)
class FittedFeatureTransformer:
    transformer: ColumnTransformer
    source_numeric_columns: list[str]
    source_categorical_columns: list[str]
    output_feature_columns: list[str]


def infer_feature_columns(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    excluded = set(config.get("exclude_feature_columns", []))
    candidates = [column for column in frame.columns if column not in excluded]
    categorical_candidates = set(config.get("categorical_feature_candidates", []))
    numeric: list[str] = []
    categorical: list[str] = []
    for column in candidates:
        if column in categorical_candidates:
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
        elif frame[column].nunique(dropna=True) <= 64:
            categorical.append(column)
    if not numeric and not categorical:
        raise ValueError("No usable feature columns were inferred.")
    return numeric, categorical


def fit_transformer(train: pd.DataFrame, config: dict[str, Any]) -> FittedFeatureTransformer:
    numeric, categorical = infer_feature_columns(train, config)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=config.get("numeric_imputation", "median"))),
                ("scaler", MinMaxScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=config.get("categorical_imputation", "most_frequent"))),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                ("scaler", MinMaxScaler()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    transformer = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    transformer.fit(train)
    names = [f"num__{column}" for column in numeric] + [f"cat__{column}" for column in categorical]
    return FittedFeatureTransformer(transformer, numeric, categorical, names)


def transform_frame(
    frame: pd.DataFrame,
    fitted: FittedFeatureTransformer,
    metadata_columns: list[str],
) -> pd.DataFrame:
    matrix = fitted.transformer.transform(frame)
    matrix = np.asarray(matrix, dtype=np.float32)
    feature_frame = pd.DataFrame(matrix, columns=fitted.output_feature_columns, index=frame.index)
    metadata = frame[[column for column in metadata_columns if column in frame.columns]].reset_index(drop=True)
    return pd.concat([metadata, feature_frame.reset_index(drop=True)], axis=1)
