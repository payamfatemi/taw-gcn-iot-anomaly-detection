from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from taw_gcn.exceptions import DataIntegrityError
from taw_gcn.utils.file_io import read_table


class BaseTrafficAdapter:
    """Convert dataset-specific columns into a canonical traffic schema."""

    def __init__(self, dataset_config: dict[str, Any]) -> None:
        self.config = dataset_config
        self.aliases: dict[str, list[str]] = dataset_config.get("aliases", {})
        self.normal_tokens = {str(v).strip().lower() for v in dataset_config.get("normal_label_tokens", [])}

    def discover_files(self, raw: str | Path) -> list[Path]:
        raw_path = Path(raw)
        if raw_path.is_file():
            return [raw_path]
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        files: list[Path] = []
        for pattern in self.config.get("file_globs", ["*.csv"]):
            files.extend(raw_path.rglob(pattern))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No supported files found under {raw_path}")
        return files

    def load(self, raw: str | Path) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in self.discover_files(raw):
            frame = read_table(path)
            frame["__source_file__"] = path.name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False)

    def _resolve(self, frame: pd.DataFrame, canonical: str, required: bool = False) -> str | None:
        for alias in self.aliases.get(canonical, [canonical]):
            if alias in frame.columns:
                return alias
        if required:
            raise DataIntegrityError(
                f"Required field '{canonical}' not found. Tried aliases: {self.aliases.get(canonical, [])}"
            )
        return None

    @staticmethod
    def _safe_series(frame: pd.DataFrame, column: str | None, default: Any = "unknown") -> pd.Series:
        if column is None:
            return pd.Series([default] * len(frame), index=frame.index)
        return frame[column]

    def map_binary_labels(self, values: pd.Series) -> pd.Series:
        def convert(value: Any) -> int:
            if pd.isna(value):
                raise DataIntegrityError("Missing labels are not allowed.")
            token = str(value).strip().lower()
            if token in self.normal_tokens:
                return 0
            try:
                numeric = float(token)
                if numeric == 0.0:
                    return 0
                if numeric == 1.0:
                    return 1
            except ValueError:
                pass
            if any(normal in token for normal in ("benign", "normal")):
                return 0
            return 1

        return values.map(convert).astype(np.int64)

    def adapt(self, raw: str | Path) -> pd.DataFrame:
        frame = self.load(raw).copy()
        timestamp_col = self._resolve(frame, "timestamp", required=True)
        label_col = self._resolve(frame, "label", required=True)
        canonical = frame.copy()

        canonical["timestamp"] = pd.to_numeric(frame[timestamp_col], errors="coerce")
        missing_ts = canonical["timestamp"].isna()
        if missing_ts.any():
            parsed = pd.to_datetime(frame.loc[missing_ts, timestamp_col], errors="coerce", utc=True)
            canonical.loc[missing_ts, "timestamp"] = parsed.astype("int64") / 1e9
        if canonical["timestamp"].isna().any():
            raise DataIntegrityError("Some timestamps could not be parsed.")

        for field, default in (
            ("src_ip", "unknown"),
            ("dst_ip", "unknown"),
            ("src_port", -1),
            ("dst_port", -1),
            ("protocol", "unknown"),
            ("service", "unknown"),
            ("conn_state", "unknown"),
            ("capture_id", None),
            ("scenario_id", "unknown"),
        ):
            column = self._resolve(frame, field, required=False)
            if field == "capture_id" and column is None:
                canonical[field] = frame["__source_file__"].astype(str)
            else:
                canonical[field] = self._safe_series(frame, column, default)

        canonical["raw_label"] = frame[label_col].astype(str)
        canonical["label"] = self.map_binary_labels(frame[label_col])
        # Remove dataset-specific label aliases so attack labels cannot re-enter
        # the feature matrix under another column name.
        label_aliases = set(self.aliases.get("label", []))
        label_aliases.discard("label")
        canonical = canonical.drop(columns=[c for c in label_aliases if c in canonical.columns], errors="ignore")
        source = frame["__source_file__"].astype(str)
        stable = (
            source + "|" + canonical.index.astype(str) + "|" + canonical["timestamp"].astype(str)
        )
        canonical["record_id"] = stable.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:24])
        canonical = canonical.drop(columns=["__source_file__"], errors="ignore")
        return canonical
