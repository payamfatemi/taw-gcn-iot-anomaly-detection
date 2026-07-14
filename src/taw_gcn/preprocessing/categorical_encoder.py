from __future__ import annotations

from sklearn.preprocessing import OrdinalEncoder


def make_categorical_encoder() -> OrdinalEncoder:
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
