from __future__ import annotations

import pandas as pd


def stratified_sample(frame: pd.DataFrame, records: int, seed: int) -> pd.DataFrame:
    if records > len(frame):
        raise ValueError(f"Requested {records} records but only {len(frame)} are available.")
    fractions = frame.groupby("label", group_keys=False).apply(
        lambda group: group.sample(
            n=max(1, round(records * len(group) / len(frame))),
            random_state=seed,
        )
    )
    if len(fractions) > records:
        fractions = fractions.sample(n=records, random_state=seed)
    elif len(fractions) < records:
        remainder = frame.drop(index=fractions.index).sample(n=records - len(fractions), random_state=seed)
        fractions = pd.concat([fractions, remainder])
    return pd.DataFrame(fractions.sample(frac=1.0, random_state=seed).reset_index(drop=True))
