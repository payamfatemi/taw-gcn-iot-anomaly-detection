from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if history.empty:
        return
    figure = plt.figure()
    plt.plot(history["epoch"], history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.tight_layout()
    figure.savefig(output_dir / "training_loss.png", dpi=200)
    plt.close(figure)

    if "val_f1" in history.columns:
        figure = plt.figure()
        plt.plot(history["epoch"], history["val_f1"])
        plt.xlabel("Epoch")
        plt.ylabel("Validation F1")
        plt.tight_layout()
        figure.savefig(output_dir / "validation_f1.png", dpi=200)
        plt.close(figure)
