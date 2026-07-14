from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import torch
from torch import nn

from taw_gcn.evaluation.evaluator import evaluate_model
from taw_gcn.training.checkpoint import save_checkpoint
from taw_gcn.training.early_stopping import EarlyStopping
from taw_gcn.types import GraphData
from taw_gcn.utils.file_io import ensure_dir, write_json

ModelT = TypeVar("ModelT", bound=nn.Module)


def _positive_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = torch.sum(labels == 1).float()
    negatives = torch.sum(labels == 0).float()
    if positives <= 0:
        return torch.tensor(1.0, device=labels.device)
    return negatives / positives


def train_model(
    model: ModelT,
    train_graph: GraphData,
    validation_graph: GraphData,
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[ModelT, pd.DataFrame, dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    training_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    if training_cfg.get("class_weighting") == "balanced":
        loss_function = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(train_graph.y))
    else:
        loss_function = nn.BCEWithLogitsLoss()

    es_cfg = training_cfg.get("early_stopping", {})
    early = EarlyStopping(
        patience=int(es_cfg.get("patience", 20)),
        minimum_delta=float(es_cfg.get("minimum_delta", 0.0)),
        mode=str(es_cfg.get("mode", "max")),
    )
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] = {}
    best_path = output_dir / "best_model.pth"
    start = time.perf_counter()

    for epoch in range(1, int(training_cfg["maximum_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_graph.x, train_graph.edge_index, train_graph.edge_weight)
        loss = loss_function(logits, train_graph.y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        clip = training_cfg.get("gradient_clip_norm")
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()

        validation_metrics, _, _ = evaluate_model(
            model,
            validation_graph,
            float(training_cfg.get("decision_threshold", 0.5)),
            int(config.get("evaluation", {}).get("zero_division", 0)),
        )
        row = {"epoch": epoch, "train_loss": float(loss.item()), **{f"val_{k}": v for k, v in validation_metrics.items()}}
        history.append(row)
        monitor_name = str(es_cfg.get("monitor", "f1"))
        monitor_value = float(validation_metrics[monitor_name])
        improved, should_stop = early.update(monitor_value)
        if improved:
            best_metrics = validation_metrics
            save_checkpoint(best_path, model, optimizer, epoch, validation_metrics, config)
        interval = int(training_cfg.get("save_every_n_epochs", 0))
        if interval > 0 and epoch % interval == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pth", model, optimizer, epoch, validation_metrics, config)
        if bool(es_cfg.get("enabled", True)) and should_stop:
            break

    total_seconds = time.perf_counter() - start
    if best_path.exists():
        payload = torch.load(best_path, map_location=train_graph.x.device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    save_checkpoint(output_dir / "final_model.pth", model, optimizer, len(history), best_metrics, config)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    summary = {
        "epochs_completed": len(history),
        "training_seconds": total_seconds,
        "best_validation_metrics": best_metrics,
        "best_checkpoint": str(best_path),
    }
    write_json(summary, output_dir / "training_summary.json")
    return model, history_frame, summary
