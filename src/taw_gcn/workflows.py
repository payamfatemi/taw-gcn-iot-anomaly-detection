from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from taw_gcn.evaluation.evaluator import evaluate_model
from taw_gcn.graph.builder import build_graph
from taw_gcn.graph.storage import load_graph, save_graph
from taw_gcn.models.taw_gcn import TAWGCN
from taw_gcn.reporting.artifacts import write_predictions
from taw_gcn.training.trainer import train_model
from taw_gcn.utils.device import resolve_device
from taw_gcn.utils.file_io import ensure_dir, read_table, write_json


def _find_table(directory: Path, stem: str) -> Path:
    for suffix in (".csv", ".parquet", ".tsv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No processed table found for {stem} under {directory}")


def load_feature_columns(run_dir: str | Path) -> list[str]:
    path = Path(run_dir) / "transformers" / "feature_columns.json"
    return list(json.loads(path.read_text(encoding="utf-8")))


def build_graphs_from_processed(config: dict[str, Any], run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    processed_dir = run_dir / "processed"
    graph_dir = ensure_dir(run_dir / "graphs")
    feature_columns = load_feature_columns(run_dir)
    outputs: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        frame = read_table(_find_table(processed_dir, split))
        graph = build_graph(frame, feature_columns, config)
        output = graph_dir / f"{split}_graph.pt"
        save_graph(graph, output)
        write_json(graph.metadata, graph_dir / f"{split}_graph_statistics.json")
        outputs[split] = output
    return outputs


def train_from_graphs(config: dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    train_graph = load_graph(run_dir / "graphs" / "train_graph.pt").to(device)
    validation_graph = load_graph(run_dir / "graphs" / "validation_graph.pt").to(device)
    model = TAWGCN.from_config(train_graph.x.shape[1], config).to(device)
    train_model(model, train_graph, validation_graph, config, run_dir / "checkpoints")
    return run_dir / "checkpoints" / "best_model.pth"


def load_trained_model(config: dict[str, Any], checkpoint: str | Path, input_dim: int, device: torch.device) -> TAWGCN:
    model = TAWGCN.from_config(input_dim, config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def evaluate_from_graphs(
    config: dict[str, Any],
    run_dir: str | Path,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    test_graph = load_graph(run_dir / "graphs" / "test_graph.pt").to(device)
    checkpoint = Path(checkpoint) if checkpoint is not None else run_dir / "checkpoints" / "best_model.pth"
    model = load_trained_model(config, checkpoint, test_graph.x.shape[1], device)
    metrics, probabilities, predictions = evaluate_model(
        model,
        test_graph,
        float(config["training"].get("decision_threshold", 0.5)),
        int(config.get("evaluation", {}).get("zero_division", 0)),
    )
    write_json(metrics, run_dir / "test_metrics.json")
    write_predictions(
        test_graph.record_ids,
        test_graph.y.detach().cpu().numpy().astype(int),
        probabilities,
        predictions,
        run_dir / "test_predictions.csv",
    )
    return metrics
