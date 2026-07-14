from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taw_gcn.evaluation.evaluator import evaluate_model
from taw_gcn.graph.builder import build_graph
from taw_gcn.graph.storage import save_graph
from taw_gcn.models.taw_gcn import TAWGCN
from taw_gcn.preprocessing.pipeline import prepare_dataset
from taw_gcn.reporting.artifacts import environment_manifest, write_checksums, write_predictions
from taw_gcn.training.trainer import train_model
from taw_gcn.utils.device import resolve_device
from taw_gcn.utils.file_io import ensure_dir, write_json, write_yaml
from taw_gcn.utils.reproducibility import seed_everything
from taw_gcn.utils.timing import timed


def run_pipeline(
    config: dict[str, Any],
    raw: str | Path,
    run_name: str | None = None,
) -> Path:
    seed = int(config["project"].get("seed", 42))
    seed_everything(seed, bool(config["project"].get("deterministic", True)))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_name = config["dataset"]["name"]
    resolved_name = run_name or f"{timestamp}_{dataset_name}_seed{seed}"
    run_dir = ensure_dir(Path(config["paths"]["output_root"]) / resolved_name)
    write_yaml(config, run_dir / "resolved_config.yaml")
    timings: dict[str, float] = {}

    with timed(timings, "preprocessing_seconds"):
        splits = prepare_dataset(config, raw, run_dir)

    graphs = {}
    graph_dir = ensure_dir(run_dir / "graphs")
    with timed(timings, "graph_construction_seconds"):
        for name, frame in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        ):
            graph = build_graph(frame, splits.feature_columns, config)
            graphs[name] = graph
            save_graph(graph, graph_dir / f"{name}_graph.pt")
            write_json(graph.metadata, graph_dir / f"{name}_graph_statistics.json")

    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    graphs = {name: graph.to(device) for name, graph in graphs.items()}
    model = TAWGCN.from_config(len(splits.feature_columns), config)
    model.to(device)
    with timed(timings, "training_and_validation_seconds"):
        model, _, training_summary = train_model(
            model,
            graphs["train"],
            graphs["validation"],
            config,
            run_dir / "checkpoints",
        )

    with timed(timings, "test_evaluation_seconds"):
        metrics, probabilities, predictions = evaluate_model(
            model,
            graphs["test"],
            float(config["training"].get("decision_threshold", 0.5)),
            int(config.get("evaluation", {}).get("zero_division", 0)),
        )
    write_json(metrics, run_dir / "test_metrics.json")
    write_predictions(
        graphs["test"].record_ids,
        graphs["test"].y.detach().cpu().numpy().astype(int),
        probabilities,
        predictions,
        run_dir / "test_predictions.csv",
    )
    write_json(timings, run_dir / "runtime_profile.json")
    write_json(
        {
            "run_name": resolved_name,
            "dataset": dataset_name,
            "seed": seed,
            "device": str(device),
            "status": "completed",
            "environment": environment_manifest(),
            "training": training_summary,
            "test_metrics": metrics,
        },
        run_dir / "experiment_manifest.json",
    )
    write_checksums(run_dir)
    return run_dir
