from pathlib import Path

from taw_gcn.data.demo_generator import generate_demo_frame
from taw_gcn.pipeline import run_pipeline


def test_end_to_end_pipeline(tmp_path: Path, config) -> None:
    raw = tmp_path / "demo.csv"
    generate_demo_frame(180, 7).to_csv(raw, index=False)
    config["paths"]["output_root"] = str(tmp_path / "outputs")
    config["training"]["maximum_epochs"] = 2
    config["graph"]["minimum_average_degree"] = 0.0
    run_dir = run_pipeline(config, raw, "integration")
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "checkpoints/best_model.pth").exists()
    assert (run_dir / "test_predictions.csv").exists()
    assert (run_dir / "checksums.sha256").exists()
