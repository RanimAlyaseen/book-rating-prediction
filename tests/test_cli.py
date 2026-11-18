from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from book_rating.artifacts import load_bundle


def test_training_cli_executes_end_to_end(synthetic_frame, tmp_path):
    dataset = tmp_path / "synthetic.csv"
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    report = tmp_path / "experiment.md"
    figures = tmp_path / "figures"
    synthetic_frame.to_csv(dataset, index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "book_rating.train",
            "--data",
            str(dataset),
            "--model-out",
            str(model),
            "--metrics-out",
            str(metrics),
            "--report-out",
            str(report),
            "--figures-dir",
            str(figures),
            "--quick",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert model.stat().st_size > 0
    assert set(payload["models"]) == {
        "dummy_most_frequent",
        "logistic_regression",
        "random_forest",
    }
    assert payload["split"]["title_group_overlap"] == 0
    assert (figures / "confusion_matrix.png").is_file()


def test_public_synthetic_demo_writes_only_separate_outputs(demo_run):
    payload = demo_run["metrics"]
    output_root = demo_run["output_root"]
    bundle = load_bundle(demo_run["model"])

    assert payload["data_origin"] == "synthetic_demo"
    assert bundle["data_origin"] == "synthetic_demo"
    assert (output_root / "demo-metrics.json").is_file()
    assert (output_root / "demo-results.md").is_file()
    assert (output_root / "figures" / "confusion_matrix.png").is_file()
    report = (output_root / "demo-results.md").read_text(encoding="utf-8")
    assert "fabricated demonstration rows" in report
    assert "do not measure real-world" in report


def test_research_command_rejects_bundled_synthetic_fixture(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "book_rating.train",
            "--data",
            str(root / "data" / "synthetic_books.csv"),
            "--model-out",
            str(tmp_path / "model.joblib"),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
            "--report-out",
            str(tmp_path / "report.md"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--quick",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be trained with `python -m book_rating.demo`" in result.stderr
    assert not (tmp_path / "metrics.json").exists()
