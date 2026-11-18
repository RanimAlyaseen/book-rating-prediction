from __future__ import annotations

import json
import subprocess
import sys


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
