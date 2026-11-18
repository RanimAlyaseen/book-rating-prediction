"""Train a clearly labeled demonstration model from bundled synthetic rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import run


def run_demo(
    repository_root: Path | None = None,
    output_root: Path | None = None,
    model_out: Path | None = None,
) -> dict[str, object]:
    """Run the synthetic workflow without writing research reports or figures."""

    root = repository_root or Path(__file__).resolve().parents[2]
    demo_outputs = output_root or root / "demo-output"
    demo_model = model_out or root / "models" / "book_rating_demo_pipeline.joblib"
    args = argparse.Namespace(
        data=str(root / "data" / "synthetic_books.csv"),
        model_out=str(demo_model),
        metrics_out=str(demo_outputs / "demo-metrics.json"),
        figures_dir=str(demo_outputs / "figures"),
        report_out=str(demo_outputs / "demo-results.md"),
        quick=True,
        data_origin="synthetic_demo",
    )
    return run(args)


def main() -> None:
    metrics = run_demo()
    print(
        json.dumps(
            {
                "data_origin": metrics["data_origin"],
                "selected_model": metrics["selected_model"],
                "model_path": "models/book_rating_demo_pipeline.joblib",
                "outputs": "demo-output/",
                "warning": (
                    "Synthetic outputs verify the software workflow only and do not "
                    "establish real-world performance."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
