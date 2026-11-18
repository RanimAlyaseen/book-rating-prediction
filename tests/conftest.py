from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from book_rating.demo import run_demo


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    genres = ["Fiction", "Non Fiction", "Childrens"]
    sub_genres = ["Adventure", "Business", "Education", "Mystery", "Self-Help"]
    for index in range(120):
        positive = index % 3 != 0
        name = "Repeated Synthetic Edition" if index in {1, 2} else f"Synthetic Book {index:03d}"
        rows.append(
            {
                "book_id": index + 1,
                "name": name,
                "author": f"Synthetic Author {index % 17:02d}",
                "rating": 4.7 if positive else 4.3,
                "year": 2005 + index % 19,
                "genre": genres[index % len(genres)],
                "price": round(5 + rng.uniform(0, 25), 2),
                "num_reviews": int(rng.lognormal(6.5 if positive else 5.5, 0.7)),
                "num_pages": int(rng.integers(120, 700)),
                "sub_genre": sub_genres[index % len(sub_genres)],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    temporary = tmp_path_factory.mktemp("public-demo")
    output_root = temporary / "outputs"
    model_out = temporary / "book_rating_demo_pipeline.joblib"
    metrics = run_demo(root, output_root, model_out)
    return {
        "metrics": metrics,
        "output_root": output_root,
        "model": model_out,
    }
