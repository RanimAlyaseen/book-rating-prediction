"""Trusted local model artifact creation, loading, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .constants import FEATURE_COLUMNS
from .validation import validate_prediction_input


REQUIRED_BUNDLE_KEYS = {
    "pipeline",
    "feature_columns",
    "target_threshold",
    "allowed_categories",
    "versions",
    "source_sha256",
    "data_origin",
}


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    """Serialize only a bundle created by this repository's training command."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a trusted, locally generated model bundle and verify its contract."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model artifact not found: {source}")
    bundle = joblib.load(source)
    if not isinstance(bundle, dict):
        raise ValueError("Model artifact must contain a bundle dictionary")
    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle))
    if missing:
        raise ValueError(f"Model artifact is missing keys: {', '.join(missing)}")
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Model artifact feature order is incompatible")
    pipeline = bundle["pipeline"]
    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        raise ValueError("Model artifact does not contain a fitted classifier")
    return bundle


def predict_one(
    bundle: dict[str, Any], values: dict[str, object]
) -> tuple[int, float]:
    """Validate one request and return class plus positive-class probability."""

    validated = validate_prediction_input(values, bundle["allowed_categories"])
    frame = pd.DataFrame([validated], columns=bundle["feature_columns"])
    prediction = int(bundle["pipeline"].predict(frame)[0])
    probability = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    return prediction, probability
