"""Dataset loading, validation, and deterministic preparation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from .constants import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN, TARGET_THRESHOLD


class DataValidationError(ValueError):
    """Raised when the training data violates the documented schema."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a supported CSV or Excel training dataset."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise DataValidationError(f"Unsupported dataset format: {suffix}")


def normalize_title(value: object) -> str:
    """Create a stable grouping key for repeated titles or editions."""

    text = str(value).strip().casefold()
    return re.sub(r"\s+", " ", text)


def validate_schema(frame: pd.DataFrame) -> None:
    """Validate required fields and defensible numeric domains."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    required = list(REQUIRED_COLUMNS)
    if frame.loc[:, required].isna().any().any():
        columns = frame.loc[:, required].columns[
            frame.loc[:, required].isna().any()
        ]
        raise DataValidationError(
            f"Missing values in required columns: {', '.join(columns)}"
        )
    if not frame["rating"].between(0, 5).all():
        raise DataValidationError("rating values must be between 0 and 5")
    if not frame["year"].between(1400, 2100).all():
        raise DataValidationError("year values must be between 1400 and 2100")
    if (frame["num_reviews"] < 0).any():
        raise DataValidationError("num_reviews values must be nonnegative")
    if (frame["num_pages"] <= 0).any():
        raise DataValidationError("num_pages values must be positive")
    if frame["genre"].astype("string").str.strip().eq("").any():
        raise DataValidationError("genre values must not be blank")
    if frame["sub_genre"].astype("string").str.strip().eq("").any():
        raise DataValidationError("sub_genre values must not be blank")


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the documented invalid prices and derive target/group columns."""

    validate_schema(frame)
    prepared = frame.loc[frame["price"] > 0].copy()
    if prepared.empty:
        raise DataValidationError("No rows remain after removing nonpositive prices")
    prepared[TARGET_COLUMN] = prepared["rating"].ge(TARGET_THRESHOLD).astype(int)
    prepared["title_group"] = prepared["name"].map(normalize_title)
    if prepared[TARGET_COLUMN].nunique() != 2:
        raise DataValidationError("The derived target must contain both classes")
    return prepared


def split_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features, target, and title groups in a fixed order."""

    return (
        frame.loc[:, list(FEATURE_COLUMNS)].copy(),
        frame[TARGET_COLUMN].copy(),
        frame["title_group"].copy(),
    )
