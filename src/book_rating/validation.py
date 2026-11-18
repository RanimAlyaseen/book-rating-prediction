"""Prediction-input validation shared by the UI and tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .constants import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


class InputValidationError(ValueError):
    """Raised when a prediction request is incomplete or outside safe ranges."""


NUMERIC_LIMITS = {
    "year": (1400, 2100),
    "price": (0.01, 1000.0),
    "num_reviews": (0, 10_000_000),
    "num_pages": (1, 10_000),
}


def validate_prediction_input(
    values: Mapping[str, object],
    allowed_categories: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Validate and normalize one model input without depending on column order."""

    missing = [column for column in FEATURE_COLUMNS if column not in values]
    if missing:
        raise InputValidationError(f"Missing input fields: {', '.join(missing)}")

    validated: dict[str, object] = {}
    for column in NUMERIC_FEATURES:
        try:
            number = float(values[column])
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"{column} must be numeric") from exc
        lower, upper = NUMERIC_LIMITS[column]
        if not lower <= number <= upper:
            raise InputValidationError(
                f"{column} must be between {lower:g} and {upper:g}"
            )
        if column in {"year", "num_reviews", "num_pages"}:
            if not number.is_integer():
                raise InputValidationError(f"{column} must be a whole number")
            validated[column] = int(number)
        else:
            validated[column] = number

    for column in CATEGORICAL_FEATURES:
        text = str(values[column]).strip()
        if not text:
            raise InputValidationError(f"{column} must not be blank")
        if allowed_categories and column in allowed_categories:
            if text not in set(allowed_categories[column]):
                raise InputValidationError(f"Unknown {column}: {text}")
        validated[column] = text

    return {column: validated[column] for column in FEATURE_COLUMNS}
