from __future__ import annotations

import pytest

from book_rating.validation import InputValidationError, validate_prediction_input


VALID = {
    "year": 2018,
    "price": 12.99,
    "num_reviews": 250,
    "num_pages": 320,
    "genre": "Fiction",
    "sub_genre": "Adventure",
}


def test_input_normalization_uses_fixed_feature_order():
    shuffled = dict(reversed(list(VALID.items())))
    validated = validate_prediction_input(shuffled)
    assert list(validated) == [
        "year",
        "price",
        "num_reviews",
        "num_pages",
        "genre",
        "sub_genre",
    ]


def test_missing_input_is_rejected():
    values = dict(VALID)
    values.pop("price")
    with pytest.raises(InputValidationError, match="Missing input fields: price"):
        validate_prediction_input(values)


def test_numeric_range_is_enforced():
    values = dict(VALID, price=0)
    with pytest.raises(InputValidationError, match="price must be between"):
        validate_prediction_input(values)


def test_unknown_controlled_category_is_rejected():
    allowed = {"genre": ["Fiction"], "sub_genre": ["Adventure"]}
    values = dict(VALID, genre="Unknown")
    with pytest.raises(InputValidationError, match="Unknown genre"):
        validate_prediction_input(values, allowed)
