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


@pytest.mark.parametrize("column", ["year", "num_reviews", "num_pages"])
def test_fractional_integer_fields_are_rejected(column):
    values = dict(VALID)
    values[column] = float(values[column]) + 0.5
    with pytest.raises(InputValidationError, match=f"{column} must be a whole number"):
        validate_prediction_input(values)


def test_integral_float_fields_are_preserved_as_integers():
    values = dict(VALID, year=2018.0, num_reviews=250.0, num_pages=320.0)
    validated = validate_prediction_input(values)
    assert validated["year"] == 2018
    assert validated["num_reviews"] == 250
    assert validated["num_pages"] == 320
    assert all(
        isinstance(validated[column], int)
        for column in ["year", "num_reviews", "num_pages"]
    )
