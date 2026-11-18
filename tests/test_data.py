from __future__ import annotations

import pytest

from book_rating.data import DataValidationError, prepare_dataset, split_features_target


def test_preparation_removes_nonpositive_price_and_derives_target(synthetic_frame):
    synthetic_frame.loc[0, "price"] = 0
    prepared = prepare_dataset(synthetic_frame)
    X, y, groups = split_features_target(prepared)

    assert len(prepared) == 119
    assert list(X.columns) == [
        "year",
        "price",
        "num_reviews",
        "num_pages",
        "genre",
        "sub_genre",
    ]
    assert set(y.unique()) == {0, 1}
    assert groups.loc[1] == groups.loc[2]


def test_missing_required_column_is_rejected(synthetic_frame):
    with pytest.raises(DataValidationError, match="Missing required columns: num_pages"):
        prepare_dataset(synthetic_frame.drop(columns="num_pages"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("rating", 5.5, "rating values must be between 0 and 5"),
        ("num_reviews", -1, "num_reviews values must be nonnegative"),
        ("num_pages", 0, "num_pages values must be positive"),
        ("genre", "", "genre values must not be blank"),
    ],
)
def test_invalid_domains_are_rejected(synthetic_frame, column, value, message):
    synthetic_frame.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message):
        prepare_dataset(synthetic_frame)
