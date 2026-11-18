"""Shared project constants."""

TARGET_THRESHOLD = 4.6
TARGET_COLUMN = "high_rating"

NUMERIC_FEATURES = ("year", "price", "num_reviews", "num_pages")
CATEGORICAL_FEATURES = ("genre", "sub_genre")
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

REQUIRED_COLUMNS = (
    "name",
    "author",
    "rating",
    *FEATURE_COLUMNS,
)

RANDOM_STATE = 42
