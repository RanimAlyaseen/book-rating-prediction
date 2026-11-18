"""Streamlit demonstration for the trained book high-rating classifier."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from book_rating.artifacts import load_bundle, predict_one
from book_rating.validation import InputValidationError


st.set_page_config(page_title="Book High-Rating Prediction", layout="centered")
st.title("Book high-rating prediction")
st.write(
    "Estimate whether a book's rating belongs to the high-rating class "
    "(rating at least 4.6) from six metadata fields."
)

default_model = Path(__file__).resolve().parent / "models" / "book_rating_pipeline.joblib"
model_path = Path(os.environ.get("BOOK_RATING_MODEL_PATH", default_model))


@st.cache_resource
def cached_bundle(path: str):
    return load_bundle(path)


try:
    bundle = cached_bundle(str(model_path))
except (FileNotFoundError, EOFError, ValueError, ImportError, AttributeError) as exc:
    st.error("A compatible trained model is not available.")
    st.code(
        "python -m book_rating.train --data <authorized-dataset.xlsx>",
        language="powershell",
    )
    st.caption(f"Model loading detail: {exc}")
    st.stop()

genres = bundle["allowed_categories"]["genre"]
sub_genres = bundle["allowed_categories"]["sub_genre"]

with st.form("prediction_form"):
    genre = st.selectbox("Genre", genres)
    sub_genre = st.selectbox("Sub-genre", sub_genres)
    year = st.number_input("Publication year", min_value=1400, max_value=2100, value=2018, step=1)
    price = st.number_input("Price", min_value=0.01, max_value=1000.0, value=12.99, step=0.5)
    num_reviews = st.number_input(
        "Number of reviews", min_value=0, max_value=10_000_000, value=250, step=10
    )
    num_pages = st.number_input(
        "Number of pages", min_value=1, max_value=10_000, value=320, step=10
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        prediction, probability = predict_one(
            bundle,
            {
                "year": year,
                "price": price,
                "num_reviews": num_reviews,
                "num_pages": num_pages,
                "genre": genre,
                "sub_genre": sub_genre,
            },
        )
    except InputValidationError as exc:
        st.error(str(exc))
    else:
        label = "High-rated (rating ≥ 4.6)" if prediction == 1 else "Not high-rated (rating < 4.6)"
        st.subheader(label)
        st.metric("Estimated probability of rating ≥ 4.6", f"{probability:.1%}")

st.caption(
    "Academic demonstration only. The model uses a small curated dataset and does "
    "not provide personalized recommendations or quality guarantees."
)
