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

models_dir = Path(__file__).resolve().parent / "models"
real_model = models_dir / "book_rating_pipeline.joblib"
demo_model = models_dir / "book_rating_demo_pipeline.joblib"
explicit_model = os.environ.get("BOOK_RATING_MODEL_PATH")
model_path = Path(explicit_model) if explicit_model else (
    real_model if real_model.is_file() else demo_model
)


@st.cache_resource
def cached_bundle(path: str):
    return load_bundle(path)


try:
    bundle = cached_bundle(str(model_path))
except (FileNotFoundError, EOFError, ValueError, ImportError, AttributeError) as exc:
    st.error("A compatible trained model is not available.")
    st.code(
        "python -m book_rating.demo\nstreamlit run app.py",
        language="powershell",
    )
    st.write(
        "The command above creates a local model only from the included fabricated rows. "
        "Authorized private-data training remains documented separately."
    )
    st.caption(f"Model loading detail: {exc}")
    st.stop()

if bundle["data_origin"] == "synthetic_demo":
    st.warning(
        "Synthetic demonstration model loaded. Its rows and relationships are fabricated; "
        "predictions and metrics do not establish performance on real books."
    )
else:
    st.info(
        "Private-data experimental model loaded. The source workbook and model binary are "
        "not published in this repository."
    )

st.warning(
    "Published private-data evaluation was weak: held-out accuracy was 45.7% and "
    "ROC-AUC was 43.3%. The selected model did not reliably outperform the majority baseline."
)

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
        st.metric("Uncalibrated model score for the high-rating class", f"{probability:.1%}")
        st.caption(
            "This score is not calibrated and is not a verified likelihood that a book "
            "will receive a rating of at least 4.6."
        )

st.caption(
    "Academic demonstration only. The model uses a small curated dataset and does "
    "not provide personalized recommendations or quality guarantees."
)
