# Model artifacts

Run the documented training command to create `book_rating_pipeline.joblib` in this directory. Generated Joblib and pickle files are ignored because they are version-sensitive binary artifacts and should only be loaded when they were produced locally from trusted code and data.

The repository records the model-selection results, exact package versions, source-data fingerprint, and figures without publishing the private dataset or generated model binary.

`python -m book_rating.demo` creates a separate `book_rating_demo_pipeline.joblib` from fabricated public rows. The artifact records `data_origin=synthetic_demo`, is ignored by Git, and is identified explicitly by the Streamlit interface.
