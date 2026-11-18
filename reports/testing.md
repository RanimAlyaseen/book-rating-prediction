# Verification record

Verification was performed in an isolated Python virtual environment on Windows. The original university directory was read only; all generated outputs were written to the portfolio repository.

## Environment

- Python 3.12.14
- Joblib 1.6.0
- Matplotlib 3.11.2
- NumPy 2.5.5
- OpenPyXL 3.1.5
- pandas 3.1.1
- pytest 9.2.0
- scikit-learn 1.9.1
- Streamlit 1.67.0

`python -m pip check` reported no broken requirements.

## Automated tests

Command:

```powershell
python -m pytest
```

Final-patch development result: **22 passed in 33.30 seconds** (pytest-reported duration).

A new virtual environment was created from scratch after the patch. Installing `requirements-dev.txt`, installing the project in editable mode, and running `python -m pip check` all succeeded. The final suite result in that clean environment was **22 passed in 35.04 seconds**.

The suite verifies schema failures, invalid values, cleaning and target derivation, group-aware split isolation, rejection of fractional integer fields, training-only model selection, independence from fabricated held-out labels and scores, serialization, inference in a fresh Python process, deterministic prediction under reordered input mappings, command-line training, isolated synthetic demonstration outputs, rejection of the bundled fixture by the research command, and the application's real, synthetic, and missing-model states.

## Full private-data run

The documented training command completed successfully against the fingerprinted private workbook. Candidate search and selection used training folds only. Logistic Regression was frozen as the winner before the selected model, dummy baseline, and descriptive Random Forest comparison were evaluated on the holdout.

The regenerated held-out metrics exactly matched the checked JSON output: accuracy 0.457143, positive F1 0.536585, macro F1 0.440706, and ROC-AUC 0.433431.

Comparison with the preceding commit confirmed exact equality for every best parameter, cross-validation score, held-out score, confusion matrix, and bootstrap interval. All four research figures were unchanged. Only evaluation-sequence/provenance metadata and explanatory report text changed.

## Public synthetic demonstration

Commands:

```powershell
python -m book_rating.demo
$env:BOOK_RATING_MODEL_PATH = "models/book_rating_demo_pipeline.joblib"
streamlit run app.py
```

The command completed using only `data/synthetic_books.csv`. It produced an ignored 5,925-byte demonstration model plus `demo-output/demo-metrics.json`, a synthetic-only report, and four synthetic-only figures. No tracked research output changed.

The generated bundle recorded `data_origin=synthetic_demo`. Streamlit loaded it with no exceptions, displayed the synthetic-data warning and weak real-experiment disclosure, and returned a default uncalibrated model score of 73.6%. The interface explicitly stated that the score is not a verified likelihood.

The exact demonstration command was also rerun with the clean interpreter. A headless Streamlit server started on `127.0.0.1:8765`, and `/_stcore/health` returned HTTP 200 with body `ok` before the test server was stopped.

## Artifact and application checks

- The locally generated Joblib bundle loaded successfully in a separate Python process.
- Prediction remained stable when the input mapping order changed.
- Streamlit's `AppTest` completed with no uncaught exceptions for both real and synthetic bundles.
- The real artifact was identified as a private-data experimental model.
- The synthetic artifact was identified as a fabricated demonstration model.
- Scores were labeled as uncalibrated and accompanied by a non-likelihood warning.
- Running without an artifact displayed the safe synthetic-demo command instead of raising an exception.

Both model artifacts are intentionally ignored by Git and excluded from the review archive. The private model can be recreated only from an authorized copy of the source data; the synthetic model can be regenerated from the included fabricated fixture.
