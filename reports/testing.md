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

Development-environment result: **14 passed in 28.18 seconds** (pytest-reported duration).

A second virtual environment was then created from scratch. Installing `requirements-dev.txt`, installing the project in editable mode, and running `python -m pip check` all succeeded. The suite result in that clean environment was **14 passed in 31.03 seconds**.

The final pre-commit rerun was **14 passed in 22.43 seconds**.

The suite verifies schema failures, invalid values, cleaning and target derivation, group-aware split isolation, training, input feature order, serialization, inference in a fresh Python process, deterministic prediction under reordered input mappings, command-line training, and the application's missing-model state.

## Full private-data run

The documented training command completed successfully against the fingerprinted private workbook. It selected Logistic Regression using training-only cross-validated macro F1, wrote all reports and figures, and created a complete local ignored model bundle.

The regenerated held-out metrics exactly matched the checked JSON output: accuracy 0.457143, positive F1 0.536585, macro F1 0.440706, and ROC-AUC 0.433431.

## Artifact and application checks

- The locally generated Joblib bundle loaded successfully in a separate Python process.
- Prediction remained stable when the input mapping order changed.
- Streamlit's `AppTest` completed with no uncaught exceptions using the actual local bundle.
- Submitting the default form displayed the `High-rated` class, a 93.5% estimated probability, and the academic-use disclaimer.
- Running without an artifact displayed recovery instructions instead of raising an exception.

The model artifact is intentionally ignored by Git and excluded from the review archive. It can be recreated only from an authorized copy of the source data.
