# Historical report results

The following values appear in the original final report. They are retained for academic context only. The supplied source code does not reproduce these experiments, and the exact split assignments, preprocessing variants, random seeds, tuning grids, and SHAP computation are unavailable.

## Reported non-ensemble results

| Model | CV accuracy | CV F1 | Test accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.558 | 0.627 | 0.529 | 0.602 | 0.616 |
| SVM with RBF kernel | 0.540 | 0.592 | 0.543 | 0.590 | 0.580 |
| Naive Bayes | 0.335 | 0.153 | 0.414 | 0.255 | 0.556 |

## Reported ensemble results

| Model | CV accuracy | CV F1 | Test accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.655 | 0.778 | 0.629 | 0.764 | 0.604 |
| XGBoost | 0.644 | 0.758 | 0.614 | 0.738 | 0.604 |
| Gradient Boosting | 0.633 | 0.758 | 0.600 | 0.736 | 0.554 |

Other Stage 3 documents report materially different Logistic Regression values, including cross-validated F1 of 0.794 and held-out F1 of 0.782. This inconsistency is another reason not to present the historical numbers as verified benchmarks.

The portfolio experiment uses a new fixed group-aware split, training-only model selection, exact dependency versions, and machine-readable outputs. Its lower results are not adjusted to match the historical report.
