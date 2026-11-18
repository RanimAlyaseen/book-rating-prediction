# Reproduced experiment results

These results come from the portfolio pipeline, not from the original course report. Model selection used mean macro F1 across five group-aware training folds. The held-out test set was not used for hyperparameter selection.

## Data and split

- Clean observations: 348
- Positive class: 241 (69.3%)
- Training observations: 278
- Held-out observations: 70
- Title-group overlap: 0

## Results

### Training-only cross-validation

| Model | Accuracy | Precision | Recall | Positive F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Most Frequent | 0.691 ± 0.007 | 0.691 ± 0.007 | 1.000 ± 0.000 | 0.817 ± 0.005 | 0.409 ± 0.002 | 0.500 ± 0.000 |
| Logistic Regression | 0.634 ± 0.076 | 0.792 ± 0.071 | 0.635 ± 0.066 | 0.705 ± 0.066 | 0.610 ± 0.079 | 0.652 ± 0.055 |
| Random Forest | 0.615 ± 0.039 | 0.742 ± 0.019 | 0.677 ± 0.060 | 0.707 ± 0.039 | 0.570 ± 0.034 | 0.623 ± 0.050 |

### Held-out test

| Model | Accuracy | Precision | Recall | Positive F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Most Frequent | 0.700 | 0.700 | 1.000 | 0.824 | 0.412 | 0.500 |
| Logistic Regression | 0.457 | 0.667 | 0.449 | 0.537 | 0.441 | 0.433 |
| Random Forest | 0.529 | 0.700 | 0.571 | 0.629 | 0.491 | 0.476 |

The training-only selection rule chose **Logistic Regression**. On the small held-out set, its 95% bootstrap interval was 0.326–0.554 for macro F1 and 0.343–0.571 for accuracy. These intervals describe resampling variability within one held-out set, not performance on a new population.

The selected Logistic Regression did not beat the majority-class dummy on held-out accuracy, positive-class F1, or ROC-AUC. It improved held-out macro F1 only from 0.412 to 0.441. Random Forest produced the highest held-out macro F1 (0.491), but the pipeline did not switch models after viewing test results. The experiment therefore does not establish a reliable predictive advantage over the simple baseline.

## Limitations

The dataset contains only 348 cleaned observations and one repeated-title group. The source population is a curated bestseller collection rather than a representative sample of all books. The high-rating label is derived from a narrow rating range, category sizes are small, and review counts are highly skewed. Results should be treated as an academic demonstration, not as evidence that the model generalizes to a commercial catalog.
