"""Command-line training and evaluation entry point."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import save_bundle
from .constants import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_THRESHOLD
from .data import file_sha256, load_dataset, prepare_dataset, split_features_target
from .modeling import (
    bootstrap_intervals,
    make_holdout_split,
    refit_for_deployment,
    run_model_search,
)


def _round(value: float) -> float:
    return round(float(value), 6)


def _plot_class_distribution(prepared: pd.DataFrame, output: Path) -> None:
    counts = prepared[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Not high-rated", "High-rated"]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    bars = axis.bar(labels, counts.reindex([0, 1]).values, color=["#64748B", "#2563EB"])
    axis.set_title("Class distribution after cleaning")
    axis.set_ylabel("Books")
    axis.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts.reindex([0, 1]).values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, count + 3, str(int(count)), ha="center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_feature_distributions(prepared: pd.DataFrame, output: Path) -> None:
    columns = ["year", "price", "num_reviews", "num_pages"]
    titles = ["Publication year", "Price", "Review count (log scale)", "Page count"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    for axis, column, title in zip(axes.flat, columns, titles, strict=True):
        values = prepared[column]
        if column == "num_reviews":
            values = np.log1p(values)
        axis.hist(values, bins=20, color="#2563EB", alpha=0.85, edgecolor="white")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_model_comparison(results: dict[str, dict[str, object]], output: Path) -> None:
    names = list(results)
    display = [name.replace("_", " ").title() for name in names]
    metrics = ["accuracy", "f1", "macro_f1", "roc_auc"]
    labels = ["Accuracy", "Positive F1", "Macro F1", "ROC-AUC"]
    x = np.arange(len(names))
    width = 0.18
    fig, axis = plt.subplots(figsize=(10, 5.2))
    colors = ["#64748B", "#2563EB", "#0F766E", "#7C3AED"]
    for index, (metric, label, color) in enumerate(zip(metrics, labels, colors, strict=True)):
        values = [results[name]["test"][metric] for name in names]
        axis.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    axis.set_xticks(x, display)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out score")
    axis.set_title("Held-out model comparison")
    axis.legend(ncols=2, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(matrix: list[list[int]], model_name: str, output: Path) -> None:
    values = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks([0, 1], ["Not high-rated", "High-rated"])
    axis.set_yticks([0, 1], ["Not high-rated", "High-rated"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"Held-out confusion matrix\n{model_name.replace('_', ' ').title()}")
    midpoint = values.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > midpoint else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _error_analysis(
    prepared: pd.DataFrame, test_index: np.ndarray, predictions: np.ndarray
) -> dict[str, list[dict[str, object]]]:
    detail = prepared.iloc[test_index][["genre", "sub_genre", TARGET_COLUMN]].copy()
    detail["prediction"] = predictions
    detail["error"] = detail[TARGET_COLUMN] != detail["prediction"]
    detail["false_positive"] = (detail[TARGET_COLUMN] == 0) & (detail["prediction"] == 1)
    detail["false_negative"] = (detail[TARGET_COLUMN] == 1) & (detail["prediction"] == 0)

    output: dict[str, list[dict[str, object]]] = {}
    for column in ["genre", "sub_genre"]:
        grouped = (
            detail.groupby(column, observed=True)
            .agg(
                observations=("error", "size"),
                errors=("error", "sum"),
                error_rate=("error", "mean"),
                false_positives=("false_positive", "sum"),
                false_negatives=("false_negative", "sum"),
            )
            .reset_index()
            .sort_values(["errors", "observations"], ascending=False)
        )
        grouped["error_rate"] = grouped["error_rate"].round(6)
        output[column] = grouped.to_dict("records")
    return output


def _write_experiment_report(metrics: dict[str, object], output: Path) -> None:
    cv_rows = []
    test_rows = []
    for name, result in metrics["models"].items():
        cv = result["cv"]
        test = result["test"]
        display_name = name.replace("_", " ").title()
        cv_rows.append(
            "| {name} | {accuracy:.3f} ± {accuracy_std:.3f} | "
            "{precision:.3f} ± {precision_std:.3f} | {recall:.3f} ± {recall_std:.3f} | "
            "{f1:.3f} ± {f1_std:.3f} | {macro:.3f} ± {macro_std:.3f} | "
            "{auc:.3f} ± {auc_std:.3f} |".format(
                name=display_name,
                accuracy=cv["accuracy"]["mean"],
                accuracy_std=cv["accuracy"]["std"],
                precision=cv["precision"]["mean"],
                precision_std=cv["precision"]["std"],
                recall=cv["recall"]["mean"],
                recall_std=cv["recall"]["std"],
                f1=cv["f1"]["mean"],
                f1_std=cv["f1"]["std"],
                macro=cv["macro_f1"]["mean"],
                macro_std=cv["macro_f1"]["std"],
                auc=cv["roc_auc"]["mean"],
                auc_std=cv["roc_auc"]["std"],
            )
        )
        test_rows.append(
            "| {name} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | "
            "{f1:.3f} | {macro:.3f} | {auc:.3f} |".format(
                name=name.replace("_", " ").title(),
                **test,
                macro=test["macro_f1"],
                auc=test["roc_auc"],
            )
        )
    selected = metrics["selected_model"]
    intervals = metrics["selected_model_test_intervals_95"]
    text = f"""# Reproduced experiment results

These results come from the portfolio pipeline, not from the original course report. Model selection used mean macro F1 across five group-aware training folds. The held-out test set was not used for hyperparameter selection.

## Data and split

- Clean observations: {metrics['data']['clean_rows']}
- Positive class: {metrics['data']['positive_count']} ({metrics['data']['positive_rate']:.1%})
- Training observations: {metrics['split']['train_rows']}
- Held-out observations: {metrics['split']['test_rows']}
- Title-group overlap: {metrics['split']['title_group_overlap']}

## Results

### Training-only cross-validation

| Model | Accuracy | Precision | Recall | Positive F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cv_rows)}

### Held-out test

| Model | Accuracy | Precision | Recall | Positive F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(test_rows)}

The training-only selection rule chose **{selected.replace('_', ' ').title()}**. On the small held-out set, its 95% bootstrap interval was {intervals['macro_f1'][0]:.3f}–{intervals['macro_f1'][1]:.3f} for macro F1 and {intervals['accuracy'][0]:.3f}–{intervals['accuracy'][1]:.3f} for accuracy. These intervals describe resampling variability within one held-out set, not performance on a new population.

The selected Logistic Regression did not beat the majority-class dummy on held-out accuracy, positive-class F1, or ROC-AUC. It improved held-out macro F1 only from 0.412 to 0.441. Random Forest produced the highest held-out macro F1 (0.491), but the pipeline did not switch models after viewing test results. The experiment therefore does not establish a reliable predictive advantage over the simple baseline.

## Limitations

The dataset contains only 348 cleaned observations and one repeated-title group. The source population is a curated bestseller collection rather than a representative sample of all books. The high-rating label is derived from a narrow rating range, category sizes are small, and review counts are highly skewed. Results should be treated as an academic demonstration, not as evidence that the model generalizes to a commercial catalog.
"""
    output.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.data)
    raw = load_dataset(source)
    prepared = prepare_dataset(raw)
    X, y, groups = split_features_target(prepared)
    train_index, test_index = make_holdout_split(X, y, groups)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    groups_train = groups.iloc[train_index]
    model_results, estimators = run_model_search(
        X_train, y_train, groups_train, X_test, y_test, quick=args.quick
    )
    selected_name = max(
        model_results,
        key=lambda name: model_results[name]["cv"]["macro_f1"]["mean"],
    )
    selected_evaluation_model = estimators[selected_name]
    selected_predictions = selected_evaluation_model.predict(X_test)
    intervals = bootstrap_intervals(y_test.to_numpy(), selected_predictions)
    deployment_model = refit_for_deployment(selected_evaluation_model, X, y)

    allowed_categories = {
        column: sorted(prepared[column].astype(str).unique().tolist())
        for column in ["genre", "sub_genre"]
    }
    versions = {
        package: version(package)
        for package in ["numpy", "pandas", "scikit-learn", "joblib"]
    }
    source_digest = file_sha256(source)
    bundle = {
        "pipeline": deployment_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_threshold": TARGET_THRESHOLD,
        "allowed_categories": allowed_categories,
        "versions": versions,
        "source_sha256": source_digest,
        "selected_model": selected_name,
    }
    save_bundle(bundle, args.model_out)

    overlap = set(groups.iloc[train_index]) & set(groups.iloc[test_index])
    metrics: dict[str, object] = {
        "data": {
            "source_filename": source.name,
            "source_sha256": source_digest,
            "raw_rows": int(len(raw)),
            "clean_rows": int(len(prepared)),
            "removed_nonpositive_price_rows": int((raw["price"] <= 0).sum()),
            "positive_count": int(y.sum()),
            "negative_count": int((1 - y).sum()),
            "positive_rate": _round(y.mean()),
            "exact_duplicate_rows": int(raw.duplicated().sum()),
            "repeated_title_groups": int(
                groups[groups.duplicated(keep=False)].nunique()
            ),
        },
        "split": {
            "method": "first fold of shuffled StratifiedGroupKFold(n_splits=5)",
            "random_state": 42,
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "train_positive_rate": _round(y_train.mean()),
            "test_positive_rate": _round(y_test.mean()),
            "title_group_overlap": len(overlap),
        },
        "selection_metric": "mean cross-validated macro_f1 on training data",
        "models": model_results,
        "selected_model": selected_name,
        "selected_model_test_intervals_95": intervals,
        "error_analysis": _error_analysis(prepared, test_index, selected_predictions),
        "versions": versions,
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    _plot_class_distribution(prepared, figures / "class_distribution.png")
    _plot_feature_distributions(prepared, figures / "feature_distributions.png")
    _plot_model_comparison(model_results, figures / "model_comparison.png")
    _plot_confusion_matrix(
        model_results[selected_name]["confusion_matrix"],
        selected_name,
        figures / "confusion_matrix.png",
    )

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_experiment_report(metrics, report_path)
    return metrics


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data", required=True, help="Authorized CSV or Excel dataset")
    command.add_argument(
        "--model-out", default="models/book_rating_pipeline.joblib"
    )
    command.add_argument("--metrics-out", default="reports/metrics.json")
    command.add_argument("--figures-dir", default="figures")
    command.add_argument(
        "--report-out", default="reports/experiment-results.md"
    )
    command.add_argument(
        "--quick", action="store_true", help="Use reduced grids for automated tests"
    )
    return command


def main() -> None:
    args = parser().parse_args()
    metrics = run(args)
    print(json.dumps({"selected_model": metrics["selected_model"], "test": metrics["models"][metrics["selected_model"]]["test"]}, indent=2))


if __name__ == "__main__":
    main()
