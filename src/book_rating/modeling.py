"""Leakage-controlled model construction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


@dataclass(frozen=True)
class CandidateSpec:
    """One model family and its training-only search space."""

    estimator: object
    param_grid: dict[str, list[object]]


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing fitted only inside each training fold."""

    regular_numeric = ("year", "price", "num_pages")
    regular_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    reviews_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", regular_pipeline, regular_numeric),
            ("reviews", reviews_pipeline, ("num_reviews",)),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(estimator: object) -> Pipeline:
    """Wrap preprocessing and a classifier in one serializable pipeline."""

    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def candidate_specs(quick: bool = False) -> dict[str, CandidateSpec]:
    """Return the deliberate, compact model comparison."""

    forest_grid: dict[str, list[object]]
    logistic_grid: dict[str, list[object]]
    if quick:
        logistic_grid = {
            "model__C": [1.0],
            "model__class_weight": [None],
        }
        forest_grid = {
            "model__n_estimators": [40],
            "model__max_depth": [None],
            "model__min_samples_leaf": [1],
            "model__class_weight": ["balanced_subsample"],
        }
    else:
        logistic_grid = {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        }
        forest_grid = {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 4],
            "model__class_weight": ["balanced_subsample"],
        }

    return {
        "dummy_most_frequent": CandidateSpec(
            DummyClassifier(strategy="most_frequent"), {}
        ),
        "logistic_regression": CandidateSpec(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            logistic_grid,
        ),
        "random_forest": CandidateSpec(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            forest_grid,
        ),
    }


def scoring() -> dict[str, object]:
    """Metrics used consistently for model selection and reporting."""

    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "macro_f1": "f1_macro",
        "roc_auc": "roc_auc",
    }


def make_holdout_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Create a fixed 80/20 group-aware holdout without using model results."""

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_index, test_index = next(splitter.split(X, y, groups))
    return train_index, test_index


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate held-out classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    predictions: np.ndarray,
    iterations: int = 2000,
) -> dict[str, list[float]]:
    """Estimate test-set variability with a fixed nonparametric bootstrap."""

    rng = np.random.default_rng(RANDOM_STATE)
    samples: dict[str, list[float]] = {"accuracy": [], "f1": [], "macro_f1": []}
    for _ in range(iterations):
        indexes = rng.integers(0, len(y_true), size=len(y_true))
        truth = y_true[indexes]
        predicted = predictions[indexes]
        samples["accuracy"].append(float(accuracy_score(truth, predicted)))
        samples["f1"].append(float(f1_score(truth, predicted, zero_division=0)))
        samples["macro_f1"].append(
            float(f1_score(truth, predicted, average="macro", zero_division=0))
        )
    return {
        metric: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for metric, values in samples.items()
    }


def run_model_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    quick: bool = False,
) -> tuple[dict[str, dict[str, object]], dict[str, Pipeline]]:
    """Tune candidate pipelines using training folds only."""

    cv = list(
        StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1
        ).split(X_train, y_train, groups_train)
    )
    results: dict[str, dict[str, object]] = {}
    estimators: dict[str, Pipeline] = {}

    for name, spec in candidate_specs(quick=quick).items():
        search = GridSearchCV(
            build_pipeline(spec.estimator),
            param_grid=spec.param_grid,
            scoring=scoring(),
            refit="macro_f1",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_index = search.best_index_
        estimator = search.best_estimator_
        cv_metrics = {
            metric: {
                "mean": float(search.cv_results_[f"mean_test_{metric}"][best_index]),
                "std": float(search.cv_results_[f"std_test_{metric}"][best_index]),
            }
            for metric in scoring()
        }
        results[name] = {
            "best_params": search.best_params_,
            "cv": cv_metrics,
        }
        estimators[name] = estimator

    return results, estimators


def select_model(
    training_results: dict[str, dict[str, object]],
    estimators: dict[str, Pipeline],
) -> tuple[str, Pipeline]:
    """Freeze the winner from training-fold macro F1 without holdout input."""

    selected_name = max(
        training_results,
        key=lambda name: training_results[name]["cv"]["macro_f1"]["mean"],
    )
    return selected_name, estimators[selected_name]


def evaluate_holdout(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Evaluate one already-fitted pipeline on the held-out set."""

    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    evaluation = {
        "test": evaluate_predictions(y_test, predictions, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return evaluation, predictions, probabilities


def refit_for_deployment(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Refit selected hyperparameters on all available verified rows."""

    return clone(estimator).fit(X, y)
