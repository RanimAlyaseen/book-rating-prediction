from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
from sklearn.linear_model import LogisticRegression

from book_rating.artifacts import load_bundle, predict_one, save_bundle
from book_rating.constants import FEATURE_COLUMNS, TARGET_THRESHOLD
from book_rating.data import prepare_dataset, split_features_target
from book_rating.modeling import (
    build_pipeline,
    evaluate_holdout,
    make_holdout_split,
    run_model_search,
    select_model,
)


def build_bundle(synthetic_frame):
    prepared = prepare_dataset(synthetic_frame)
    X, y, groups = split_features_target(prepared)
    train_index, test_index = make_holdout_split(X, y, groups)
    assert not (set(groups.iloc[train_index]) & set(groups.iloc[test_index]))
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, random_state=42))
    pipeline.fit(X.iloc[train_index], y.iloc[train_index])
    return {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_threshold": TARGET_THRESHOLD,
        "allowed_categories": {
            "genre": sorted(prepared["genre"].unique()),
            "sub_genre": sorted(prepared["sub_genre"].unique()),
        },
        "versions": {"scikit-learn": "test"},
        "source_sha256": "synthetic-test-data",
        "data_origin": "synthetic_demo",
    }


def test_training_and_prediction_are_stable(synthetic_frame):
    bundle = build_bundle(synthetic_frame)
    values = {
        "year": 2018,
        "price": 12.99,
        "num_reviews": 250,
        "num_pages": 320,
        "genre": "Fiction",
        "sub_genre": "Adventure",
    }
    first = predict_one(bundle, values)
    second = predict_one(bundle, dict(reversed(list(values.items()))))
    assert first == second
    assert first[0] in {0, 1}
    assert 0 <= first[1] <= 1


def test_serialized_bundle_loads_in_new_process(synthetic_frame, tmp_path):
    artifact = tmp_path / "model.joblib"
    save_bundle(build_bundle(synthetic_frame), artifact)
    loaded = load_bundle(artifact)
    assert hasattr(loaded["pipeline"], "predict_proba")

    code = (
        "import json,sys; "
        "from book_rating.artifacts import load_bundle,predict_one; "
        "b=load_bundle(sys.argv[1]); "
        "v={'year':2018,'price':12.99,'num_reviews':250,'num_pages':320,"
        "'genre':'Fiction','sub_genre':'Adventure'}; "
        "print(json.dumps(predict_one(b,v)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    prediction, probability = json.loads(result.stdout)
    assert prediction in {0, 1}
    assert 0 <= probability <= 1


def test_selection_rule_does_not_depend_on_holdout_labels_or_scores(synthetic_frame):
    prepared = prepare_dataset(synthetic_frame)
    X, y, groups = split_features_target(prepared)
    train_index, test_index = make_holdout_split(X, y, groups)
    training_results, estimators = run_model_search(
        X.iloc[train_index],
        y.iloc[train_index],
        groups.iloc[train_index],
        quick=True,
    )

    selected_name, _ = select_model(training_results, estimators)
    evaluate_holdout(estimators[selected_name], X.iloc[test_index], y.iloc[test_index])
    evaluate_holdout(
        estimators[selected_name], X.iloc[test_index], 1 - y.iloc[test_index]
    )

    fabricated_scores = deepcopy(training_results)
    for name, result in fabricated_scores.items():
        result["test"] = {"macro_f1": 1.0 if name != selected_name else 0.0}
    selected_after_fabricated_holdout, _ = select_model(
        fabricated_scores, estimators
    )
    assert selected_after_fabricated_holdout == selected_name
