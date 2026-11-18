from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_handles_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("BOOK_RATING_MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = Path(__file__).resolve().parents[1] / "app.py"
    test = AppTest.from_file(str(app), default_timeout=15).run()
    assert not test.exception
    assert any("compatible trained model" in item.value for item in test.error)


def test_streamlit_labels_synthetic_model_and_uncalibrated_score(
    monkeypatch, demo_run
):
    monkeypatch.setenv("BOOK_RATING_MODEL_PATH", str(demo_run["model"]))
    app = Path(__file__).resolve().parents[1] / "app.py"
    test = AppTest.from_file(str(app), default_timeout=15).run()
    assert not test.exception
    assert any("Synthetic demonstration model" in item.value for item in test.warning)
    assert any("held-out accuracy was 45.7%" in item.value for item in test.warning)

    test.button[0].click()
    test.run()
    assert not test.exception
    assert test.metric[0].label == "Uncalibrated model score for the high-rating class"
    assert any("not a verified likelihood" in item.value for item in test.caption)
