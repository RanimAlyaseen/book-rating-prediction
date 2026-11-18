from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_handles_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("BOOK_RATING_MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = Path(__file__).resolve().parents[1] / "app.py"
    test = AppTest.from_file(str(app), default_timeout=15).run()
    assert not test.exception
    assert any("compatible trained model" in item.value for item in test.error)
