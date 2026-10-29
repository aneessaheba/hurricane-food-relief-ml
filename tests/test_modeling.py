import numpy as np
import pandas as pd
from src.modeling import build_preprocessor

def test_empty_feature_group_stays_empty():
    # An explicitly empty group must not fall back to the config defaults
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "state": ["FL", "LA", "FL"]})
    pre = build_preprocessor(continuous=["a"], categorical=["state"], binary=[])
    out = pre.fit_transform(df)
    # 1 scaled column + 1 one-hot column (drop-first of 2 states)
    assert out.shape == (3, 2)

def test_continuous_nan_is_imputed():
    # SMOTE rejects NaN, so the preprocessor must median-impute continuous columns
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "state": ["FL", "LA", "FL"]})
    pre = build_preprocessor(continuous=["a"], categorical=["state"], binary=[])
    out = pre.fit_transform(df)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# Model package save / load
# ---------------------------------------------------------------------------
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src import modeling
from src.modeling import load_model_package, save_model_package


@pytest.fixture
def fitted():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    le = LabelEncoder().fit(["Low", "Medium", "High", "Severe"])
    y = le.transform(["Low", "Severe", "Low", "Severe"])
    return LogisticRegression().fit(X, y), le, ["a", "b"]


def test_save_and_load_round_trip_with_metadata(tmp_path, fitted):
    model, le, features = fitted
    path = save_model_package(tmp_path / "m.pkl", model, le, features,
                              train_storms=["Matthew"], calibration_storms=["Laura"])
    pkg = load_model_package(path, expected_features=features, strict=True)
    meta = pkg["metadata"]
    assert pkg["features"] == features and pkg["calibrator"] is None
    assert meta["classes"] == ["High", "Low", "Medium", "Severe"]
    assert meta["train_storms"] == ["Matthew"] and meta["calibration_storms"] == ["Laura"]
    assert meta["calibrated"] is False
    assert meta["versions"]["scikit-learn"] is not None


def test_load_rejects_feature_mismatch(tmp_path, fitted):
    model, le, features = fitted
    path = save_model_package(tmp_path / "m.pkl", model, le, features)
    with pytest.raises(ValueError, match="different features"):
        load_model_package(path, expected_features=["a"])


def test_load_rejects_missing_keys(tmp_path):
    joblib.dump({"pipe": object()}, tmp_path / "bad.pkl")
    with pytest.raises(ValueError, match="missing"):
        load_model_package(tmp_path / "bad.pkl")


def test_load_accepts_legacy_package_with_warning(tmp_path, fitted, caplog):
    model, le, features = fitted
    joblib.dump({"pipe": model, "label_encoder": le, "features": features}, tmp_path / "old.pkl")
    with caplog.at_level("WARNING"):
        pkg = load_model_package(tmp_path / "old.pkl")
    X = pd.DataFrame({"a": [0.5, 2.5], "b": [1.0, 0.0]})
    assert (pkg["pipe"].predict(X) == model.predict(X)).all()
    assert "no metadata" in caplog.text


def test_version_drift_warns_or_raises(tmp_path, fitted, monkeypatch, caplog):
    model, le, features = fitted
    path = save_model_package(tmp_path / "m.pkl", model, le, features)
    monkeypatch.setattr(modeling, "_package_versions",
                        lambda: {"scikit-learn": "0.0.1", "xgboost": None,
                                 "imbalanced-learn": None, "numpy": None, "pandas": None})
    with caplog.at_level("WARNING"):
        load_model_package(path)
    assert "scikit-learn saved with" in caplog.text
    with pytest.raises(ValueError, match="library versions differ"):
        load_model_package(path, strict=True)
