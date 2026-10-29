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
