"""
Evaluation, SHAP, and equity audit utilities.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# SHAP for multi-class trees
# -----------------------------------------------------------------------------
def shap_multiclass(model, X: pd.DataFrame, class_names=None):
    """
    Compute SHAP values for a tree-based multi-class classifier. Returns
    the raw SHAP object with a `.values` array of shape
    (n_samples, n_features, n_classes). The Severe class is indexed as the
    last class by default (the caller should pass an explicit index if the
    label encoder reordered classes).
    """
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    return explainer, sv


def severe_class_idx(class_names, target="Severe") -> int:
    for i, c in enumerate(class_names):
        if str(c) == target:
            return i
    return len(class_names) - 1


# -----------------------------------------------------------------------------
# Equity audit via Fairlearn
# -----------------------------------------------------------------------------
def equity_audit(
    y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series,
    severe_label="Severe",
) -> pd.DataFrame:
    """
    Stratified accuracy + recall-for-Severe by SVI quartile, plus Fairlearn
    demographic_parity_ratio and equalized_odds_difference.
    """
    from fairlearn.metrics import (
        MetricFrame, demographic_parity_ratio, equalized_odds_difference,
    )
    from sklearn.metrics import accuracy_score, recall_score, f1_score

    def _recall_severe(y_t, y_p):
        return recall_score(y_t == severe_label, y_p == severe_label, zero_division=0)

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "f1_weighted": lambda a, b: f1_score(a, b, average="weighted", zero_division=0),
            "recall_severe": _recall_severe,
        },
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive,
    )
    summary = mf.by_group.copy()
    summary.loc["__overall__"] = mf.overall
    summary["dp_ratio"] = demographic_parity_ratio(
        y_true=(y_true == severe_label),
        y_pred=(y_pred == severe_label),
        sensitive_features=sensitive,
    )
    summary["eo_diff"] = equalized_odds_difference(
        y_true=(y_true == severe_label),
        y_pred=(y_pred == severe_label),
        sensitive_features=sensitive,
    )
    return summary


def svi_quartile(svi: pd.Series) -> pd.Series:
    return pd.qcut(svi, q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])


# -----------------------------------------------------------------------------
# SHAP mean-abs by subgroup
# -----------------------------------------------------------------------------
def shap_mean_abs_by_group(
    shap_values_for_class: np.ndarray, feature_names, group: pd.Series,
) -> pd.DataFrame:
    """
    Given 2-D SHAP array (n_samples, n_features) for ONE class, return a
    DataFrame of mean|SHAP| per feature per group category.
    """
    out = {}
    for g, idx in group.groupby(group).groups.items():
        idx = np.asarray(idx)
        mab = np.mean(np.abs(shap_values_for_class[idx, :]), axis=0)
        out[str(g)] = mab
    return pd.DataFrame(out, index=feature_names).sort_values(
        by=list(out.keys())[0], ascending=False
    )
