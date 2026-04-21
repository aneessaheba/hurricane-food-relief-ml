"""
Food Relief Priority Index: probability(High or Severe) × food fragility.

CRITICAL: every MinMaxScaler used to normalize a fragility component MUST be
fitted on the TRAIN split only, then transform applied to val/test. Otherwise
information from future storms leaks into the priority score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FRAGILITY_COMPONENTS = {
    # (column, invert?)  invert=True means "low value = high fragility"
    "food_desert_flag":            False,   # 0/1 directly
    "snap_retailers_per_1k":       True,    # low density = high fragility
    "dist_nearest_supermarket_mi": False,   # far = high fragility
    "pct_no_vehicle":              False,   # high % = high fragility
}


def fit_fragility_scalers(train_df: pd.DataFrame) -> Dict[str, MinMaxScaler]:
    """Fit one MinMaxScaler per component on the TRAIN split only."""
    scalers: Dict[str, MinMaxScaler] = {}
    for col in FRAGILITY_COMPONENTS:
        if col == "food_desert_flag":
            continue
        s = MinMaxScaler()
        s.fit(train_df[[col]].fillna(0))
        scalers[col] = s
    return scalers


def apply_fragility(
    df: pd.DataFrame, scalers: Dict[str, MinMaxScaler],
) -> pd.DataFrame:
    """Compute food_fragility_score for every row using pre-fit scalers."""
    df = df.copy()
    parts = []
    for col, invert in FRAGILITY_COMPONENTS.items():
        if col == "food_desert_flag":
            parts.append(df[col].fillna(0).astype(float))
            continue
        x = scalers[col].transform(df[[col]].fillna(0)).ravel()
        if invert:
            x = 1.0 - x
        parts.append(pd.Series(x, index=df.index))
    df["food_fragility_score"] = 0.25 * sum(parts)
    return df


def priority_index(
    df: pd.DataFrame, proba: np.ndarray, class_names,
    high_labels=("High", "Severe"),
) -> pd.DataFrame:
    """
    Multiply P(High or Severe) by food_fragility_score. Returns df with
    prob_high_or_severe, food_relief_priority_index, priority_index_norm,
    and priority_rank columns.
    """
    high_cols = [i for i, c in enumerate(class_names) if c in high_labels]
    if not high_cols:
        raise ValueError(f"No matching classes in {class_names} for {high_labels}")
    prob_hs = proba[:, high_cols].sum(axis=1)
    df = df.copy()
    df["prob_high_or_severe"] = prob_hs
    df["food_relief_priority_index"] = prob_hs * df["food_fragility_score"].to_numpy()
    mx = df["food_relief_priority_index"].max() or 1.0
    df["priority_index_norm"] = df["food_relief_priority_index"] / mx
    df["priority_rank"] = (
        df["priority_index_norm"].rank(ascending=False, method="dense").astype(int)
    )
    return df


def save_scalers(scalers: Dict[str, MinMaxScaler], path: Path) -> None:
    joblib.dump(scalers, path)


def load_scalers(path: Path) -> Dict[str, MinMaxScaler]:
    return joblib.load(path)
