"""
Food Relief Priority Index — combines classifier risk with food-access fragility.

food_fragility_score (0-1): composite of food access deficiencies and SVI.
    Higher = more food-fragile community.

food_relief_priority_index: weighted blend of P(High ∪ Severe) and food fragility.
    Higher = higher priority for food relief allocation.

Scalers are fit on TRAIN only and persisted so TEST predictions don't leak the
test distribution into the fragility score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Components where a HIGHER raw value means MORE fragile
FRAGILITY_HIGHER_IS_WORSE = [
    "dist_nearest_supermarket_mi",
    "snap_households_avg",
    "svi_overall",
]
# Components where a HIGHER raw value means LESS fragile (inverted)
FRAGILITY_HIGHER_IS_BETTER = [
    "snap_retailer_count",
    "snap_retailers_per_1k",
]
# Binary fragility flags (already 0/1, higher = more fragile)
FRAGILITY_BINARY = ["food_desert_flag"]


def fit_fragility_scalers(train_df: pd.DataFrame) -> Dict[str, MinMaxScaler]:
    """Fit a MinMaxScaler for each continuous fragility component on TRAIN."""
    scalers: Dict[str, MinMaxScaler] = {}
    for col in FRAGILITY_HIGHER_IS_WORSE + FRAGILITY_HIGHER_IS_BETTER:
        if col not in train_df.columns:
            continue
        vals = train_df[col].dropna().to_numpy(dtype=float).reshape(-1, 1)
        if len(vals) == 0:
            continue
        scalers[col] = MinMaxScaler().fit(vals)
    return scalers


def save_scalers(scalers: Dict[str, MinMaxScaler], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, path)


def load_scalers(path: Path) -> Dict[str, MinMaxScaler]:
    return joblib.load(path)


def _scale_col(values: pd.Series, scaler: MinMaxScaler) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    mask = ~np.isnan(arr)
    if mask.any():
        out[mask] = scaler.transform(arr[mask].reshape(-1, 1)).ravel()
    return np.clip(out, 0.0, 1.0)


def apply_fragility(df: pd.DataFrame, scalers: Dict[str, MinMaxScaler]) -> pd.DataFrame:
    """Compute `food_fragility_score` (0-1) per row using fitted scalers."""
    df = df.copy()
    components = []

    for col in FRAGILITY_HIGHER_IS_WORSE:
        if col in df.columns and col in scalers:
            components.append(_scale_col(df[col], scalers[col]))

    for col in FRAGILITY_HIGHER_IS_BETTER:
        if col in df.columns and col in scalers:
            components.append(1.0 - _scale_col(df[col], scalers[col]))

    for col in FRAGILITY_BINARY:
        if col in df.columns:
            components.append(df[col].fillna(0).astype(float).clip(0, 1).to_numpy())

    if not components:
        df["food_fragility_score"] = np.nan
        return df

    stacked = np.vstack(components)
    df["food_fragility_score"] = np.nanmean(stacked, axis=0)
    return df


def priority_index(
    test_df: pd.DataFrame,
    proba: np.ndarray,
    class_names: list,
    weight_risk: float = 0.6,
    weight_fragility: float = 0.4,
) -> pd.DataFrame:
    """
    Combine P(High ∪ Severe) with the food-fragility score.

    Returns the input df with these columns added:
      - prob_high_or_severe         : sum of model probabilities for High + Severe
      - food_relief_priority_index  : weighted blend of risk and fragility
      - priority_index_norm         : index rescaled to [0,1] for mapping
      - priority_rank               : 1 = highest priority
    """
    df = test_df.copy().reset_index(drop=True)

    if "food_fragility_score" not in df.columns:
        raise ValueError("Run apply_fragility() before priority_index()")

    high_severe_idx = [i for i, c in enumerate(class_names) if c in ("High", "Severe")]
    if not high_severe_idx:
        raise ValueError(f"class_names must include 'High' and/or 'Severe', got {class_names}")
    df["prob_high_or_severe"] = proba[:, high_severe_idx].sum(axis=1)

    risk = df["prob_high_or_severe"].fillna(0).to_numpy()
    frag = df["food_fragility_score"].fillna(0).to_numpy()
    df["food_relief_priority_index"] = weight_risk * risk + weight_fragility * frag

    idx = df["food_relief_priority_index"]
    rng = idx.max() - idx.min()
    df["priority_index_norm"] = (idx - idx.min()) / rng if rng > 0 else 0.0

    df["priority_rank"] = (
        idx.rank(ascending=False, method="min").astype(int)
    )
    return df
