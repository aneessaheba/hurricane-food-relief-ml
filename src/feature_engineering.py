"""
Feature engineering: compute targets, bin severity class, assign splits,
and impute missing values group by group.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    HURRICANE_META, SEVERITY_BINS, SEVERITY_LABELS,
    FEATURE_GROUPS,
)


# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute verified_damage_per_1000, pct_major_substantial, and
    approved_dollars_per_capita. Handles zero-population / zero-inspected
    rows without ZeroDivisionError.
    """
    df = df.copy()
    pop = df["population"].replace(0, np.nan)
    insp = df["total_inspected"].replace(0, np.nan)

    df["verified_damage_per_1000"] = (df["total_inspected"] / pop) * 1000
    df["pct_major_substantial"] = np.where(
        df["total_inspected"] > 0,
        df["total_major_substantial"] / insp * 100,
        0.0,
    )
    df["approved_dollars_per_capita"] = df["total_approved_dollars"] / pop
    return df


def bin_severity(
    s: pd.Series,
    bins: List[float] = SEVERITY_BINS,
    labels: List[str] = SEVERITY_LABELS,
    qcut_fallback_threshold: float = 0.80,
) -> pd.Series:
    """
    Bin `verified_damage_per_1000` into Low/Medium/High/Severe.

    If the initial binning places more than `qcut_fallback_threshold` of
    observations in a single class, fall back to quantile binning.
    """
    out = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    counts = out.value_counts(normalize=True, dropna=True)
    if not counts.empty and counts.max() > qcut_fallback_threshold:
        print(f"[warn] severity binning dominated by one class "
              f"({counts.idxmax()}={counts.max():.1%}); falling back to qcut")
        out = pd.qcut(s, q=4, labels=labels, duplicates="drop")
    return out


# -----------------------------------------------------------------------------
# Train/val/test split from hurricane metadata
# -----------------------------------------------------------------------------
def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    split_map = {h["disaster_number"]: h["split"] for h in HURRICANE_META}
    name_map  = {h["disaster_number"]: h["name"] for h in HURRICANE_META}
    year_map  = {h["disaster_number"]: h["year"] for h in HURRICANE_META}
    cat_map   = {h["disaster_number"]: h["category"] for h in HURRICANE_META}
    wind_map  = {h["disaster_number"]: h["max_wind_kt"] for h in HURRICANE_META}
    df = df.copy()
    df["train_test_split"] = df["disaster_number"].map(split_map)
    df["hurricane_name"]   = df["disaster_number"].map(name_map)
    df["hurricane_year"]   = df["disaster_number"].map(year_map)
    df["hurricane_category"] = df["disaster_number"].map(cat_map)
    # Fall back to per-hurricane max wind if the at-landfall column is missing
    if "max_wind_speed_kt" not in df.columns:
        df["max_wind_speed_kt"] = df["disaster_number"].map(wind_map)
    return df


# -----------------------------------------------------------------------------
# Missing-value handling
# -----------------------------------------------------------------------------
def _state_median_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        med = df.groupby("state")[c].transform("median")
        df[c] = df[c].fillna(med).fillna(df[c].median())
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute per the project spec:
      * SVI columns: -999 -> NaN -> state-median
      * Food access columns: NaN -> 0 and add food_data_missing flag
      * Flood column: NaN -> 0 (not mapped == not in flood zone)
      * Others: < 5% missing -> median impute within state
    """
    df = df.copy()

    # SVI: -999 sentinel already stripped in acquisition but re-assert
    svi_cols = FEATURE_GROUPS["svi"]
    for c in svi_cols:
        if c in df.columns:
            df[c] = df[c].replace(-999, np.nan)
    df = _state_median_impute(df, svi_cols)

    # Food access: NaN -> 0, plus missing flag
    food_cols = FEATURE_GROUPS["food_access"]
    any_missing = df[food_cols].isna().any(axis=1) if set(food_cols).issubset(df.columns) else pd.Series(False, index=df.index)
    df["food_data_missing"] = any_missing.astype(int)
    for c in food_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Flood
    for c in FEATURE_GROUPS["flood"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Demographics + storm: state-median impute
    num_cols = FEATURE_GROUPS["demographics"] + FEATURE_GROUPS["storm"]
    df = _state_median_impute(df, num_cols)

    return df


# -----------------------------------------------------------------------------
# Derived demographic shares
# -----------------------------------------------------------------------------
def derive_demographic_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ACS raw counts to percent shares."""
    df = df.copy()
    pop = df["population"].replace(0, np.nan)

    if "poverty_count" in df.columns:
        df["pct_poverty"] = df["poverty_count"] / pop * 100
    if "renters" in df.columns:
        # B25003_003 = renter-occupied households; scale by total pop as a proxy
        # (better: total households B25003_001, but we use what spec lists).
        df["pct_renters"] = df["renters"] / pop * 100

    elderly_cols = [c for c in df.columns if c.startswith(("male_6", "male_7", "male_8",
                                                            "female_6", "female_7", "female_8"))]
    if elderly_cols:
        df["pct_elderly_65plus"] = df[elderly_cols].sum(axis=1) / pop * 100

    if "white_alone" in df.columns:
        df["pct_minority"] = (1 - df["white_alone"] / pop) * 100

    if "no_vehicle_households" in df.columns:
        df["pct_no_vehicle"] = df["no_vehicle_households"] / pop * 100

    if "mobile_homes" in df.columns:
        df["pct_mobile_homes"] = df["mobile_homes"] / pop * 100

    return df
