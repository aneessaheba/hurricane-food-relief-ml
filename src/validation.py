"""
Data checks for the fused ZIP x hurricane table (ABT).

`validate_abt` returns a list of `Issue`s. Errors are violations the rest of
the pipeline relies on (duplicate keys, impossible values, labels that do not
match their bins); warnings are suspicious values worth a look but not fatal.
`source_coverage` reports, per storm, how often each feature group has a gap.

Run against the processed table with:
    python -m src.validation data/processed/abt.csv
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    FEATURE_GROUPS, HURRICANE_META, SEVERITY_BINS, SEVERITY_LABELS,
)

KEY = ["disaster_number", "zip_code"]
# ACS codes for "estimate not available" (e.g. -666666666 for median income).
CENSUS_SENTINEL_MAX = -100_000_000
FAR_FROM_TRACK_KM = 1000


@dataclass
class Issue:
    level: str      # "error" or "warning"
    check: str
    message: str
    n_rows: int = 0

    def __str__(self) -> str:
        return f"[{self.level}] {self.check}: {self.message}"


def _issue(issues: List[Issue], level: str, check: str, mask: pd.Series, what: str) -> None:
    n = int(mask.sum())
    if n:
        issues.append(Issue(level, check, f"{n:,} rows {what}", n))


def validate_abt(df: pd.DataFrame) -> List[Issue]:
    """Check the ABT's keys, value ranges, split labels and target classes."""
    issues: List[Issue] = []

    missing = [c for c in KEY + ["population", "train_test_split"] if c not in df.columns]
    if missing:
        return [Issue("error", "columns", f"missing required columns {missing}")]

    zips = df["zip_code"].astype(str)
    _issue(issues, "error", "zip_code", ~zips.str.fullmatch(r"\d{5}"),
           "with a zip_code that is not 5 digits")
    _issue(issues, "error", "unique_key", df.duplicated(KEY, keep=False),
           "share a (disaster_number, zip_code) key")
    _issue(issues, "error", "population", ~(df["population"] > 0),
           "with missing or non-positive population")

    meta = {h["disaster_number"]: h for h in HURRICANE_META}
    expected_split = df["disaster_number"].map({k: v["split"] for k, v in meta.items()})
    _issue(issues, "error", "split", df["train_test_split"] != expected_split,
           "whose train_test_split does not match HURRICANE_META")

    for c in FEATURE_GROUPS["svi"]:
        if c in df.columns:
            _issue(issues, "error", c, (df[c] < 0) | (df[c] > 1),
                   "with SVI outside [0, 1]")

    if {"verified_damage_per_1000", "damage_severity_class"} <= set(df.columns):
        expected = pd.cut(df["verified_damage_per_1000"], bins=SEVERITY_BINS,
                          labels=SEVERITY_LABELS, include_lowest=True).astype(str)
        _issue(issues, "error", "damage_severity_class",
               expected != df["damage_severity_class"].astype(str),
               "whose class does not match SEVERITY_BINS")

    numeric = df.select_dtypes("number")
    for c in numeric.columns:
        _issue(issues, "warning", c, numeric[c] <= CENSUS_SENTINEL_MAX,
               "hold a Census 'not available' code (e.g. -666666666)")

    for c in [c for c in df.columns if c.startswith("pct_")]:
        _issue(issues, "warning", c, (df[c] < 0) | (df[c] > 100),
               "outside 0-100%")

    if "distance_to_track_km" in df.columns:
        _issue(issues, "warning", "distance_to_track_km",
               df["distance_to_track_km"] > FAR_FROM_TRACK_KM,
               f"more than {FAR_FROM_TRACK_KM:,} km from the storm track "
               "(possible mis-keyed ZIP)")

    if "state" in df.columns:
        affected = df["disaster_number"].map(
            {k: set(v["states_affected"]) for k, v in meta.items()})
        outside = [s not in a if isinstance(a, set) else False
                   for s, a in zip(df["state"], affected)]
        _issue(issues, "warning", "state", pd.Series(outside, index=df.index),
               "in a state not listed in the storm's states_affected")

    return issues


def source_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Share of rows per storm with at least one column of a feature group missing."""
    groups = {g: [c for c in FEATURE_GROUPS[g] if c in df.columns]
              for g in ("demographics", "svi", "food_access", "flood", "storm")}
    by = df["hurricane_name"] if "hurricane_name" in df.columns else df["disaster_number"]
    out = pd.DataFrame({
        g: df[cols].isna().any(axis=1).groupby(by).mean()
        for g, cols in groups.items() if cols
    })
    out.index.name = by.name
    return out


def assert_valid(df: pd.DataFrame) -> List[Issue]:
    """Raise ValueError on any error-level issue; return the warnings."""
    issues = validate_abt(df)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise ValueError("ABT failed validation:\n" + "\n".join(map(str, errors)))
    return issues


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/abt.csv")
    abt = pd.read_csv(path, dtype={"zip_code": str})
    found = validate_abt(abt)
    for i in found:
        print(i)
    print(f"\n{len(abt):,} rows checked; "
          f"{sum(i.level == 'error' for i in found)} errors, "
          f"{sum(i.level == 'warning' for i in found)} warnings")
    print("\nShare of rows with a gap in each feature group, by storm:")
    print(source_coverage(abt).round(3).to_string())
    sys.exit(1 if any(i.level == "error" for i in found) else 0)
