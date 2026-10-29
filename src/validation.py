"""
Data checks for the fused ZIP x hurricane table (ABT).

`validate_abt` returns a list of `Issue`s. Errors are violations the rest of
the pipeline relies on (duplicate keys, impossible values, labels that do not
match their bins); warnings are suspicious values worth a look but not fatal.
`source_coverage` reports, per storm, how often each feature group has a gap,
and `check_against_dictionary` keeps the table in step with
outputs/data_dictionary.csv.

Run against the corrected table built by src.pipeline (the default) or any ABT:
    python -m src.validation [data/processed/abt_clean.csv]
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

    if "state" in df.columns:
        from src.data_fusion import zip_matches_state
        wrong = [not zip_matches_state(z, s) for z, s in zip(zips, df["state"])]
        _issue(issues, "error", "zip_state", pd.Series(wrong, index=df.index),
               "whose ZIP does not belong to their state (mis-keyed; see finalize_abt)")

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


# Dictionary dtype -> numpy dtype kinds that satisfy it.
_DTYPE_KINDS = {"int": "iu", "float": "fiu", "string": "O", "binary": "biuf"}
# Columns written by the priority-index step, not stored in the ABT itself.
_OUTPUT_GROUPS = {"derived_output"}


def check_against_dictionary(df: pd.DataFrame, dictionary: pd.DataFrame) -> List[Issue]:
    """
    Compare the ABT with outputs/data_dictionary.csv: every ABT column must be
    documented with a matching dtype, and binary columns must hold only 0/1.
    Documented columns absent from the ABT are warnings (outputs of later
    steps are exempt).
    """
    issues: List[Issue] = []
    doc = dictionary.set_index("column")
    dup = doc.index[doc.index.duplicated()].unique().tolist()
    if dup:
        issues.append(Issue("error", "dictionary", f"duplicate entries {dup}"))
        doc = doc[~doc.index.duplicated()]

    undocumented = [c for c in df.columns if c not in doc.index]
    if undocumented:
        issues.append(Issue("error", "dictionary",
                            f"ABT columns missing from the dictionary: {undocumented}"))

    absent = [c for c, g in doc["group"].items()
              if c not in df.columns and g not in _OUTPUT_GROUPS]
    if absent:
        issues.append(Issue("warning", "dictionary",
                            f"documented columns not in the ABT: {absent}"))

    for c in df.columns.intersection(doc.index):
        declared = doc.at[c, "dtype"]
        kinds = _DTYPE_KINDS.get(declared)
        if kinds is None:
            issues.append(Issue("error", "dictionary", f"{c}: unknown dtype {declared!r}"))
        elif df[c].dtype.kind not in kinds:
            issues.append(Issue("error", "dictionary",
                                f"{c}: documented as {declared}, stored as {df[c].dtype}"))
        elif declared == "binary":
            _issue(issues, "error", c, ~df[c].isin([0, 1]) & df[c].notna(),
                   "with a binary value other than 0/1")
    return issues


def assert_valid(df: pd.DataFrame) -> List[Issue]:
    """Raise ValueError on any error-level issue; return the warnings."""
    issues = validate_abt(df)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise ValueError("ABT failed validation:\n" + "\n".join(map(str, errors)))
    return issues


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/abt_clean.csv")
    abt = pd.read_csv(path, dtype={"zip_code": str})
    found = validate_abt(abt)
    dictionary_path = Path(__file__).resolve().parents[1] / "outputs" / "data_dictionary.csv"
    if dictionary_path.exists():
        found += check_against_dictionary(abt, pd.read_csv(dictionary_path))
    for i in found:
        print(i)
    print(f"\n{len(abt):,} rows checked; "
          f"{sum(i.level == 'error' for i in found)} errors, "
          f"{sum(i.level == 'warning' for i in found)} warnings")
    print("\nShare of rows with a gap in each feature group, by storm:")
    print(source_coverage(abt).round(3).to_string())
    sys.exit(1 if any(i.level == "error" for i in found) else 0)
