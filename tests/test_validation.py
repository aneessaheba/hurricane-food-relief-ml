import numpy as np
import pandas as pd
import pytest

from src.validation import assert_valid, source_coverage, validate_abt


@pytest.fixture
def abt():
    # Two Ida (TEST, LA) rows and one Harvey (TRAIN, TX) row.
    return pd.DataFrame({
        "disaster_number": [4611, 4611, 4332],
        "zip_code": ["70001", "70002", "77001"],
        "state": ["LA", "LA", "TX"],
        "hurricane_name": ["Ida", "Ida", "Harvey"],
        "train_test_split": ["TEST", "TEST", "TRAIN"],
        "population": [1000.0, 2000.0, 500.0],
        "median_income": [50000.0, 42000.0, 61000.0],
        "pct_poverty": [10.0, 20.0, 5.0],
        "svi_overall": [0.2, 0.9, 0.5],
        "distance_to_track_km": [10.0, 40.0, 25.0],
        "verified_damage_per_1000": [1.0, 30.0, 10.0],
        "damage_severity_class": ["Low", "Severe", "High"],
        "snap_participation_pct": [np.nan, 12.0, 3.0],
    })


def checks(issues, level):
    return {i.check for i in issues if i.level == level}


def test_clean_table_has_no_issues(abt):
    assert validate_abt(abt) == []
    assert assert_valid(abt) == []


def test_duplicate_key_is_an_error(abt):
    abt.loc[1, "zip_code"] = "70001"
    issues = validate_abt(abt)
    assert "unique_key" in checks(issues, "error")
    assert next(i for i in issues if i.check == "unique_key").n_rows == 2


@pytest.mark.parametrize("col,value,check", [
    ("zip_code", "7001", "zip_code"),
    ("population", 0.0, "population"),
    ("train_test_split", "TRAIN", "split"),
    ("svi_overall", 1.5, "svi_overall"),
    ("damage_severity_class", "Medium", "damage_severity_class"),
])
def test_hard_errors(abt, col, value, check):
    abt.loc[0, col] = value
    assert check in checks(validate_abt(abt), "error")
    with pytest.raises(ValueError, match=check):
        assert_valid(abt)


@pytest.mark.parametrize("col,value,check", [
    ("median_income", -666666666.0, "median_income"),
    ("pct_poverty", 557.5, "pct_poverty"),
    ("distance_to_track_km", 2050.0, "distance_to_track_km"),
    ("state", "CA", "state"),
])
def test_suspicious_values_are_warnings(abt, col, value, check):
    abt.loc[2, col] = value
    issues = assert_valid(abt)  # warnings do not raise
    assert check in checks(issues, "warning")


def test_missing_required_columns():
    issues = validate_abt(pd.DataFrame({"zip_code": ["70001"]}))
    assert [i.check for i in issues] == ["columns"]


def test_source_coverage_reports_gaps_per_storm(abt):
    cov = source_coverage(abt)
    assert cov.loc["Ida", "food_access"] == 0.5
    assert cov.loc["Harvey", "food_access"] == 0.0
    assert cov.loc["Ida", "svi"] == 0.0


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------
from pathlib import Path

from src.validation import check_against_dictionary

ROOT = Path(__file__).resolve().parents[1]


def _dictionary(rows):
    return pd.DataFrame(rows, columns=["column", "group", "dtype", "units", "source", "description"])


def test_dictionary_file_parses_with_six_fields():
    d = pd.read_csv(ROOT / "outputs" / "data_dictionary.csv")
    assert list(d.columns) == ["column", "group", "dtype", "units", "source", "description"]
    assert not d["column"].duplicated().any()
    assert d["description"].notna().all()


def test_dictionary_matches_table():
    df = pd.DataFrame({"zip_code": ["70001"], "population": [10.0], "food_desert_flag": [1.0]})
    d = _dictionary([
        ["zip_code", "identifier", "string", "", "", ""],
        ["population", "demographic", "float", "", "", ""],
        ["food_desert_flag", "food_access", "binary", "", "", ""],
        ["priority_rank", "derived_output", "int", "", "", ""],  # later step: exempt
    ])
    assert check_against_dictionary(df, d) == []


def test_dictionary_flags_undocumented_wrong_type_and_bad_binary():
    df = pd.DataFrame({"zip_code": ["70001"], "population": ["ten"],
                       "food_desert_flag": [2.0], "extra": [1]})
    d = _dictionary([
        ["zip_code", "identifier", "string", "", "", ""],
        ["population", "demographic", "float", "", "", ""],
        ["food_desert_flag", "food_access", "binary", "", "", ""],
        ["median_income", "demographic", "float", "", "", ""],
    ])
    msgs = [str(i) for i in check_against_dictionary(df, d)]
    assert any("missing from the dictionary: ['extra']" in m for m in msgs)
    assert any("population: documented as float" in m for m in msgs)
    assert any("binary value other than 0/1" in m for m in msgs)
    assert any("[warning]" in m and "median_income" in m for m in msgs)
