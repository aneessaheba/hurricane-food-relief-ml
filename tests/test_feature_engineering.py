import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import compute_targets, derive_demographic_shares, impute_missing, bin_severity

def test_compute_targets(dummy_df):
    result = compute_targets(dummy_df)
    
    # 32003: 100 inspected / 10000 pop * 1000 = 10
    assert result.loc[result["zip_code"] == "32003", "verified_damage_per_1000"].iloc[0] == 10.0
    
    # 32003: 20 major / 100 inspected = 20%
    assert result.loc[result["zip_code"] == "32003", "pct_major_substantial"].iloc[0] == 20.0
    
    # 00000: 0 pop should not raise zero division error, it should be NaN
    assert pd.isna(result.loc[result["zip_code"] == "00000", "verified_damage_per_1000"].iloc[0])

def test_compute_targets_missing_columns():
    df = pd.DataFrame({"population": [100]})
    with pytest.raises(ValueError, match="compute_targets expected columns"):
        compute_targets(df)

def test_impute_missing(dummy_df):
    result = impute_missing(dummy_df)
    # Check that -999 was imputed
    assert result.loc[result["zip_code"] == "32004", "svi_overall"].iloc[0] != -999
    # Median of [0.8, 0.2] is 0.5
    assert result.loc[result["zip_code"] == "32004", "svi_overall"].iloc[0] == 0.5

def test_derive_demographic_shares(dummy_df):
    result = derive_demographic_shares(dummy_df)
    assert "pct_poverty" in result.columns
    # 32003: 1000 / 10000 = 10%
    assert result.loc[result["zip_code"] == "32003", "pct_poverty"].iloc[0] == 10.0

def test_bin_severity_normal():
    s = pd.Series([0.5, 1.0, 3.0, 5.0, 9.0, 12.0, 20.0, 30.0] * 10)
    result = bin_severity(s)
    assert set(result.dropna().unique()).issubset({"Low", "Medium", "High", "Severe"})

def test_bin_severity_qcut_fallback_no_crash():
    # 90% same value — triggers qcut fallback; must not raise ValueError
    s = pd.Series([0.1] * 90 + [5.0] * 5 + [15.0] * 3 + [30.0] * 2)
    result = bin_severity(s)
    assert len(result) == 100
    assert result.notna().any()

def test_derive_demographic_shares_uses_total_households():
    df = pd.DataFrame({
        "zip_code": ["32003"],
        "population": [10000],
        "total_households": [4000],
        "renters": [1200],
    })
    result = derive_demographic_shares(df)
    # 1200 renters / 4000 households = 30%
    assert result["pct_renters"].iloc[0] == 30.0


# ---------------------------------------------------------------------------
# Severity bins, split assignment and imputation
# ---------------------------------------------------------------------------
from config import HURRICANE_META
from src.feature_engineering import assign_split, _state_median_impute


def test_bin_severity_thresholds_are_right_inclusive():
    # SEVERITY_BINS = (-inf, 2], (2, 8], (8, 25], (25, inf). No ABT row sits
    # exactly on a threshold, so this only pins the convention.
    s = pd.Series([0.0, 2.0, 2.01, 8.0, 8.01, 25.0, 25.01, 500.0] * 3)
    result = bin_severity(s).tolist()[:8]
    assert result == ["Low", "Low", "Medium", "Medium", "High", "High", "Severe", "Severe"]


def test_bin_severity_keeps_nan_rows_unlabelled():
    s = pd.Series([1.0, np.nan, 10.0, 30.0, 5.0] * 4)
    result = bin_severity(s)
    assert result.isna().sum() == 4
    assert result.notna().sum() == 16


def test_assign_split_matches_hurricane_meta():
    df = pd.DataFrame({"disaster_number": [h["disaster_number"] for h in HURRICANE_META]})
    out = assign_split(df)
    for h, (_, row) in zip(HURRICANE_META, out.iterrows()):
        assert row["train_test_split"] == h["split"]
        assert row["hurricane_name"] == h["name"]
        assert row["hurricane_year"] == h["year"]
        assert row["hurricane_category"] == h["category"]
        assert row["max_wind_speed_kt"] == h["max_wind_kt"]
    assert out.groupby("train_test_split")["hurricane_name"].nunique().to_dict() == {
        "TRAIN": 5, "VAL": 4, "TEST": 5}


def test_assign_split_keeps_existing_wind_and_leaves_unknown_storms_blank():
    df = pd.DataFrame({"disaster_number": [4611, 1], "max_wind_speed_kt": [99.0, 50.0]})
    out = assign_split(df)
    assert out["max_wind_speed_kt"].tolist() == [99.0, 50.0]
    assert out.loc[1, ["train_test_split", "hurricane_name"]].isna().all()


def test_temporal_split_never_trains_on_later_storms():
    years = {s: [h["year"] for h in HURRICANE_META if h["split"] == s]
             for s in ("TRAIN", "VAL", "TEST")}
    assert max(years["TRAIN"]) < min(years["VAL"]) <= max(years["VAL"]) < min(years["TEST"])


def test_state_median_impute_uses_state_then_global_median():
    df = pd.DataFrame({
        "state": ["FL", "FL", "FL", "TX", "TX", "GA"],
        "x": [1.0, 3.0, np.nan, 10.0, np.nan, np.nan],
    })
    out = _state_median_impute(df, ["x", "not_a_column"])
    assert out["x"].tolist() == [1.0, 3.0, 2.0, 10.0, 10.0, 3.0]  # GA has no data: global median
    assert df["x"].isna().sum() == 3  # input left untouched


def test_impute_missing_zero_fills_food_and_flood_and_flags_gaps():
    df = pd.DataFrame({
        "state": ["FL", "FL"],
        "food_desert_flag": [1.0, np.nan],
        "snap_retailer_count": [3.0, 4.0],
        "snap_retailers_per_1k": [0.5, 0.2],
        "dist_nearest_supermarket_mi": [1.0, 2.0],
        "snap_participation_pct": [5.0, 6.0],
        "pct_in_100yr_floodplain": [np.nan, 4.0],
        "nri_cflood_score": [10.0, np.nan],
        "population": [100.0, np.nan],
    })
    out = impute_missing(df)
    assert out["food_data_missing"].tolist() == [0, 1]
    assert out["food_desert_flag"].tolist() == [1.0, 0.0]
    assert out["pct_in_100yr_floodplain"].tolist() == [0.0, 4.0]
    assert out["nri_cflood_score"].tolist() == [10.0, 0.0]
    assert out["population"].tolist() == [100.0, 100.0]  # state median


def test_compute_targets_zero_inspected_gives_zero_share():
    df = pd.DataFrame({"population": [1000], "total_inspected": [0],
                       "total_major_substantial": [0], "total_approved_dollars": [0.0]})
    out = compute_targets(df)
    assert out["verified_damage_per_1000"].iloc[0] == 0.0
    assert out["pct_major_substantial"].iloc[0] == 0.0


def test_derive_demographic_shares_population_denominators():
    df = pd.DataFrame({
        "population": [1000.0], "white_alone": [600.0],
        "no_vehicle_households": [50.0], "mobile_homes": [20.0],
        "male_65_66": [30.0], "female_85_over": [20.0],
    })
    out = derive_demographic_shares(df)
    assert out["pct_minority"].iloc[0] == pytest.approx(40.0)
    assert out["pct_no_vehicle"].iloc[0] == pytest.approx(5.0)
    assert out["pct_mobile_homes"].iloc[0] == pytest.approx(2.0)
    assert out["pct_elderly_65plus"].iloc[0] == pytest.approx(5.0)
    # Without total_households, renters fall back to population (as in the published ABT).
    out = derive_demographic_shares(df.assign(renters=[300.0]))
    assert out["pct_renters"].iloc[0] == pytest.approx(30.0)
