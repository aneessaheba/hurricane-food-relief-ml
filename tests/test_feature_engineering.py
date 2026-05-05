import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import compute_targets, derive_demographic_shares, impute_missing

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
