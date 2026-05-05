import pytest
import numpy as np
import pandas as pd
from src.priority_index import fit_fragility_scalers, apply_fragility, priority_index

def test_fragility_scalers():
    df_train = pd.DataFrame({
        "dist_nearest_supermarket_mi": [1.0, 10.0],
        "snap_retailer_count": [0.0, 5.0],
        "food_desert_flag": [0.0, 1.0]
    })
    
    scalers = fit_fragility_scalers(df_train)
    
    df_test = pd.DataFrame({
        "dist_nearest_supermarket_mi": [5.5],
        "snap_retailer_count": [2.5],
        "food_desert_flag": [1.0]
    })
    
    result = apply_fragility(df_test, scalers)
    score = result["food_fragility_score"].iloc[0]
    
    # distance is 5.5 (0.5 scaled) - WORSE is higher so 0.5
    # count is 2.5 (0.5 scaled) - BETTER is higher so inverted is 1 - 0.5 = 0.5
    # desert flag is 1.0
    # Mean of (0.5, 0.5, 1.0) is 2.0 / 3 = 0.666...
    assert "food_fragility_score" in result.columns
    assert np.isclose(score, 0.666666)
    
def test_priority_index():
    df_test = pd.DataFrame({
        "food_fragility_score": [1.0, 0.0]
    })
    
    proba = np.array([
        [0.1, 0.1, 0.4, 0.4], # 0.8 high/severe
        [0.8, 0.2, 0.0, 0.0]  # 0.0 high/severe
    ])
    
    class_names = ["Low", "Medium", "High", "Severe"]
    
    res = priority_index(df_test, proba, class_names, weight_risk=0.6, weight_fragility=0.4)
    
    idx_0 = res.loc[0, "food_relief_priority_index"]
    idx_1 = res.loc[1, "food_relief_priority_index"]
    
    # row 0: risk 0.8 * 0.6 + frag 1.0 * 0.4 = 0.48 + 0.4 = 0.88
    assert np.isclose(idx_0, 0.88)
    # row 1: risk 0.0 * 0.6 + frag 0.0 * 0.4 = 0.0
    assert np.isclose(idx_1, 0.0)
    
    # Priority norm should strictly be [0, 1]
    assert res["priority_index_norm"].max() <= 1.0
    assert res["priority_index_norm"].min() >= 0.0
