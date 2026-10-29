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


# ---------------------------------------------------------------------------
# Re-weighting the FRPI (dashboard slider)
# ---------------------------------------------------------------------------
from pathlib import Path

from src.priority_index import rerank

RANKINGS = Path(__file__).resolve().parents[1] / "outputs" / "priority_rankings.csv"


def _ranked():
    return pd.DataFrame({
        "hurricane_name": ["Ida", "Ida", "Ida", "Ian", "Ian"],
        "prob_high_or_severe": [0.9, 0.5, 0.1, 0.8, 0.2],
        "food_fragility_score": [0.1, 0.5, 0.9, 0.3, 0.7],
    })


def test_rerank_extremes_rank_by_one_component():
    risk_only = rerank(_ranked(), weight_risk=1.0)
    assert risk_only["priority_rank"].tolist() == [1, 2, 3, 1, 2]
    fragility_only = rerank(_ranked(), weight_risk=0.0)
    assert fragility_only["priority_rank"].tolist() == [3, 2, 1, 2, 1]


def test_rerank_normalizes_within_each_storm():
    out = rerank(_ranked(), weight_risk=0.6)
    for _, g in out.groupby("hurricane_name"):
        assert g["priority_index_norm"].min() == pytest.approx(0.0)
        assert g["priority_index_norm"].max() == pytest.approx(1.0)


def test_rerank_constant_storm_and_bad_weight():
    flat = _ranked().assign(prob_high_or_severe=0.5, food_fragility_score=0.5)
    assert (rerank(flat)["priority_index_norm"] == 0.0).all()
    with pytest.raises(ValueError):
        rerank(_ranked(), weight_risk=1.2)


@pytest.mark.skipif(not RANKINGS.exists(), reason="priority_rankings.csv not built")
def test_rerank_reproduces_published_rankings_at_default_weight():
    r = pd.read_csv(RANKINGS, dtype={"zip_code": str})
    out = rerank(r, weight_risk=0.6)
    assert np.allclose(out["food_relief_priority_index"], r["food_relief_priority_index"])
    assert np.allclose(out["priority_index_norm"], r["priority_index_norm"])
    assert (out["priority_rank"] == r["priority_rank"]).all()
