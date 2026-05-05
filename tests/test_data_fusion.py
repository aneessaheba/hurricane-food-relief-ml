import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from src.data_fusion import pct_in_floodplain, snap_retailers_per_zip, distance_to_track_km, tract_to_zip

def test_pct_in_floodplain_empty():
    zcta = gpd.GeoDataFrame()
    nfhl = gpd.GeoDataFrame()
    result = pct_in_floodplain(zcta, nfhl)
    assert result.empty
    assert "pct_in_100yr_floodplain" in result.columns

def test_pct_in_floodplain_missing_geom(dummy_zcta_gdf):
    nfhl = pd.DataFrame({"FLD_ZONE": ["A"]}) # Not a gdf, missing geom
    with pytest.raises(ValueError, match="pct_in_floodplain requires GeoDataFrames"):
        pct_in_floodplain(dummy_zcta_gdf, nfhl)

def test_distance_to_track_km():
    p1 = Point(-80.0, 30.0)
    # Line directly north
    track = LineString([(-81.0, 25.0), (-81.0, 35.0)])
    
    # 1 degree of longitude at 30N is roughly 96 km.
    # The exact distance should be computed and be > 0.
    dist = distance_to_track_km(p1, track)
    assert dist > 90 and dist < 100

def test_tract_to_zip():
    df = pd.DataFrame({
        "TRACT": ["12019000100", "12019000200"],
        "val": [100.0, 50.0]
    })
    cw = pd.DataFrame({
        "TRACT": ["12019000100", "12019000200"],
        "ZIP": ["32003", "32003"],
        "RES_RATIO": [0.8, 0.2]
    })
    
    result = tract_to_zip(df, cw, tract_col="TRACT", value_cols=["val"])
    assert result.loc[result["zip_code"] == "32003", "val"].iloc[0] == 90.0 # 100*0.8 + 50*0.2
