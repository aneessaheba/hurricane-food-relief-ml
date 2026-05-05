import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "zip_code": ["32003", "32004", "00000"],
        "population": [10000, 5000, 0],
        "total_inspected": [100, 50, 0],
        "total_major_substantial": [20, 10, 0],
        "total_approved_dollars": [500000, 250000, 0],
        "poverty_count": [1000, 1000, 0],
        "svi_overall": [0.8, -999, 0.2],
        "state": ["FL", "FL", "FL"]
    })

@pytest.fixture
def dummy_zcta_gdf():
    df = pd.DataFrame({
        "ZCTA5CE20": ["32003", "32004"],
        "Latitude": [30.1, 30.2],
        "Longitude": [-81.6, -81.7]
    })
    return gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    )

@pytest.fixture
def dummy_nfhl_gdf():
    df = pd.DataFrame({
        "FLD_ZONE": ["A", "X"],
        "COUNTY_FIPS": ["12019", "12019"],
        "Latitude": [30.1, 30.2],
        "Longitude": [-81.6, -81.7]
    })
    # just point geometries for simplicity, though actual is polygon
    return gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    )
