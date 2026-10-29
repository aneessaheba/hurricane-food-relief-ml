import numpy as np
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


# ---------------------------------------------------------------------------
# Owner/renter reconciliation
# ---------------------------------------------------------------------------
from pyproj import Geod
from shapely.geometry import Polygon

from src.data_fusion import (
    build_track_linestring, compute_distance_to_track, merge_housing_assistance,
)


def _owners(**overrides):
    row = {"disasterNumber": 4611, "zipCode": "70001", "county": "Orleans",
           "state": "LA", "validRegistrations": 10, "totalInspected": 8,
           "totalApprovedIhpAmount": 1000.0,
           "noFemaInspectedDamage": 2,
           "femaInspectedDamageBetween1And10000": 3,
           "femaInspectedDamageBetween10001And20000": 1,
           "femaInspectedDamageBetween20001And30000": 1,
           "femaInspectedDamageGreaterThan30000": 1}
    row.update(overrides)
    return row


def _renters(**overrides):
    row = {"disasterNumber": 4611, "zipCode": "70001", "county": "Orleans",
           "state": "LA", "validRegistrations": 6, "totalInspected": 5,
           "totalApprovedIhpAmount": 400.0,
           "totalInspectedWithNoDamage": 1, "totalWithModerateDamage": 2,
           "totalWithMajorDamage": 1, "totalWithSubstantialDamage": 1}
    row.update(overrides)
    return row


def test_merge_housing_assistance_sums_owners_and_renters():
    out = merge_housing_assistance(pd.DataFrame([_owners()]), pd.DataFrame([_renters()]))
    row = out.iloc[0]
    assert len(out) == 1
    assert row["total_inspected"] == 13            # 8 owners + 5 renters
    assert row["validRegistrations"] == 16
    assert row["total_approved_dollars"] == 1400.0
    # owner >$20k buckets (1 + 1) + renter major + substantial (1 + 1)
    assert row["total_major_substantial"] == 4
    assert row["total_no_damage"] == 3
    assert row["total_moderate"] == 3              # owner $10-20k + renter moderate
    assert row["total_minor"] == 3                 # renters have no minor bucket


def test_merge_housing_assistance_collapses_multi_county_zips():
    # A ZIP split across two counties must not Cartesian-explode on the join.
    owners = pd.DataFrame([_owners(county="A", totalInspected=4),
                           _owners(county="B", totalInspected=6)])
    renters = pd.DataFrame([_renters(county="A", totalInspected=1),
                            _renters(county="B", totalInspected=2)])
    out = merge_housing_assistance(owners, renters)
    assert len(out) == 1
    assert out.iloc[0]["total_inspected"] == 13


def test_merge_housing_assistance_keeps_owner_only_and_renter_only_zips():
    owners = pd.DataFrame([_owners(zipCode="70001")])
    renters = pd.DataFrame([_renters(zipCode="70002")])
    out = merge_housing_assistance(owners, renters).set_index("zip_code")
    assert out.loc["70001", "total_inspected"] == 8
    assert out.loc["70002", "total_inspected"] == 5
    assert out.loc["70002", "state"] == "LA"


def test_merge_housing_assistance_zero_pads_zip_codes():
    out = merge_housing_assistance(pd.DataFrame([_owners(zipCode=1234)]),
                                   pd.DataFrame([_renters(zipCode=1234)]))
    assert out["zip_code"].tolist() == ["01234"]


# ---------------------------------------------------------------------------
# Tract -> ZIP crosswalk
# ---------------------------------------------------------------------------
def test_tract_to_zip_splits_one_tract_across_zips():
    df = pd.DataFrame({"tract": ["12019000100"], "val": [40.0]})
    cw = pd.DataFrame({"TRACT": ["12019000100", "12019000100"],
                       "ZIP": ["32003", "32004"], "RES_RATIO": [0.25, 0.75]})
    out = tract_to_zip(df, cw, tract_col="tract", value_cols=["val"]).set_index("zip_code")
    # Each ZIP receives only this tract, so its weighted average is the tract value.
    assert out.loc["32003", "val"] == pytest.approx(40.0)
    assert out.loc["32004", "val"] == pytest.approx(40.0)


def test_tract_to_zip_zero_pads_keys():
    df = pd.DataFrame({"tract": [1019000100], "val": [5.0]})
    cw = pd.DataFrame({"TRACT": ["01019000100"], "ZIP": [1234], "RES_RATIO": [1.0]})
    out = tract_to_zip(df, cw, tract_col="tract", value_cols=["val"])
    assert out["zip_code"].tolist() == ["01234"]


def test_tract_to_zip_raises_when_no_tract_matches():
    df = pd.DataFrame({"tract": ["99999999999"], "val": [1.0]})
    cw = pd.DataFrame({"TRACT": ["12019000100"], "ZIP": ["32003"], "RES_RATIO": [1.0]})
    with pytest.raises(ValueError, match="0 rows"):
        tract_to_zip(df, cw, tract_col="tract", value_cols=["val"])


@pytest.mark.xfail(strict=True, reason=(
    "Known issue: a NaN tract value is summed as 0 but its weight stays in the "
    "denominator, pulling the ZIP average toward 0. Left as-is because fixing it "
    "changes the frozen ABT behind the published results."))
def test_tract_to_zip_ignores_missing_tract_values():
    df = pd.DataFrame({"tract": ["12019000100", "12019000200"], "val": [100.0, np.nan]})
    cw = pd.DataFrame({"TRACT": ["12019000100", "12019000200"],
                       "ZIP": ["32003", "32003"], "RES_RATIO": [0.5, 0.5]})
    out = tract_to_zip(df, cw, tract_col="tract", value_cols=["val"])
    assert out["val"].iloc[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Geodesic distance to the storm track
# ---------------------------------------------------------------------------
_GEOD = Geod(ellps="WGS84")


def test_distance_to_track_matches_reference_geodesic():
    # One degree of longitude along the equator on WGS84 is 111.32 km.
    track = LineString([(0.0, -5.0), (0.0, 5.0)])
    assert distance_to_track_km(Point(1.0, 0.0), track) == pytest.approx(111.32, abs=0.01)


def test_distance_to_track_is_geodesic_not_degrees():
    # At 60N a degree of longitude is about half as long as at the equator.
    track = LineString([(0.0, 55.0), (0.0, 65.0)])
    d = distance_to_track_km(Point(1.0, 60.0), track)
    _, _, ref = _GEOD.inv(1.0, 60.0, 0.0, 60.0)
    assert d == pytest.approx(ref / 1000, rel=1e-3)
    assert 55 < d < 57


def test_distance_to_track_zero_on_track_and_nan_without_track():
    track = LineString([(-80.0, 25.0), (-80.0, 30.0)])
    assert distance_to_track_km(Point(-80.0, 27.0), track) == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(distance_to_track_km(Point(-80.0, 27.0), None))


def test_build_track_linestring_orders_by_time_and_filters_storm():
    ib = pd.DataFrame({
        "NAME": ["IAN", "IAN", "IAN", "IAN", "IDA"],
        "SEASON": [2022, 2022, 2022, 2021, 2021],
        "ISO_TIME": ["2022-09-28 06:00", "2022-09-28 00:00", "2022-09-28 12:00",
                     "2021-01-01 00:00", "2021-08-29 00:00"],
        "LAT": [26.0, 25.0, np.nan, 10.0, 29.0],
        "LON": [-82.0, -83.0, -81.0, -50.0, -90.0],
    })
    line = build_track_linestring(ib, "Ian", 2022)
    assert list(line.coords) == [(-83.0, 25.0), (-82.0, 26.0)]
    assert build_track_linestring(ib, "Ida", 2021) is None  # a single point is not a track


def test_compute_distance_to_track_uses_projected_centroids():
    square = Polygon([(-80.1, 26.9), (-79.9, 26.9), (-79.9, 27.1), (-80.1, 27.1)])
    zcta = gpd.GeoDataFrame({"ZCTA5CE20": [33401]}, geometry=[square], crs="EPSG:4326")
    track = LineString([(-81.0, 20.0), (-81.0, 35.0)])
    out = compute_distance_to_track(zcta, track)
    assert out["zip_code"].tolist() == ["33401"]
    _, _, ref = _GEOD.inv(-80.0, 27.0, -81.0, 27.0)
    assert out["distance_to_track_km"].iloc[0] == pytest.approx(ref / 1000, rel=5e-3)


# ---------------------------------------------------------------------------
# SNAP retailers
# ---------------------------------------------------------------------------
def test_snap_retailers_per_zip_counts_and_supermarket_distance():
    a = Polygon([(-80.0, 26.0), (-79.9, 26.0), (-79.9, 26.1), (-80.0, 26.1)])
    b = Polygon([(-79.5, 26.0), (-79.4, 26.0), (-79.4, 26.1), (-79.5, 26.1)])
    zcta = gpd.GeoDataFrame({"ZCTA5CE20": ["33001", "33002"]}, geometry=[a, b],
                            crs="EPSG:4326")
    retailers = pd.DataFrame({
        "Latitude": [26.05, 26.06, 26.05],
        "Longitude": [-79.95, -79.94, -79.45],
        "Store_Type": ["Supermarket", "Convenience Store", "Convenience Store"],
    })
    out = snap_retailers_per_zip(retailers, zcta).set_index("zip_code")
    assert out.loc["33001", "snap_retailer_count"] == 2
    assert out.loc["33002", "snap_retailer_count"] == 1
    # The supermarket lies inside 33001; 33002 is roughly 40 km (25 mi) east.
    assert out.loc["33001", "dist_nearest_supermarket_mi"] == pytest.approx(0.0)
    assert 20 < out.loc["33002", "dist_nearest_supermarket_mi"] < 30
