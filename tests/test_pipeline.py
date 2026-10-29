"""End-to-end test of src.pipeline on a miniature raw-data folder (one storm, four ZIPs)."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from config import HURRICANE_META
from src import pipeline

IDA = next(h for h in HURRICANE_META if h["name"] == "Ida")
DN = IDA["disaster_number"]
# Three Louisiana ZIPs along the track and one New York ZIP keyed to an Ida record.
ZIPS = ["70112", "70113", "70114", "10466"]
TRACTS = ["22071000100", "22071000200", "22071000300", "36005000100"]


def _ha(kind):
    rows = []
    for z, insp in zip(ZIPS, [400, 60, 5, 1]):
        row = {"disasterNumber": DN, "zipCode": z, "county": "Orleans", "state": "LA",
               "validRegistrations": insp * 2, "totalInspected": insp,
               "totalApprovedIhpAmount": insp * 100.0}
        if kind == "owners":
            row.update(noFemaInspectedDamage=1, femaInspectedDamageGreaterThan30000=1)
        else:
            row.update(totalInspectedWithNoDamage=1, totalWithMajorDamage=1)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "raw"
    (raw / "snap_retailers").mkdir(parents=True)
    _ha("owners").to_csv(raw / f"fema_housing_owners_{DN}.csv", index=False)
    _ha("renters").to_csv(raw / f"fema_housing_renters_{DN}.csv", index=False)
    pd.DataFrame({
        "zip_code": ZIPS, "population": [10000, 8000, 1000, 50000],
        "median_income": [40000, -666666666, 60000, 50000],
        "poverty_count": [2000, 1600, 100, 5000], "renters": [3000, 2000, 300, 9000],
        "white_alone": [3000, 4000, 800, 20000], "no_vehicle_households": [500, 400, 50, 9000],
        "mobile_homes": [100, 80, 1500, 0],  # 70114: 1,500 homes for 1,000 people
    }).to_csv(raw / "census_acs5_zcta.csv", index=False)
    pd.DataFrame({"TRACT": TRACTS, "ZIP": ZIPS, "RES_RATIO": 1.0}).to_excel(
        raw / "hud_tract_zip.xlsx", index=False)
    pd.DataFrame({"FIPS": TRACTS, "RPL_THEME1": [0.9, 0.5, 0.2, 0.7],
                  "RPL_THEME2": 0.5, "RPL_THEME3": 0.5, "RPL_THEME4": 0.5,
                  "RPL_THEMES": [0.9, -999, 0.2, 0.7]}).to_csv(raw / "cdc_svi_2022.csv", index=False)
    with pd.ExcelWriter(raw / "food_access_atlas.xlsx") as xw:
        pd.DataFrame({"CensusTract": TRACTS, "LILATracts_1And10": [1, 0, 0, 0],
                      "lalowi1": 10.0, "lahunv1share": 0.1,
                      "TractSNAP": [800, np.nan, 20, 3000]}).to_excel(
            xw, sheet_name="Food Access Research Atlas", index=False)
    boxes = [box(-90.10, 29.90, -90.05, 29.95), box(-90.05, 29.90, -90.00, 29.95),
             box(-89.70, 29.90, -89.65, 29.95), box(-73.90, 40.85, -73.85, 40.90)]
    (raw / "zcta").mkdir()
    gpd.GeoDataFrame({"ZCTA5CE20": ZIPS}, geometry=boxes, crs="EPSG:4326").to_file(
        raw / "zcta" / "zcta.shp")
    pd.DataFrame({"Latitude": [29.92, 29.93], "Longitude": [-90.07, -90.02],
                  "Store_Type": ["Supermarket", "Convenience Store"]}).to_csv(
        raw / "snap_retailers" / "retailers.csv", index=False)
    track = pd.DataFrame({"NAME": "IDA", "SEASON": 2021,
                          "ISO_TIME": ["2021-08-29 12:00", "2021-08-29 18:00", "2021-08-30 00:00"],
                          "LAT": [29.0, 29.9, 30.8], "LON": [-90.1, -90.1, -90.1]})
    units = pd.DataFrame([{c: "" for c in track.columns}])  # IBTrACS second header row
    pd.concat([units, track]).to_csv(raw / "ibtracs_na.csv", index=False)
    pd.DataFrame({"TRACTFIPS": TRACTS, "CFLD_RISKS": [80.0, 60.0, 10.0, 5.0],
                  "HRCN_RISKS": [90.0, 85.0, 70.0, 20.0]}).to_csv(raw / "fema_nri_tract.csv",
                                                                   index=False)
    return tmp_path


def test_pipeline_builds_a_valid_abt(raw):
    paths = pipeline.default_paths(raw)
    pipeline.run(["fuse", "features"], paths, meta=[IDA])

    fused = pd.read_csv(paths["fused"], dtype={"zip_code": str})
    assert len(fused) == 4
    abt = pd.read_csv(paths["abt"], dtype={"zip_code": str}).set_index("zip_code")
    dropped = pd.read_csv(paths["dropped"], dtype={"zip_code": str})

    # The New York ZIP on a Louisiana record is set aside, not scored.
    assert sorted(abt.index) == ["70112", "70113", "70114"]
    assert dropped["zip_code"].tolist() == ["10466"]
    # Targets: 400 owner + 400 renter inspections per 10,000 residents.
    assert abt.loc["70112", "verified_damage_per_1000"] == pytest.approx(80.0)
    assert (abt["train_test_split"] == "TEST").all()
    # Census code cleared and imputed from the other Louisiana ZIPs.
    assert abt.loc["70113", "median_income"] == pytest.approx(50000.0)
    # Shares capped at 100%; SVI -999 imputed; SNAP participation left for the model.
    assert abt.loc["70114", "pct_mobile_homes"] == 100.0
    assert 0 <= abt.loc["70113", "svi_overall"] <= 1
    assert np.isnan(abt.loc["70113", "snap_participation_pct"])
    assert abt.loc["70113", "food_data_missing"] == 1
    # Spatial joins and track geometry.
    assert abt.loc["70112", "snap_retailer_count"] == 1
    assert abt.loc["70112", "food_desert_flag"] == 1
    assert abt.loc["70112", "distance_to_track_km"] < abt.loc["70114", "distance_to_track_km"]
    assert abt.loc["70112", "nri_cflood_score"] == 80.0


def test_pipeline_skips_finished_stages_unless_forced(raw, monkeypatch):
    paths = pipeline.default_paths(raw)
    pipeline.run(["fuse", "features"], paths, meta=[IDA])
    calls = []
    monkeypatch.setattr(pipeline, "fuse", lambda *a, **k: calls.append("fuse"))
    pipeline.run(["fuse", "features"], paths, meta=[IDA])
    assert calls == []
    with pytest.raises(Exception):  # forced: reaches the (stubbed) fuse
        pipeline.run(["fuse"], paths, meta=[IDA], force=True)
    assert calls == ["fuse"]


def test_pipeline_refuses_to_write_an_invalid_abt(raw, monkeypatch):
    paths = pipeline.default_paths(raw)
    pipeline.run(["fuse"], paths, meta=[IDA])
    fused = pd.read_csv(paths["fused"], dtype={"zip_code": str})
    dup = fused[fused["zip_code"] == "70112"]
    pd.concat([fused, dup]).to_csv(paths["fused"], index=False)  # duplicate key
    with pytest.raises(ValueError, match="unique_key"):
        pipeline.run(["features"], paths, meta=[IDA], force=True)
    assert not paths["abt"].exists()


def test_cli_reports_missing_inputs(tmp_path, monkeypatch):
    real = pipeline.default_paths
    monkeypatch.setattr(pipeline, "default_paths", lambda root=None: real(tmp_path))
    assert pipeline.main(["--from", "fuse", "--to", "fuse"]) == 2
