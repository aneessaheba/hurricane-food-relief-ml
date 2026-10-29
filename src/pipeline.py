"""
End-to-end data pipeline: raw downloads -> fused table -> validated ABT.

    python -m src.pipeline                      # every stage, skipping finished ones
    python -m src.pipeline --from features      # rebuild the ABT from the fused table
    python -m src.pipeline --force              # recompute every stage
    python -m src.pipeline --to fuse            # stop after the fused table

Stages (each writes one artifact; a stage is skipped when its artifact exists
unless --force is given or an earlier stage reran):

  download   data/raw/*                         src.data_acquisition.download_all
  fuse       data/interim/fused_zip_hurricane.csv
             FEMA owners+renters per (disaster, ZIP), joined with ACS shares,
             SVI / Food Atlas / NRI aggregated from tracts, SNAP retailers,
             geodesic distance to each storm track and the NFHL overlay
  features   data/processed/abt_clean.csv (+ abt_dropped_rows.csv)
             splits, targets, severity classes, imputation, finalize_abt,
             then src.validation; refuses to write an ABT with errors

The paper's results come from data/processed/abt.csv, the ABT as published
(version 1, in the share bundle). It predates the fixes in finalize_abt and
is kept frozen; abt_clean.csv is the corrected table and feeds the paper's
"cleaned data" robustness check (scripts/camera_ready_experiments.py cleaned).

This replaces running notebooks 01, 03 and 04 by hand; the notebooks remain
for exploration. Logs go to stderr and data/interim/pipeline.log.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_PATHS, HURRICANE_META, TARGET_COL  # noqa: E402

log = logging.getLogger(__name__)

STAGES = ["download", "fuse", "features"]
NFHL_STATE_FIPS = ["48", "22", "12", "37", "45", "13", "01", "28"]


def default_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    """Artifact locations; `root` redirects everything (used by tests)."""
    if root is None:
        raw, interim, processed = (DATA_PATHS[k] for k in ("raw", "interim", "processed"))
    else:
        root = Path(root)
        raw, interim, processed = root / "raw", root / "interim", root / "processed"
    return {
        "raw": raw,
        "fused": interim / "fused_zip_hurricane.csv",
        "abt": processed / "abt_clean.csv",
        "dropped": processed / "abt_dropped_rows.csv",
        "log": interim / "pipeline.log",
    }


# -----------------------------------------------------------------------------
# fuse: one loader per source, each returning a ZIP-keyed frame
# -----------------------------------------------------------------------------
def load_housing_assistance(raw: Path, meta: Iterable[dict]) -> pd.DataFrame:
    """FEMA Owners + Renters summed per (disaster, ZIP) for every storm."""
    from src.data_fusion import merge_housing_assistance
    frames = []
    for h in meta:
        dn = h["disaster_number"]
        o = pd.read_csv(raw / f"fema_housing_owners_{dn}.csv")
        r = pd.read_csv(raw / f"fema_housing_renters_{dn}.csv")
        frames.append(merge_housing_assistance(o, r))
    ha = pd.concat(frames, ignore_index=True)
    log.info("housing assistance: %s zip x hurricane rows", f"{len(ha):,}")
    return ha


def load_acs(raw: Path) -> pd.DataFrame:
    """ACS 5-year ZCTA counts converted to demographic shares."""
    from src.feature_engineering import derive_demographic_shares
    path = raw / "census_acs5_zcta.csv"
    if not path.exists():  # same 2021 vintage, as written by fetch_camera_ready_data.py
        path = raw / "acs5_2021_zcta.csv"
    acs = pd.read_csv(path, dtype={"zip_code": str})
    acs["zip_code"] = acs["zip_code"].str.zfill(5)
    return derive_demographic_shares(acs)


def load_crosswalk(raw: Path) -> pd.DataFrame:
    return pd.read_excel(raw / "hud_tract_zip.xlsx")


def load_svi_zip(raw: Path, cw: pd.DataFrame) -> pd.DataFrame:
    """CDC SVI 2022 theme percentiles, tract -> ZIP."""
    from src.data_fusion import tract_to_zip
    svi = pd.read_csv(raw / "cdc_svi_2022.csv", low_memory=False).replace(-999, np.nan)
    keep = ["FIPS", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4", "RPL_THEMES"]
    out = tract_to_zip(svi[keep], cw, tract_col="FIPS", value_cols=keep[1:])
    return out.rename(columns={
        "RPL_THEME1": "svi_socioeconomic", "RPL_THEME2": "svi_household_comp",
        "RPL_THEME3": "svi_minority_lang", "RPL_THEME4": "svi_housing_transport",
        "RPL_THEMES": "svi_overall"})


def load_food_atlas_zip(raw: Path, cw: pd.DataFrame) -> pd.DataFrame:
    """USDA Food Access Research Atlas, tract -> ZIP, plus the food-desert flag."""
    from src.data_fusion import tract_to_zip
    fa = pd.read_excel(raw / "food_access_atlas.xlsx", sheet_name="Food Access Research Atlas")
    keep = ["CensusTract", "LILATracts_1And10", "lalowi1", "lahunv1share", "TractSNAP"]
    out = tract_to_zip(fa[keep], cw, tract_col="CensusTract", value_cols=keep[1:])
    out["food_desert_flag"] = (out["LILATracts_1And10"] >= 0.5).astype(int)
    return out.rename(columns={"TractSNAP": "snap_participation_pct"})


def load_zcta(raw: Path):
    import geopandas as gpd
    return gpd.read_file(next((raw / "zcta").rglob("*.shp")))


def load_snap_zip(raw: Path, zcta) -> pd.DataFrame:
    """SNAP retailer counts and nearest-supermarket distance per ZIP."""
    from src.data_fusion import snap_retailers_per_zip
    snap_csv = next((raw / "snap_retailers").rglob("*.csv"))
    return snap_retailers_per_zip(pd.read_csv(snap_csv), zcta)


def load_track_distances(raw: Path, zcta, meta: Iterable[dict]) -> pd.DataFrame:
    """Geodesic distance from every ZCTA to each storm's IBTrACS track."""
    from src.data_fusion import build_track_linestring, compute_distance_to_track
    ibt = pd.read_csv(raw / "ibtracs_na.csv", low_memory=False, skiprows=[1])
    rows = []
    for h in meta:
        track = build_track_linestring(ibt, h["name"], h["year"])
        if track is None:
            log.warning("no IBTrACS track for %s %s", h["name"], h["year"])
        d = compute_distance_to_track(zcta, track)
        d["disaster_number"] = h["disaster_number"]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_floodplain(raw: Path, zcta) -> pd.DataFrame:
    """NFHL SFHA overlay; sparse (state files are optional)."""
    import geopandas as gpd
    from src.data_fusion import pct_in_floodplain
    rows = [pct_in_floodplain(zcta, gpd.read_file(p))
            for p in (raw / f"nfhl_sfha_{s}.geojson" for s in NFHL_STATE_FIPS) if p.exists()]
    if not rows:
        log.warning("no NFHL state files; pct_in_100yr_floodplain will be empty")
        return pd.DataFrame({"zip_code": [], "pct_in_100yr_floodplain": []})
    flood = pd.concat(rows, ignore_index=True)
    return flood.groupby("zip_code")["pct_in_100yr_floodplain"].max().reset_index()


def load_nri_zip(raw: Path, cw: pd.DataFrame) -> pd.DataFrame:
    """FEMA National Risk Index coastal-flood and hurricane scores, tract -> ZIP."""
    from src.data_fusion import tract_to_zip
    path = raw / "fema_nri_tract.csv"
    if not path.exists():
        log.warning("fema_nri_tract.csv missing; NRI scores will be empty")
        return pd.DataFrame({"zip_code": [], "nri_cflood_score": [], "nri_hrcn_score": []})
    nri = pd.read_csv(path, dtype={"TRACTFIPS": str}, low_memory=False)
    cols = {k: v for k, v in {"CFLD_RISKS": "nri_cflood_score",
                              "HRCN_RISKS": "nri_hrcn_score"}.items() if k in nri.columns}
    if not cols:  # some NRI releases only carry the composite score
        cols = {c: f"nri_{c.lower()}" for c in ("RISK_SCORE",) if c in nri.columns}
    nri_t = nri[["TRACTFIPS", *cols]].rename(columns=cols)
    return tract_to_zip(nri_t, cw, tract_col="TRACTFIPS", value_cols=list(cols.values()))


def fuse(raw: Path, meta: Iterable[dict] = HURRICANE_META) -> pd.DataFrame:
    """Join every source onto the FEMA (disaster, ZIP) rows."""
    meta = list(meta)
    ha = load_housing_assistance(raw, meta)
    cw = load_crosswalk(raw)
    zcta = load_zcta(raw)
    fused = (ha
             .merge(load_acs(raw), on="zip_code", how="left")
             .merge(load_svi_zip(raw, cw), on="zip_code", how="left")
             .merge(load_food_atlas_zip(raw, cw), on="zip_code", how="left")
             .merge(load_snap_zip(raw, zcta), on="zip_code", how="left")
             .merge(load_track_distances(raw, zcta, meta),
                    on=["zip_code", "disaster_number"], how="left")
             .merge(load_floodplain(raw, zcta), on="zip_code", how="left")
             .merge(load_nri_zip(raw, cw), on="zip_code", how="left"))
    fused["snap_retailers_per_1k"] = (
        fused["snap_retailer_count"] / (fused["population"].replace(0, np.nan) / 1000))
    if len(fused) != len(ha):
        raise ValueError(f"a join duplicated rows: {len(ha):,} -> {len(fused):,}")
    return fused


# -----------------------------------------------------------------------------
# features: fused table -> ABT
# -----------------------------------------------------------------------------
def build_abt(fused: pd.DataFrame):
    """Splits, targets, severity classes, imputation and final clean-up."""
    from src.data_fusion import drop_mismatched_zips
    from src.feature_engineering import (
        assign_split, bin_severity, compute_targets, finalize_abt, impute_missing,
    )
    df = fused.copy()
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    n = len(df)
    df = df[df["population"].notna() & (df["population"] > 0)].copy()
    log.info("dropped %d rows with no Census population match", n - len(df))
    # Set mis-keyed ZIPs aside before imputing so they do not shift state medians.
    df, mismatched = drop_mismatched_zips(df)
    df = assign_split(df)
    df = compute_targets(df)
    df["damage_severity_class"] = bin_severity(df[TARGET_COL])
    df = impute_missing(df)
    abt, _ = finalize_abt(df)  # ZIPs already checked; clears codes, caps shares
    return abt, assign_split(mismatched)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(stages: List[str], paths: Dict[str, Path], force: bool = False,
        meta: Iterable[dict] = HURRICANE_META, census_key: Optional[str] = None) -> None:
    from src.validation import assert_valid

    rerun = force
    if "download" in stages:
        from src.data_acquisition import download_all
        download_all(census_key=census_key, force=force, log_file=paths["log"])

    if "fuse" in stages:
        if paths["fused"].exists() and not rerun:
            log.info("[skip] fuse: %s exists", paths["fused"])
        else:
            fused = fuse(paths["raw"], meta)
            paths["fused"].parent.mkdir(parents=True, exist_ok=True)
            fused.to_csv(paths["fused"], index=False)
            log.info("wrote %s %s", paths["fused"], fused.shape)
            rerun = True

    if "features" in stages:
        if paths["abt"].exists() and not rerun:
            log.info("[skip] features: %s exists", paths["abt"])
        else:
            fused = pd.read_csv(paths["fused"], dtype={"zip_code": str})
            abt, dropped = build_abt(fused)
            for issue in assert_valid(abt):  # raises before anything is written
                log.warning("%s", issue)
            paths["abt"].parent.mkdir(parents=True, exist_ok=True)
            abt.to_csv(paths["abt"], index=False)
            dropped.to_csv(paths["dropped"], index=False)
            log.info("wrote %s %s (%d rows set aside in %s)",
                     paths["abt"], abt.shape, len(dropped), paths["dropped"].name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the ABT from raw sources.")
    ap.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    ap.add_argument("--to", dest="end", choices=STAGES, default=STAGES[-1])
    ap.add_argument("--force", action="store_true", help="recompute finished stages")
    ap.add_argument("--census-key", default=None)
    args = ap.parse_args(argv)
    stages = STAGES[STAGES.index(args.start):STAGES.index(args.end) + 1]
    if not stages:
        ap.error("--from must not come after --to")

    paths = default_paths()
    from src.data_acquisition import setup_logging
    setup_logging(log_file=paths["log"])
    try:
        run(stages, paths, force=args.force or args.start != STAGES[0],
            census_key=args.census_key)
    except (FileNotFoundError, StopIteration) as e:
        log.error("missing input: %s — run the download stage or see README", e)
        return 2
    except ValueError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
