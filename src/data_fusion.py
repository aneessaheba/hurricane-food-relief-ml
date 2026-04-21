"""
Data fusion: join raw data sources into a single zip-code × hurricane table.

Key operations:
  * tract-to-zip aggregation using HUD RES_RATIO weighted averaging
  * SNAP retailer point-in-polygon spatial join
  * geodesic zip-centroid-to-storm-track distance via pyproj.Geod (never Euclidean)
  * flood-plain overlay in EPSG:5070 (equal-area) processed county-by-county
  * Owners + Renters SUMMED per (disasterNumber, zipCode) via outer join
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from tqdm import tqdm

from config import CRS_LATLON, CRS_AREA, SFHA_ZONES

_GEOD = Geod(ellps="WGS84")


# -----------------------------------------------------------------------------
# Tract -> ZIP weighted aggregation (HUD crosswalk)
# -----------------------------------------------------------------------------
def tract_to_zip(
    df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    tract_col: str,
    value_cols: Iterable[str],
    zip_col_out: str = "zip_code",
) -> pd.DataFrame:
    """
    Aggregate tract-level `df` up to ZIP using RES_RATIO-weighted averaging.

    Parameters
    ----------
    df : DataFrame keyed on `tract_col` (11-digit GEOID string).
    crosswalk : HUD TRACT_ZIP crosswalk with columns TRACT, ZIP, RES_RATIO.
    tract_col : the column in df that matches crosswalk.TRACT.
    value_cols : columns to weighted-average.
    """
    cw = crosswalk.rename(columns=str.upper)[["TRACT", "ZIP", "RES_RATIO"]].copy()
    cw["TRACT"] = cw["TRACT"].astype(str).str.zfill(11)
    cw["ZIP"] = cw["ZIP"].astype(str).str.zfill(5)
    df = df.copy()
    df[tract_col] = df[tract_col].astype(str).str.zfill(11)

    merged = cw.merge(df, left_on="TRACT", right_on=tract_col, how="inner")
    if merged.empty:
        raise ValueError("tract_to_zip: crosswalk join produced 0 rows — check tract keys.")

    def _wavg(g: pd.DataFrame) -> pd.Series:
        w = g["RES_RATIO"].to_numpy()
        wsum = w.sum() or 1.0
        return pd.Series({c: np.nansum(g[c].to_numpy() * w) / wsum for c in value_cols})

    out = merged.groupby("ZIP").apply(_wavg).reset_index()
    out = out.rename(columns={"ZIP": zip_col_out})
    return out


# -----------------------------------------------------------------------------
# SNAP retailer spatial join
# -----------------------------------------------------------------------------
def snap_retailers_per_zip(
    retailers: pd.DataFrame, zcta_gdf: gpd.GeoDataFrame,
    lat_col: str = "Latitude", lon_col: str = "Longitude",
    zip_col: str = "ZCTA5CE20",
) -> pd.DataFrame:
    """
    Spatial-join SNAP retailers to ZCTA polygons. Both reprojected to EPSG:4326
    for point-in-polygon. Returns DataFrame with zip_code + snap_retailer_count
    + dist_nearest_supermarket_mi.
    """
    pts = gpd.GeoDataFrame(
        retailers.copy(),
        geometry=gpd.points_from_xy(retailers[lon_col], retailers[lat_col]),
        crs=CRS_LATLON,
    )
    zcta = zcta_gdf.to_crs(CRS_LATLON)
    joined = gpd.sjoin(pts, zcta[[zip_col, "geometry"]],
                       predicate="within", how="left")
    counts = (joined.groupby(zip_col).size()
              .rename("snap_retailer_count").reset_index())

    # Nearest supermarket distance: project everything to EPSG:5070 for meters
    zcta_m = zcta_gdf.to_crs(CRS_AREA)
    store_type_col = next((c for c in pts.columns
                           if c.lower().replace(" ", "_") == "store_type"), None)
    if store_type_col is not None:
        super_pts = pts[pts[store_type_col].astype(str)
                        .str.contains("Super", case=False, na=False)]
    else:
        super_pts = pts.iloc[0:0]
    if not super_pts.empty:
        super_m = super_pts.to_crs(CRS_AREA)
        sjoin_nn = gpd.sjoin_nearest(
            zcta_m[[zip_col, "geometry"]], super_m[["geometry"]],
            distance_col="dist_m",
        )
        dist = (sjoin_nn.groupby(zip_col)["dist_m"].min()
                .rename("dist_nearest_supermarket_mi").reset_index())
        dist["dist_nearest_supermarket_mi"] = dist["dist_nearest_supermarket_mi"] / 1609.344
    else:
        dist = pd.DataFrame({zip_col: [], "dist_nearest_supermarket_mi": []})

    out = counts.merge(dist, on=zip_col, how="outer")
    out = out.rename(columns={zip_col: "zip_code"})
    out["zip_code"] = out["zip_code"].astype(str).str.zfill(5)
    out["snap_retailer_count"] = out["snap_retailer_count"].fillna(0).astype(int)
    return out


# -----------------------------------------------------------------------------
# Distance from zip centroid to storm track (GEODESIC)
# -----------------------------------------------------------------------------
def build_track_linestring(ibtracs: pd.DataFrame,
                           name: str, season: int) -> Optional[LineString]:
    """Extract IBTrACS track for one storm as a shapely LineString (lon, lat)."""
    sub = ibtracs[(ibtracs["NAME"].str.upper() == name.upper())
                  & (ibtracs["SEASON"] == season)].sort_values("ISO_TIME")
    sub = sub.dropna(subset=["LAT", "LON"])
    if len(sub) < 2:
        return None
    return LineString(zip(sub["LON"].astype(float), sub["LAT"].astype(float)))


def distance_to_track_km(centroid: Point, track: LineString) -> float:
    """
    Return the geodesic distance in km from a zip centroid to the nearest
    point on the storm track, using pyproj.Geod.inv. Never use Euclidean.
    """
    if track is None:
        return np.nan
    _, nearest = nearest_points(centroid, track)
    _az12, _az21, meters = _GEOD.inv(
        centroid.x, centroid.y, nearest.x, nearest.y
    )
    return meters / 1000.0


def compute_distance_to_track(
    zcta_gdf: gpd.GeoDataFrame, track: LineString,
    zip_col: str = "ZCTA5CE20",
) -> pd.DataFrame:
    """Vectorized-ish per-zip distance to one storm's track."""
    cents = zcta_gdf.to_crs(CRS_LATLON).geometry.centroid
    zips = zcta_gdf[zip_col].astype(str).str.zfill(5).to_numpy()
    out = []
    for z, c in tqdm(list(zip(zips, cents)), desc="dist_to_track"):
        out.append((z, distance_to_track_km(c, track)))
    return pd.DataFrame(out, columns=["zip_code", "distance_to_track_km"])


# -----------------------------------------------------------------------------
# NFHL SFHA overlay -> pct_in_100yr_floodplain
# -----------------------------------------------------------------------------
def pct_in_floodplain(
    zcta_gdf: gpd.GeoDataFrame, nfhl_gdf: gpd.GeoDataFrame,
    zip_col: str = "ZCTA5CE20", county_col: Optional[str] = "COUNTY_FIPS",
) -> pd.DataFrame:
    """
    Compute share of each ZIP's area falling inside SFHA zones. Both inputs
    reprojected to EPSG:5070. Processed county-by-county if `county_col` is
    present in `nfhl_gdf` to keep memory under control.
    """
    zcta = zcta_gdf.to_crs(CRS_AREA).copy()
    nfhl = nfhl_gdf.to_crs(CRS_AREA).copy()
    nfhl = nfhl[nfhl["FLD_ZONE"].isin(SFHA_ZONES)]
    zcta["zip_area_m2"] = zcta.geometry.area

    results = []
    county_iter = (nfhl[county_col].dropna().unique()
                   if county_col and county_col in nfhl.columns else [None])
    for cfips in tqdm(county_iter, desc="flood overlay by county"):
        chunk = nfhl[nfhl[county_col] == cfips] if cfips is not None else nfhl
        if chunk.empty:
            continue
        try:
            inter = gpd.overlay(
                zcta[[zip_col, "geometry", "zip_area_m2"]],
                chunk[["geometry"]],
                how="intersection",
                keep_geom_type=False,
            )
        except Exception as e:
            print(f"[warn] overlay failed county={cfips}: {e}")
            continue
        if inter.empty:
            continue
        inter["inter_m2"] = inter.geometry.area
        agg = (inter.groupby(zip_col)
               .agg(inter_m2=("inter_m2", "sum"),
                    zip_area_m2=("zip_area_m2", "first"))
               .reset_index())
        results.append(agg)

    if not results:
        return pd.DataFrame({"zip_code": [], "pct_in_100yr_floodplain": []})

    merged = pd.concat(results, ignore_index=True)
    final = (merged.groupby(zip_col)
             .agg(inter_m2=("inter_m2", "sum"),
                  zip_area_m2=("zip_area_m2", "first"))
             .reset_index())
    final["pct_in_100yr_floodplain"] = (
        final["inter_m2"] / final["zip_area_m2"] * 100.0
    ).clip(0, 100)
    final = final.rename(columns={zip_col: "zip_code"})[
        ["zip_code", "pct_in_100yr_floodplain"]
    ]
    final["zip_code"] = final["zip_code"].astype(str).str.zfill(5)
    return final


# -----------------------------------------------------------------------------
# Housing Assistance target assembly (OWNERS + RENTERS SUMMED, not averaged)
# -----------------------------------------------------------------------------
# The actual v2 schemas are ASYMMETRIC:
#   OWNERS uses $ damage-amount buckets:
#     noFemaInspectedDamage, femaInspectedDamageBetween1And10000,
#     femaInspectedDamageBetween10001And20000,
#     femaInspectedDamageBetween20001And30000,
#     femaInspectedDamageGreaterThan30000
#   RENTERS uses category labels (no minor category at all):
#     totalInspectedWithNoDamage, totalWithModerateDamage,
#     totalWithMajorDamage, totalWithSubstantialDamage
# We reconcile them by treating owner >$20k damage and renter major+substantial
# as "major_substantial" (the most severe bucket).

_OWNER_SEVERITY = {
    "no_damage":         "noFemaInspectedDamage",
    "minor":             "femaInspectedDamageBetween1And10000",
    "moderate":          "femaInspectedDamageBetween10001And20000",
    "major":             "femaInspectedDamageBetween20001And30000",
    "substantial":       "femaInspectedDamageGreaterThan30000",
}
_RENTER_SEVERITY = {
    "no_damage":   "totalInspectedWithNoDamage",
    "moderate":    "totalWithModerateDamage",
    "major":       "totalWithMajorDamage",
    "substantial": "totalWithSubstantialDamage",
}
_SHARED_COUNT_COLS = [
    "validRegistrations", "totalInspected",
    "approvedForFemaAssistance",
    "totalApprovedIhpAmount", "repairReplaceAmount",
    "rentalAmount", "otherNeedsAmount",
]


def merge_housing_assistance(
    owners: pd.DataFrame, renters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sum Owners + Renters per (disasterNumber, zipCode). Owners and Renters are
    DIFFERENT populations (homeowners vs. renters), so counts are SUMMED, not
    averaged. Handles the asymmetric v2 schema (owner $ buckets vs renter
    category labels).
    """
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["zipCode"] = df["zipCode"].astype(str).str.zfill(5)
        keep = (["disasterNumber", "zipCode", "county", "state"]
                + _SHARED_COUNT_COLS
                + list(_OWNER_SEVERITY.values())
                + list(_RENTER_SEVERITY.values()))
        return df[[c for c in keep if c in df.columns]]

    o = _clean(owners)
    r = _clean(renters)
    m = o.merge(
        r, on=["disasterNumber", "zipCode"], how="outer",
        suffixes=("_owners", "_renters"),
    )

    def _sum(col: str) -> pd.Series:
        a, b = f"{col}_owners", f"{col}_renters"
        if a in m and b in m:
            return m[a].fillna(0) + m[b].fillna(0)
        if a in m: return m[a].fillna(0)
        if b in m: return m[b].fillna(0)
        return pd.Series(0.0, index=m.index)

    # Shared numeric columns
    for c in _SHARED_COUNT_COLS:
        m[c] = _sum(c)

    # Severity rollup: sum each bucket that exists in either table
    owner_major_sub = (m.get(_OWNER_SEVERITY["major"], 0).fillna(0)
                       + m.get(_OWNER_SEVERITY["substantial"], 0).fillna(0))
    renter_major_sub = (m.get(_RENTER_SEVERITY["major"], 0).fillna(0)
                        + m.get(_RENTER_SEVERITY["substantial"], 0).fillna(0))
    m["total_major_substantial"] = owner_major_sub + renter_major_sub

    owner_no = m.get(_OWNER_SEVERITY["no_damage"], 0).fillna(0)
    renter_no = m.get(_RENTER_SEVERITY["no_damage"], 0).fillna(0)
    m["total_no_damage"] = owner_no + renter_no

    owner_moderate = m.get(_OWNER_SEVERITY["moderate"], 0).fillna(0)
    renter_moderate = m.get(_RENTER_SEVERITY["moderate"], 0).fillna(0)
    m["total_moderate"] = owner_moderate + renter_moderate

    owner_minor = m.get(_OWNER_SEVERITY["minor"], 0).fillna(0)
    m["total_minor"] = owner_minor  # renters schema has no minor bucket

    # Coalesce state/county identifiers
    for idcol in ("county", "state"):
        a, b = f"{idcol}_owners", f"{idcol}_renters"
        if a in m.columns and b in m.columns:
            m[idcol] = m[a].fillna(m[b])

    # Final shaped output
    out = m.rename(columns={
        "disasterNumber": "disaster_number",
        "zipCode": "zip_code",
        "totalInspected": "total_inspected",
        "totalApprovedIhpAmount": "total_approved_dollars",
    })
    keep = ["disaster_number", "zip_code"]
    if "state" in out.columns: keep.append("state")
    if "county" in out.columns: keep.append("county")
    keep += ["total_inspected", "total_major_substantial",
             "total_no_damage", "total_minor", "total_moderate",
             "total_approved_dollars", "validRegistrations",
             "repairReplaceAmount", "rentalAmount", "otherNeedsAmount"]
    out = out[[c for c in keep if c in out.columns]].copy()
    out["zip_code"] = out["zip_code"].astype(str).str.zfill(5)
    return out
