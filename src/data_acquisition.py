"""
Data acquisition: downloads all 13 raw data sources into data/raw/.

Every function wraps network I/O in try/except with informative messages, uses
tqdm progress bars, and returns the local path(s) written. Functions are
idempotent: if the target file already exists and `force=False`, they skip.
"""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    API_ENDPOINTS, DOWNLOAD_URLS, DATA_PATHS, HURRICANE_META,
    STATES_IN_SCOPE, ACS_VARIABLES, MANUAL_DOWNLOAD_PAGES,
)

RAW = DATA_PATHS["raw"]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def download_file(url: str, dest: Path, force: bool = False, chunk: int = 1 << 15) -> Path:
    """Stream a URL to `dest` with a tqdm progress bar. Idempotent."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size:,} bytes)")
        return dest
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for block in r.iter_content(chunk_size=chunk):
                    fh.write(block)
                    pbar.update(len(block))
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e
    return dest


# -----------------------------------------------------------------------------
# FEMA API (generic paginator)
# -----------------------------------------------------------------------------
def fetch_fema_paginated(
    endpoint: str, disaster_number: int, top: int = 10000, max_pages: int = 500,
    max_retries: int = 6,
) -> pd.DataFrame:
    """
    Query a FEMA OpenFEMA v2 endpoint for a single disaster number, paginating
    with $skip and $top. Retries transient 5xx / timeouts with exponential
    backoff. Returns a DataFrame of all records.
    """
    all_rows = []
    skip = 0
    dataset_key = endpoint.rstrip("/").split("/")[-1]  # e.g. "HousingAssistanceOwners"
    pbar = tqdm(desc=f"FEMA {dataset_key} disaster={disaster_number}", unit="rec")
    for page in range(max_pages):
        params = {
            "$filter": f"disasterNumber eq {disaster_number}",
            "$top": top,
            "$skip": skip,
            "$format": "json",
        }
        data = None
        for attempt in range(max_retries):
            try:
                r = requests.get(endpoint, params=params, timeout=180)
                # retry on 5xx or 429
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} transient")
                r.raise_for_status()
                data = r.json()
                break
            except (requests.HTTPError, requests.ConnectionError,
                    requests.Timeout, ValueError) as e:
                wait = min(60, 2 ** attempt)
                pbar.write(f"  [retry {attempt+1}/{max_retries}] skip={skip} "
                           f"err={e} — sleeping {wait}s")
                time.sleep(wait)
        if data is None:
            raise RuntimeError(
                f"FEMA request failed after {max_retries} retries "
                f"(endpoint={endpoint}, disaster={disaster_number}, skip={skip})"
            )
        rows = data.get(dataset_key, [])
        if not rows:
            break
        all_rows.extend(rows)
        pbar.update(len(rows))
        if len(rows) < top:
            break
        skip += top
    pbar.close()
    return pd.DataFrame(all_rows)


def fetch_fema_housing_assistance(force: bool = False) -> dict[int, dict[str, Path]]:
    """
    Download Owners + Renters Housing Assistance for every hurricane in
    HURRICANE_META. Returns {disaster_number: {"owners": path, "renters": path}}.
    """
    out: dict[int, dict[str, Path]] = {}
    for h in HURRICANE_META:
        dn = h["disaster_number"]
        owners_path = RAW / f"fema_housing_owners_{dn}.csv"
        renters_path = RAW / f"fema_housing_renters_{dn}.csv"
        if not owners_path.exists() or force:
            df = fetch_fema_paginated(API_ENDPOINTS["fema_housing_owners"], dn)
            df.to_csv(owners_path, index=False)
            print(f"  wrote {owners_path.name} rows={len(df):,}")
        if not renters_path.exists() or force:
            df = fetch_fema_paginated(API_ENDPOINTS["fema_housing_renters"], dn)
            df.to_csv(renters_path, index=False)
            print(f"  wrote {renters_path.name} rows={len(df):,}")
        out[dn] = {"owners": owners_path, "renters": renters_path}
    return out


def fetch_fema_ihp_registrations(force: bool = False) -> dict[int, Path]:
    """
    IHP Valid Registrations — target-validation only, never the primary target.
    For Ida (4611) and Ian (4673) the registration tables are very large
    (~800k-1M rows); a 503 mid-pagination can lose work. We retry transparently
    and can also fall back to only counting registrations per zip (which is all
    notebook 02 needs anyway).
    """
    out = {}
    for h in HURRICANE_META:
        dn = h["disaster_number"]
        dest = RAW / f"fema_ihp_registrations_{dn}.csv"
        if not dest.exists() or force:
            try:
                df = fetch_fema_paginated(
                    API_ENDPOINTS["fema_ihp_registrations"], dn
                )
                df.to_csv(dest, index=False)
                print(f"  wrote {dest.name} rows={len(df):,}")
            except RuntimeError as e:
                print(f"[warn] IHP download failed for disaster {dn}: {e}")
                print(f"       skipping — this is validation-only data; "
                      f"notebook 02 will continue without this disaster.")
        out[dn] = dest
    return out


def fetch_fema_disaster_declarations(force: bool = False) -> Path:
    """Disaster declarations summary for all 8 disasters."""
    dest = RAW / "fema_disaster_declarations.csv"
    if dest.exists() and not force:
        print(f"[skip] {dest.name}")
        return dest
    dns = [h["disaster_number"] for h in HURRICANE_META]
    filt = " or ".join(f"disasterNumber eq {dn}" for dn in dns)
    params = {"$filter": filt, "$top": 10000, "$format": "json"}
    r = requests.get(API_ENDPOINTS["fema_disasters"], params=params, timeout=120)
    r.raise_for_status()
    rows = r.json().get("DisasterDeclarationsSummaries", [])
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"  wrote {dest.name} rows={len(rows):,}")
    return dest


# -----------------------------------------------------------------------------
# Census ACS 5-year at ZCTA
# -----------------------------------------------------------------------------
def fetch_census_acs(
    api_key: Optional[str] = None, force: bool = False
) -> Path:
    """
    Pull ACS 5-year variables for every ZCTA. A free API key from
    https://api.census.gov/data/key_signup.html is strongly recommended —
    unkeyed requests are throttled.
    """
    dest = RAW / "census_acs5_zcta.csv"
    if dest.exists() and not force:
        print(f"[skip] {dest.name}")
        return dest
    api_key = api_key or os.environ.get("CENSUS_API_KEY")
    varlist = ",".join(ACS_VARIABLES.keys())
    params = {
        "get": f"NAME,{varlist}",
        "for": "zip code tabulation area:*",
    }
    if api_key:
        params["key"] = api_key
    print("[census] querying ACS 5-year at ZCTA level (may take ~1 min)")
    r = requests.get(API_ENDPOINTS["census_acs5"], params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload[1:], columns=payload[0])
    df = df.rename(columns={**ACS_VARIABLES,
                            "zip code tabulation area": "zip_code"})
    # Cast numerics
    for col in ACS_VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(dest, index=False)
    print(f"  wrote {dest.name} rows={len(df):,}")
    return dest


# -----------------------------------------------------------------------------
# One-shot file downloads
# -----------------------------------------------------------------------------
def download_ibtracs(force: bool = False) -> Path:
    return download_file(DOWNLOAD_URLS["ibtracs_na"], RAW / "ibtracs_na.csv", force=force)


def download_food_atlas(force: bool = False) -> Path:
    return download_file(
        DOWNLOAD_URLS["food_atlas"],
        RAW / "food_access_atlas.xlsx",
        force=force,
    )


def download_snap_retailers(force: bool = False) -> Path:
    zpath = RAW / "snap_retailers.zip"
    if not zpath.exists() or force:
        download_file(DOWNLOAD_URLS["snap_retailers"], zpath, force=force)
    # extract
    try:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW / "snap_retailers")
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            f"SNAP retailer ZIP is corrupt — try re-downloading with force=True: {e}"
        ) from e
    return RAW / "snap_retailers"


def download_svi(force: bool = False) -> Path:
    return download_file(DOWNLOAD_URLS["cdc_svi_2022"], RAW / "cdc_svi_2022.csv", force=force)


def download_hud_crosswalk(force: bool = False) -> Path:
    return download_file(
        DOWNLOAD_URLS["hud_tract_zip"],
        RAW / "hud_tract_zip.xlsx",
        force=force,
    )


def download_zcta_shapefile(force: bool = False) -> Path:
    zpath = RAW / "zcta.zip"
    out_dir = RAW / "zcta"
    if not out_dir.exists() or force:
        download_file(DOWNLOAD_URLS["zcta_shapefile"], zpath, force=force)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out_dir)
    return out_dir


def download_storm_events(years: Iterable[int] = (2017, 2018, 2019, 2020, 2021, 2022),
                          force: bool = False) -> list[Path]:
    """
    Download bulk Storm Events details CSVs for given years. The NOAA FTP page
    lists filenames like StormEvents_details-ftp_v1.0_dYYYY_c*.csv.gz — the
    full listing URL must be scraped. For this project, we use a known pattern
    per year (user may need to update filenames if NOAA republishes).
    """
    out = []
    base = DOWNLOAD_URLS["noaa_storm_events_base"]
    for y in years:
        # Listing the bucket to discover the current filename; fall back to
        # requiring user to drop the file in manually if scraping fails.
        try:
            idx = requests.get(base, timeout=60).text
        except Exception as e:
            print(f"[warn] could not list NOAA storm-events index: {e}")
            idx = ""
        token = f"StormEvents_details-ftp_v1.0_d{y}_c"
        matches = [line for line in idx.splitlines() if token in line]
        if not matches:
            print(f"[warn] no NOAA storm-events file found for {y} — "
                  f"download manually to {RAW}/ and rerun")
            continue
        # extract filename from HTML anchor
        fname = matches[0].split(token, 1)[1].split('"', 1)[0]
        full = f"{base}{token}{fname}"
        dest = RAW / f"noaa_storm_events_{y}.csv.gz"
        download_file(full, dest, force=force)
        out.append(dest)
    return out


def download_nfhl_state(state_fips: str, force: bool = False) -> Optional[Path]:
    """
    Query the NFHL ArcGIS REST endpoint for SFHA polygons in a given state.
    Returns a GeoJSON path. For a real build, iterate county-by-county to
    avoid the 1000-record soft cap. This helper does a coarse state pull.
    """
    dest = RAW / f"nfhl_sfha_{state_fips}.geojson"
    if dest.exists() and not force:
        return dest
    params = {
        "where": "FLD_ZONE IN ('A','AE','AH','AO','V','VE') "
                 f"AND STATE_FIPS='{state_fips}'",
        "outFields": "FLD_ZONE,STATE_FIPS,COUNTY_FIPS",
        "outSR": 4326,
        "f": "geojson",
        "resultRecordCount": 2000,
    }
    try:
        r = requests.get(API_ENDPOINTS["nfhl_rest"], params=params, timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as e:
        print(f"[warn] NFHL fetch for state {state_fips} failed: {e}")
        return None
    return dest


# -----------------------------------------------------------------------------
# Convenience: run everything
# -----------------------------------------------------------------------------
def download_all(census_key: Optional[str] = None, force: bool = False) -> None:
    """Run every downloader in order. Safe to re-run."""
    print("\n=== 1. FEMA Housing Assistance (TARGET) ===")
    fetch_fema_housing_assistance(force=force)
    print("\n=== 2. FEMA IHP Valid Registrations (validation only) ===")
    fetch_fema_ihp_registrations(force=force)
    print("\n=== 3. FEMA Disaster Declarations ===")
    fetch_fema_disaster_declarations(force=force)
    print("\n=== 4. Census ACS 5-year ===")
    fetch_census_acs(api_key=census_key, force=force)
    print("\n=== 5. IBTrACS North Atlantic ===")
    download_ibtracs(force=force)
    def _try(name, fn, expected_filename):
        target = RAW / expected_filename
        if target.exists():
            print(f"[skip] {expected_filename} already present")
            return
        try:
            fn(force=force)
        except Exception as e:
            page = MANUAL_DOWNLOAD_PAGES.get(name, "")
            print(f"[warn] {name} download failed: {e}")
            print(f"       → MANUAL DOWNLOAD: open {page}")
            print(f"       → save the file to: {target}")

    print("\n=== 6. USDA Food Access Atlas ===")
    _try("food_atlas", download_food_atlas, "food_access_atlas.xlsx")
    print("\n=== 7. USDA SNAP Retailer Historical ===")
    _try("snap_retailers", download_snap_retailers, "snap_retailers.zip")
    print("\n=== 8. CDC SVI 2022 ===")
    _try("cdc_svi_2022", download_svi, "cdc_svi_2022.csv")
    print("\n=== 9. HUD Tract-ZIP Crosswalk ===")
    _try("hud_tract_zip", download_hud_crosswalk, "hud_tract_zip.xlsx")
    print("\n=== 10. Census TIGER/Line ZCTA shapefile ===")
    download_zcta_shapefile(force=force)
    print("\n=== 11. NOAA Storm Events (validation) ===")
    download_storm_events(force=force)
    print("\n=== 12. NFHL SFHA polygons (per state) ===")
    # Rough FIPS map for scope states
    state_fips = {"TX": "48", "LA": "22", "FL": "12", "NC": "37",
                  "SC": "45", "GA": "13", "AL": "01", "MS": "28"}
    for st in STATES_IN_SCOPE:
        download_nfhl_state(state_fips[st], force=force)
    print("\nAll downloads complete.")
