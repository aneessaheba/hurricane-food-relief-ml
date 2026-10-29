"""
Data acquisition: downloads all 13 raw data sources into data/raw/.

Every network call goes through `request_with_retry`, which retries rate
limits, 5xx responses, timeouts and dropped connections with exponential
backoff. Functions are idempotent: if the target file already exists and
`force=False`, they skip. Progress and warnings go to the `logging` module;
call `setup_logging()` (done by `download_all`) to see them.
"""
from __future__ import annotations

import logging
import os
import random
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
log = logging.getLogger(__name__)

TRANSIENT_STATUS = (429, 500, 502, 503, 504)


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Send pipeline log messages to stderr and, optionally, to `log_file`."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level, handlers=handlers, force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _backoff_seconds(attempt: int, retry_after: Optional[str] = None,
                     cap: float = 60.0) -> float:
    """Exponential backoff with jitter; honours a numeric Retry-After header."""
    if retry_after is not None:
        try:
            return min(cap, float(retry_after))
        except ValueError:
            pass
    return min(cap, 2 ** attempt) * (0.5 + random.random() / 2)


def request_with_retry(
    url: str, params: Optional[dict] = None, timeout: float = 180,
    max_retries: int = 6, stream: bool = False, sleep=time.sleep,
) -> requests.Response:
    """
    GET `url`, retrying rate limits (429), 5xx responses, timeouts and
    connection errors. Other 4xx errors fail immediately because retrying
    cannot fix them. `sleep` is injectable so tests run instantly.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, stream=stream)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err, retry_after = e, None
        else:
            if r.status_code not in TRANSIENT_STATUS:
                r.raise_for_status()
                return r
            last_err = requests.HTTPError(f"{r.status_code} from {url}", response=r)
            retry_after = r.headers.get("Retry-After")
            r.close()
        if attempt < max_retries - 1:
            wait = _backoff_seconds(attempt, retry_after)
            log.warning("retry %d/%d for %s after %s; sleeping %.1fs",
                        attempt + 1, max_retries, url, last_err, wait)
            sleep(wait)
    raise RuntimeError(f"Request failed after {max_retries} attempts: {url}: {last_err}") from last_err


def download_file(url: str, dest: Path, force: bool = False, chunk: int = 1 << 15,
                  max_retries: int = 4) -> Path:
    """
    Stream a URL to `dest` with a tqdm progress bar. Idempotent.

    Data is written to `<dest>.part` and renamed on success, so an interrupted
    download never leaves a truncated file that a later run would skip.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("[skip] %s already exists (%s bytes)", dest.name, f"{dest.stat().st_size:,}")
        return dest
    part = dest.with_name(dest.name + ".part")
    try:
        with request_with_retry(url, timeout=120, stream=True,
                                max_retries=max_retries) as r:
            total = int(r.headers.get("Content-Length", 0))
            with open(part, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for block in r.iter_content(chunk_size=chunk):
                    fh.write(block)
                    pbar.update(len(block))
        part.replace(dest)
    except Exception as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {e}") from e
    return dest


# -----------------------------------------------------------------------------
# FEMA API (generic paginator)
# -----------------------------------------------------------------------------
def _id_literal(value) -> str:
    """OData literal for an OpenFEMA id: integers bare, UUID strings quoted."""
    return str(value) if isinstance(value, int) else f"'{value}'"


def fetch_fema_paginated(
    endpoint: str, disaster_number: int, top: int = 10000, max_pages: int = 500,
    max_retries: int = 6, sleep=time.sleep,
) -> pd.DataFrame:
    """
    Query a FEMA OpenFEMA v2 endpoint for a single disaster number.

    Pages by keyset (`id gt <last id>`, ordered by id) rather than $skip:
    deep $skip offsets on the large IHP tables time out server-side, while a
    keyset page costs the same at any depth. Ids are integers on some
    endpoints and UUIDs on others; both are handled. Transient failures are
    retried with exponential backoff. Returns a DataFrame of all records.
    """
    all_rows = []
    last_id = None
    dataset_key = endpoint.rstrip("/").split("/")[-1]  # e.g. "HousingAssistanceOwners"
    pbar = tqdm(desc=f"FEMA {dataset_key} disaster={disaster_number}", unit="rec")
    for _ in range(max_pages):
        filt = f"disasterNumber eq {disaster_number}"
        if last_id is not None:
            filt += f" and id gt {_id_literal(last_id)}"
        params = {"$filter": filt, "$orderby": "id", "$top": top, "$format": "json"}
        try:
            r = request_with_retry(endpoint, params=params,
                                   max_retries=max_retries, sleep=sleep)
            data = r.json()
        except (RuntimeError, ValueError) as e:
            pbar.close()
            raise RuntimeError(
                f"FEMA request failed (endpoint={endpoint}, "
                f"disaster={disaster_number}, after id={last_id}): {e}"
            ) from e
        if "error" in data:
            pbar.close()
            raise RuntimeError(f"FEMA API error for {dataset_key}: {data['error']}")
        rows = data.get(dataset_key, [])
        if not rows:
            break
        all_rows.extend(rows)
        pbar.update(len(rows))
        if len(rows) < top:
            break
        last_id = rows[-1]["id"]
    else:
        log.warning("%s disaster=%s stopped at max_pages=%d; data may be incomplete",
                    dataset_key, disaster_number, max_pages)
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
            log.info("wrote %s rows=%s", owners_path.name, f"{len(df):,}")
        if not renters_path.exists() or force:
            df = fetch_fema_paginated(API_ENDPOINTS["fema_housing_renters"], dn)
            df.to_csv(renters_path, index=False)
            log.info("wrote %s rows=%s", renters_path.name, f"{len(df):,}")
        out[dn] = {"owners": owners_path, "renters": renters_path}
    return out


def fetch_fema_ihp_registrations(force: bool = False) -> dict[int, Path]:
    """
    IHP Valid Registrations — target-validation only, never the primary target.
    For Ida (4611) and Ian (4673) the registration tables are very large
    (~800k-1M rows). Keyset paging keeps each page cheap; if a disaster still
    fails it is skipped, since notebook 02 only needs per-zip counts.
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
                log.info("wrote %s rows=%s", dest.name, f"{len(df):,}")
            except RuntimeError as e:
                log.warning("IHP download failed for disaster %s: %s — skipping; "
                            "this is validation-only data and notebook 02 will "
                            "continue without this disaster.", dn, e)
        out[dn] = dest
    return out


def fetch_fema_disaster_declarations(force: bool = False) -> Path:
    """Disaster declarations summary for every disaster in HURRICANE_META."""
    dest = RAW / "fema_disaster_declarations.csv"
    if dest.exists() and not force:
        log.info("[skip] %s", dest.name)
        return dest
    dns = [h["disaster_number"] for h in HURRICANE_META]
    filt = " or ".join(f"disasterNumber eq {dn}" for dn in dns)
    params = {"$filter": filt, "$top": 10000, "$format": "json"}
    r = request_with_retry(API_ENDPOINTS["fema_disasters"], params=params, timeout=120)
    rows = r.json().get("DisasterDeclarationsSummaries", [])
    pd.DataFrame(rows).to_csv(dest, index=False)
    log.info("wrote %s rows=%s", dest.name, f"{len(rows):,}")
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
        log.info("[skip] %s", dest.name)
        return dest
    api_key = api_key or os.environ.get("CENSUS_API_KEY")
    varlist = ",".join(ACS_VARIABLES.keys())
    params = {
        "get": f"NAME,{varlist}",
        "for": "zip code tabulation area:*",
    }
    if api_key:
        params["key"] = api_key
    log.info("[census] querying ACS 5-year at ZCTA level (may take ~1 min)")
    r = request_with_retry(API_ENDPOINTS["census_acs5"], params=params, timeout=180)
    payload = r.json()
    df = pd.DataFrame(payload[1:], columns=payload[0])
    df = df.rename(columns={**ACS_VARIABLES,
                            "zip code tabulation area": "zip_code"})
    # Cast numerics
    for col in ACS_VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(dest, index=False)
    log.info("wrote %s rows=%s", dest.name, f"{len(df):,}")
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


def download_fema_nri_tract(force: bool = False) -> Optional[Path]:
    """FEMA National Risk Index — census-tract-level CSV via FEMA's ArcGIS
    Hub mirror (zip-level was discontinued in 2025). We aggregate to ZIP
    downstream via the HUD tract->zip crosswalk in notebook 03."""
    url = DOWNLOAD_URLS.get("fema_nri_tract")
    if not url:
        return None
    out_csv = RAW / "fema_nri_tract.csv"
    if out_csv.exists() and not force:
        return out_csv
    return download_file(url, out_csv, force=force)


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
    # One listing serves every year.
    try:
        idx = request_with_retry(base, timeout=60, max_retries=3).text
    except Exception as e:
        log.warning("could not list NOAA storm-events index: %s", e)
        idx = ""
    for y in years:
        token = f"StormEvents_details-ftp_v1.0_d{y}_c"
        matches = [line for line in idx.splitlines() if token in line]
        if not matches:
            log.warning("no NOAA storm-events file found for %s — "
                        "download manually to %s/ and rerun", y, RAW)
            continue
        # extract filename from HTML anchor
        fname = matches[0].split(token, 1)[1].split('"', 1)[0]
        full = f"{base}{token}{fname}"
        dest = RAW / f"noaa_storm_events_{y}.csv.gz"
        download_file(full, dest, force=force)
        out.append(dest)
    return out


_NFHL_STATE_BBOX = {
    # xmin, ymin, xmax, ymax  (WGS84)
    "48": (-106.65, 25.84, -93.51, 36.50),  # TX
    "22": (-94.04,  28.92, -88.82, 33.02),  # LA
    "12": (-87.63,  24.52, -80.03, 31.00),  # FL
    "37": (-84.32,  33.84, -75.46, 36.59),  # NC
    "45": (-83.35,  32.04, -78.54, 35.22),  # SC
    "13": (-85.61,  30.36, -80.84, 35.00),  # GA
    "01": (-88.47,  30.14, -84.89, 35.01),  # AL
    "28": (-91.65,  30.17, -88.10, 35.01),  # MS
}


def download_nfhl_state(state_fips: str, force: bool = False) -> Optional[Path]:
    """
    Query the NFHL ArcGIS REST endpoint for SFHA polygons in a given state,
    paginating at 2000 records per page (server hard limit).

    Uses a bounding-box spatial filter (spatially indexed) rather than a
    DFIRM_ID LIKE filter (non-indexed string scan that causes 500 timeouts).
    """
    import json as _json
    dest = RAW / f"nfhl_sfha_{state_fips}.geojson"

    # Load any previously saved (partial) data so we can resume
    all_features = []
    crs = None
    if dest.exists() and not force:
        try:
            cached = _json.loads(dest.read_bytes())
            if "error" in cached or "type" not in cached:
                dest.unlink()
            elif cached.get("complete"):
                return dest  # already fully downloaded
            else:
                all_features = cached.get("features", [])
                crs = cached.get("crs")
                log.info("state %s: resuming from %d cached features",
                         state_fips, len(all_features))
        except Exception:
            dest.unlink()

    bbox = _NFHL_STATE_BBOX.get(state_fips)
    if bbox is None:
        log.warning("No bounding box defined for state FIPS %s", state_fips)
        return None

    xmin, ymin, xmax, ymax = bbox
    base_params = {
        "where": "SFHA_TF='T'",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": 4326,
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,DFIRM_ID",
        "outSR": 4326,
        "geometryPrecision": 5,
        "f": "geojson",
        "resultRecordCount": 2000,
    }

    complete = False
    offset = len(all_features)
    while True:
        try:
            params = {**base_params, "resultOffset": offset}
            r = request_with_retry(API_ENDPOINTS["nfhl_rest"], params=params,
                                   timeout=300, max_retries=3)
            page = r.json()
        except Exception as e:
            log.warning("NFHL page failed for state %s at offset=%d: %s",
                        state_fips, offset, e)
            break
        if "error" in page:
            log.warning("NFHL API error for state %s offset=%d: %s",
                        state_fips, offset, page["error"])
            break
        features = page.get("features", [])
        if not features:
            complete = True
            break
        all_features.extend(features)
        if crs is None:
            crs = page.get("crs")
        log.debug("state %s: fetched %d features", state_fips, len(all_features))
        if len(features) < 2000:
            complete = True
            break
        offset += 2000
        # Save after each successful page so progress survives crashes
        geojson = {"type": "FeatureCollection", "features": all_features}
        if crs:
            geojson["crs"] = crs
        dest.write_text(_json.dumps(geojson), encoding="utf-8")

    if not all_features:
        log.warning("NFHL: no features returned for state %s", state_fips)
        return None

    geojson = {"type": "FeatureCollection", "features": all_features}
    if crs:
        geojson["crs"] = crs
    if complete:
        geojson["complete"] = True
    dest.write_text(_json.dumps(geojson), encoding="utf-8")
    status = "complete" if complete else f"partial ({len(all_features)} features)"
    log.info("state %s: %s -> %s", state_fips, status, dest.name)
    return dest


# -----------------------------------------------------------------------------
# Convenience: run everything
# -----------------------------------------------------------------------------
def download_all(census_key: Optional[str] = None, force: bool = False,
                 log_file: Optional[Path] = None) -> None:
    """Run every downloader in order. Safe to re-run."""
    setup_logging(log_file=log_file or DATA_PATHS["interim"] / "download.log")
    log.info("=== 1. FEMA Housing Assistance (TARGET) ===")
    fetch_fema_housing_assistance(force=force)
    log.info("=== 2. FEMA IHP Valid Registrations (validation only) ===")
    fetch_fema_ihp_registrations(force=force)
    log.info("=== 3. FEMA Disaster Declarations ===")
    fetch_fema_disaster_declarations(force=force)
    log.info("=== 4. Census ACS 5-year ===")
    fetch_census_acs(api_key=census_key, force=force)
    log.info("=== 5. IBTrACS North Atlantic ===")
    download_ibtracs(force=force)

    def _try(name, fn, expected_filename):
        target = RAW / expected_filename
        if target.exists():
            log.info("[skip] %s already present", expected_filename)
            return
        try:
            fn(force=force)
        except Exception as e:
            page = MANUAL_DOWNLOAD_PAGES.get(name, "")
            log.warning("%s download failed: %s\n  -> MANUAL DOWNLOAD: open %s\n"
                        "  -> save the file to: %s", name, e, page, target)

    log.info("=== 6. USDA Food Access Atlas ===")
    _try("food_atlas", download_food_atlas, "food_access_atlas.xlsx")
    log.info("=== 7. USDA SNAP Retailer Historical ===")
    _try("snap_retailers", download_snap_retailers, "snap_retailers.zip")
    log.info("=== 8. CDC SVI 2022 ===")
    _try("cdc_svi_2022", download_svi, "cdc_svi_2022.csv")
    log.info("=== 9. HUD Tract-ZIP Crosswalk ===")
    _try("hud_tract_zip", download_hud_crosswalk, "hud_tract_zip.xlsx")
    log.info("=== 10. Census TIGER/Line ZCTA shapefile ===")
    download_zcta_shapefile(force=force)
    log.info("=== 11. NOAA Storm Events (validation) ===")
    download_storm_events(force=force)
    log.info("=== 12. NFHL SFHA polygons (per state, best-effort) ===")
    state_fips = {"TX": "48", "LA": "22", "FL": "12", "NC": "37",
                  "SC": "45", "GA": "13", "AL": "01", "MS": "28",
                  "TN": "47", "VA": "51"}
    for st in STATES_IN_SCOPE:
        if st in state_fips:
            download_nfhl_state(state_fips[st], force=force)
    log.info("=== 13. FEMA National Risk Index (census-tract level) ===")
    _try("fema_nri_tract", download_fema_nri_tract, "fema_nri_tract.csv")
    log.info("All downloads complete.")
