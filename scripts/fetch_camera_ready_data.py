"""
Download the extra inputs needed by scripts/camera_ready_experiments.py.

  ihp       FEMA IHP valid registrations with foodNeed == true, counted per
            (disaster, damaged ZIP) -> data/raw/ihp_foodneed_by_zip.csv
  acs       ACS 5-year ZCTA tables for every storm year (2016-2024)
            -> data/raw/acs5_<year>_zcta.csv
  gazetteer Census 2020 ZCTA gazetteer (internal-point centroids)
            -> data/raw/zcta_gazetteer_2020.txt
  ibtracs   NOAA IBTrACS North Atlantic best tracks -> data/raw/ibtracs_na.csv
  zcta      Census 2020 ZCTA cartographic boundaries (for notebook 08 maps)
            -> data/raw/zcta/cb_2020_us_zcta520_500k.shp

Every step skips files that already exist. Usage:
    python scripts/fetch_camera_ready_data.py [ihp acs gazetteer ibtracs zcta]
"""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ACS_VARIABLES, API_ENDPOINTS, DATA_PATHS, DOWNLOAD_URLS, HURRICANE_META  # noqa: E402

RAW = DATA_PATHS["raw"]
TEST_DISASTERS = [h["disaster_number"] for h in HURRICANE_META if h["split"] == "TEST"]
ACS_YEARS = sorted({h["year"] for h in HURRICANE_META} | {2021})
ZCTA_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
GAZETTEER_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
                 "2020_Gazetteer/2020_Gaz_zcta_national.zip")


def _get(url, params=None, timeout=180, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} after error: {e}")
            time.sleep(5 * (i + 1))


def fetch_ihp_foodneed() -> None:
    dest = RAW / "ihp_foodneed_by_zip.csv"
    if dest.exists():
        print(f"[skip] {dest.name}")
        return
    parts = []
    for dn in TEST_DISASTERS:
        # One checkpoint per disaster so a timeout does not lose finished storms.
        part = RAW / f"ihp_foodneed_{dn}.csv"
        if not part.exists():
            # Keyset paging on the integer id: deep $skip pages time out server-side.
            rows, last_id = [], -1
            while True:
                params = {
                    "$filter": f"disasterNumber eq {dn} and foodNeed eq true and id gt {last_id}",
                    "$select": "id,damagedZipCode",
                    "$orderby": "id",
                    "$top": 10000, "$format": "json",
                }
                page = _get(API_ENDPOINTS["fema_ihp_registrations"], params).json()
                recs = page.get("IndividualsAndHouseholdsProgramValidRegistrations", [])
                if not recs:
                    break
                rows += [(dn, r.get("damagedZipCode")) for r in recs]
                last_id = recs[-1]["id"]
            pd.DataFrame(rows, columns=["disaster_number", "zip_code"]).to_csv(part, index=False)
        parts.append(pd.read_csv(part, dtype={"zip_code": str}))
        print(f"  disaster {dn}: {len(parts[-1]):,} registrations with foodNeed")
    df = pd.concat(parts, ignore_index=True).dropna()
    df["zip_code"] = df["zip_code"].astype(str).str[:5].str.zfill(5)
    out = df.groupby(["disaster_number", "zip_code"]).size().rename("food_need_count")
    out.reset_index().to_csv(dest, index=False)
    print(f"  wrote {dest.name} rows={len(out):,}")


def fetch_acs_vintages() -> None:
    api_key = os.environ.get("CENSUS_API_KEY")
    varlist = ",".join(ACS_VARIABLES.keys())
    for year in ACS_YEARS:
        dest = RAW / f"acs5_{year}_zcta.csv"
        if dest.exists():
            print(f"[skip] {dest.name}")
            continue
        params = {"get": f"NAME,{varlist}", "for": "zip code tabulation area:*"}
        if api_key:
            params["key"] = api_key
        url = f"https://api.census.gov/data/{year}/acs/acs5"
        payload = _get(url, params, timeout=300).json()
        df = pd.DataFrame(payload[1:], columns=payload[0])
        df = df.rename(columns={**ACS_VARIABLES, "zip code tabulation area": "zip_code"})
        for col in ACS_VARIABLES.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(dest, index=False)
        print(f"  wrote {dest.name} rows={len(df):,}")


def fetch_gazetteer() -> None:
    dest = RAW / "zcta_gazetteer_2020.txt"
    if dest.exists():
        print(f"[skip] {dest.name}")
        return
    z = zipfile.ZipFile(io.BytesIO(_get(GAZETTEER_URL).content))
    name = next(n for n in z.namelist() if n.endswith(".txt"))
    dest.write_bytes(z.read(name))
    print(f"  wrote {dest.name}")


def fetch_ibtracs() -> None:
    dest = RAW / "ibtracs_na.csv"
    if dest.exists():
        print(f"[skip] {dest.name}")
        return
    dest.write_bytes(_get(DOWNLOAD_URLS["ibtracs_na"], timeout=600).content)
    print(f"  wrote {dest.name}")


def fetch_zcta_boundaries() -> None:
    dest = RAW / "zcta"
    if any(dest.glob("*.shp")):
        print(f"[skip] {dest.name}/")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(_get(ZCTA_URL, timeout=600).content)).extractall(dest)
    print(f"  wrote {dest.name}/")


STEPS = {
    "ihp": fetch_ihp_foodneed,
    "acs": fetch_acs_vintages,
    "gazetteer": fetch_gazetteer,
    "ibtracs": fetch_ibtracs,
    "zcta": fetch_zcta_boundaries,
}

if __name__ == "__main__":
    for step in sys.argv[1:] or list(STEPS):
        print(f"=== {step} ===")
        STEPS[step]()
