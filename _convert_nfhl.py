"""
Convert downloaded NFHL .gdb files to nfhl_sfha_{fips}.geojson in data/raw/.

Usage:
    python _convert_nfhl.py path/to/NFHL_48_20241001.gdb
    python _convert_nfhl.py path/to/unzipped_folder/   # converts all .gdb inside

Place the resulting geojson files in:
    data/raw/nfhl_sfha_{fips}.geojson
"""
import sys, json
from pathlib import Path
import geopandas as gpd

STATE_FIPS = {
    "TX": "48", "LA": "22", "FL": "12", "NC": "37",
    "SC": "45", "GA": "13", "AL": "01", "MS": "28",
}
FIPS_TO_STATE = {v: k for k, v in STATE_FIPS.items()}

RAW = Path(__file__).parent / "data" / "raw"


def convert_gdb(gdb_path: Path) -> None:
    gdb_path = Path(gdb_path)
    print(f"Reading S_FLD_HAZ_AR from {gdb_path.name} ...")
    gdf = gpd.read_file(gdb_path, layer="S_FLD_HAZ_AR")

    # Keep only SFHA zones
    sfha = gdf[gdf["SFHA_TF"] == "T"].copy()
    print(f"  {len(gdf)} total polygons -> {len(sfha)} SFHA polygons")

    if sfha.empty:
        print("  [warn] No SFHA polygons found — check layer name or SFHA_TF field")
        return

    # Reproject to WGS84
    sfha = sfha.to_crs("EPSG:4326")

    # Detect state FIPS from DFIRM_ID prefix (first 2 chars)
    if "DFIRM_ID" in sfha.columns:
        fips = sfha["DFIRM_ID"].dropna().str[:2].mode()[0]
    else:
        fips = input(f"  Enter state FIPS for {gdb_path.name} (e.g. 48 for TX): ").strip()

    out = RAW / f"nfhl_sfha_{fips}.geojson"
    cols = [c for c in ["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "DFIRM_ID", "geometry"]
            if c in sfha.columns]
    geojson = json.loads(sfha[cols].to_json())
    geojson["complete"] = True
    out.write_text(json.dumps(geojson), encoding="utf-8")
    state = FIPS_TO_STATE.get(fips, "?")
    print(f"  Saved {len(sfha)} features -> {out.name}  ({state})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.suffix == ".gdb" or (target.is_dir() and list(target.glob("*.gdb"))):
        gdbs = [target] if target.suffix == ".gdb" else list(target.glob("*.gdb"))
    else:
        # Maybe they passed the unzipped folder that contains the .gdb inside
        gdbs = list(target.rglob("*.gdb"))

    if not gdbs:
        print(f"No .gdb files found under {target}")
        sys.exit(1)

    for gdb in gdbs:
        convert_gdb(gdb)
    print("\nDone. Re-run notebooks 02 and 03.")
