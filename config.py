"""
Project configuration: constants, paths, API endpoints, hurricane metadata.

All hard-coded decisions live here so notebooks and src/ modules stay generic.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------------
DATA_PATHS = {
    "raw": PROJECT_ROOT / "data" / "raw",
    "interim": PROJECT_ROOT / "data" / "interim",
    "processed": PROJECT_ROOT / "data" / "processed",
    "models": PROJECT_ROOT / "models",
    "outputs": PROJECT_ROOT / "outputs",
}
for _p in DATA_PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Hurricanes in scope (temporal split)
# -----------------------------------------------------------------------------
HURRICANE_META = [
    # ── TRAIN: 2016–2018 ─────────────────────────────────────────
    {"name": "Matthew",  "year": 2016, "disaster_number": 4283, "category": 5,
     "max_wind_kt": 145, "states_affected": ["FL","GA","SC","NC"], "split": "TRAIN"},
    {"name": "Harvey",   "year": 2017, "disaster_number": 4332, "category": 4,
     "max_wind_kt": 115, "states_affected": ["TX"],            "split": "TRAIN"},
    {"name": "Irma",     "year": 2017, "disaster_number": 4337, "category": 4,
     "max_wind_kt": 155, "states_affected": ["FL","GA","SC"],  "split": "TRAIN"},
    {"name": "Florence", "year": 2018, "disaster_number": 4393, "category": 1,
     "max_wind_kt": 80,  "states_affected": ["NC","SC"],       "split": "TRAIN"},
    {"name": "Michael",  "year": 2018, "disaster_number": 4399, "category": 5,
     "max_wind_kt": 140, "states_affected": ["FL"],            "split": "TRAIN"},
    # Dorian excluded: FEMA HA has no zip-level records for DR 4465.

    # ── VAL: 2020 season ─────────────────────────────────────────
    {"name": "Laura",    "year": 2020, "disaster_number": 4559, "category": 4,
     "max_wind_kt": 130, "states_affected": ["LA"],            "split": "VAL"},
    {"name": "Sally",    "year": 2020, "disaster_number": 4563, "category": 2,
     "max_wind_kt": 90,  "states_affected": ["AL","FL"],       "split": "VAL"},
    {"name": "Delta",    "year": 2020, "disaster_number": 4570, "category": 2,
     "max_wind_kt": 85,  "states_affected": ["LA"],            "split": "VAL"},
    {"name": "Zeta",     "year": 2020, "disaster_number": 4577, "category": 3,
     "max_wind_kt": 100, "states_affected": ["LA","MS","AL"],  "split": "VAL"},

    # ── TEST: 2021–2024 ──────────────────────────────────────────
    {"name": "Ida",      "year": 2021, "disaster_number": 4611, "category": 4,
     "max_wind_kt": 130, "states_affected": ["LA"],            "split": "TEST"},
    # Nicholas (4623) excluded — FEMA HA/IHP returned 0 records (no FEMA Individual
    # Assistance was activated; storm caused mostly commercial wind damage).
    {"name": "Ian",      "year": 2022, "disaster_number": 4673, "category": 4,
     "max_wind_kt": 140, "states_affected": ["FL"],            "split": "TEST"},
    {"name": "Idalia",   "year": 2023, "disaster_number": 4734, "category": 3,
     "max_wind_kt": 110, "states_affected": ["FL","GA","SC"],  "split": "TEST"},
    {"name": "Helene",   "year": 2024, "disaster_number": 4830, "category": 4,
     "max_wind_kt": 120, "states_affected": ["FL","GA","NC","SC","TN"], "split": "TEST"},
    {"name": "Milton",   "year": 2024, "disaster_number": 4834, "category": 3,
     "max_wind_kt": 110, "states_affected": ["FL"],            "split": "TEST"},
]

STATES_IN_SCOPE = ["TX", "LA", "FL", "NC", "SC", "GA", "AL", "MS", "TN", "VA"]

# -----------------------------------------------------------------------------
# Data source URLs / endpoints
# -----------------------------------------------------------------------------
API_ENDPOINTS = {
    "fema_housing_owners":   "https://www.fema.gov/api/open/v2/HousingAssistanceOwners",
    "fema_housing_renters":  "https://www.fema.gov/api/open/v2/HousingAssistanceRenters",
    "fema_ihp_registrations": "https://www.fema.gov/api/open/v2/IndividualsAndHouseholdsProgramValidRegistrations",
    "fema_disasters":        "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries",
    "census_acs5":           "https://api.census.gov/data/2021/acs/acs5",
    # NFHL REST — layer number drifts. 28 was SFHA_ZONE; also try layer 27/16
    # at hazards.fema.gov or the newer https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer
    "nfhl_rest":             "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query",
}

DOWNLOAD_URLS = {
    "ibtracs_na": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv",
    # The data portal URLs below rot frequently. If 404, go to the landing page,
    # right-click the download link, and save to data/raw/ manually. Notebook 03
    # will fall back gracefully if any of these are missing.
    "food_atlas":   "https://www.ers.usda.gov/sites/default/files/_laserfiche/DataFiles/80591/FoodAccessResearchAtlasData2019.xlsx",
    "cdc_svi_2022":  "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_US.csv",
    # SNAP and HUD are behind landing pages — direct URL changes frequently.
    # Left as best-guess; acquisition will print manual-download instructions on 404.
    "snap_retailers": "https://usda-snap-retailer-api-data-public.s3.amazonaws.com/historical_snap_retailer_locator_data.zip",
    "hud_tract_zip": "https://www.huduser.gov/portal/datasets/usps/TRACT_ZIP_032024.xlsx",
    # TIGER/Line ZCTA 2020
    "zcta_shapefile": "https://www2.census.gov/geo/tiger/TIGER2022/ZCTA520/tl_2022_us_zcta520.zip",
    # FEMA National Risk Index — pre-aggregated zip-level flood/hazard scores.
    # Replaces the unreliable NFHL polygon overlay. Direct CSV bundle.
    "fema_nri_tract": "https://opendata.arcgis.com/api/v3/datasets/9da4eeb936544335a6db0cd7a8448a51_0/downloads/data?format=csv&spatialRefId=4326",
    # NOAA Storm Events bulk — user fills in specific year CSV files
    "noaa_storm_events_base": "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
}

# Manual-download landing pages (shown when direct URL 404s)
MANUAL_DOWNLOAD_PAGES = {
    "food_atlas":     "https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data",
    "snap_retailers": "https://www.fns.usda.gov/snap/retailer-locator",
    "cdc_svi_2022":   "https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html",
    "hud_tract_zip":  "https://www.huduser.gov/portal/datasets/usps_crosswalk.html",
}

# -----------------------------------------------------------------------------
# Target construction
# -----------------------------------------------------------------------------
TARGET_COL = "verified_damage_per_1000"
TARGET_CLASS_COL = "damage_severity_class"
SECONDARY_TARGETS = ["pct_major_substantial", "approved_dollars_per_capita"]

# Initial severity bin edges (recalibrate after EDA if class imbalance is extreme)
# Classes: Low <2, Medium 2-8, High 8-25, Severe >25
SEVERITY_BINS = [-float("inf"), 2.0, 8.0, 25.0, float("inf")]
SEVERITY_LABELS = ["Low", "Medium", "High", "Severe"]

# -----------------------------------------------------------------------------
# Feature groups (exclude identifiers & targets from the model matrix)
# -----------------------------------------------------------------------------
FEATURE_GROUPS = {
    "identifiers": [
        "zip_code", "state", "hurricane_name", "hurricane_year",
        "disaster_number", "train_test_split",
    ],
    "demographics": [
        "population", "median_income", "pct_poverty", "pct_renters",
        "pct_elderly_65plus", "pct_minority", "pct_no_vehicle",
        "pct_mobile_homes", "housing_density_per_sqmi",
    ],
    "svi": [
        "svi_socioeconomic", "svi_household_comp", "svi_minority_lang",
        "svi_housing_transport", "svi_overall",
    ],
    "food_access": [
        "food_desert_flag", "snap_retailer_count", "snap_retailers_per_1k",
        "dist_nearest_supermarket_mi", "snap_households_avg",
    ],
    "flood": ["pct_in_100yr_floodplain", "nri_cflood_score", "nri_hrcn_score"],
    "storm": [
        "hurricane_category", "max_wind_speed_kt",
        "total_rainfall_inches", "distance_to_track_km",
    ],
    "targets": [
        "total_inspected", "total_major_substantial", "total_approved_dollars",
        "verified_damage_per_1000", "pct_major_substantial",
        "approved_dollars_per_capita", "damage_severity_class",
    ],
    "derived_output": [
        "food_fragility_score", "prob_high_or_severe", "food_relief_priority_index",
    ],
}

# Continuous vs binary vs categorical for the modeling ColumnTransformer
CONTINUOUS_FEATURES = (
    FEATURE_GROUPS["demographics"]
    + FEATURE_GROUPS["svi"]
    + ["snap_retailer_count", "snap_retailers_per_1k",
       "dist_nearest_supermarket_mi", "snap_households_avg"]
    + FEATURE_GROUPS["flood"]
    + ["max_wind_speed_kt", "total_rainfall_inches", "distance_to_track_km"]
)
BINARY_FEATURES = ["food_desert_flag", "food_data_missing"]
CATEGORICAL_FEATURES = ["hurricane_category", "state"]

# Census ACS variable codes
ACS_VARIABLES = {
    "B01003_001E": "population",
    "B19013_001E": "median_income",
    "B17001_002E": "poverty_count",
    "B25003_003E": "renters",
    "B01001_020E": "male_65_66",
    "B01001_021E": "male_67_69",
    "B01001_022E": "male_70_74",
    "B01001_023E": "male_75_79",
    "B01001_024E": "male_80_84",
    "B01001_025E": "male_85_over",
    "B01001_044E": "female_65_66",
    "B01001_045E": "female_67_69",
    "B01001_046E": "female_70_74",
    "B01001_047E": "female_75_79",
    "B01001_048E": "female_80_84",
    "B01001_049E": "female_85_over",
    "B02001_002E": "white_alone",
    "B08201_002E": "no_vehicle_households",
    "B25024_010E": "mobile_homes",
}

# SFHA flood zones considered "100-year floodplain"
SFHA_ZONES = ["A", "AE", "AH", "AO", "V", "VE"]

# CRS conventions
CRS_LATLON = "EPSG:4326"
CRS_AREA = "EPSG:5070"  # Albers Equal Area (CONUS) — use for all area computations
