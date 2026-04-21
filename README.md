# Hurricane Food Relief Priority

Predict how badly each zip code will be damaged by an incoming hurricane — using community vulnerability and storm data known before landfall — so emergency managers can prioritize food relief for food-insecure communities before ground-truth damage reports arrive.

**Team:** Group 2, DATA 245, SJSU, Fall 2025.

---

## What this project does

1. Pulls FEMA Housing Assistance (Owners + Renters) data for 8 Gulf/Atlantic hurricanes (Harvey, Florence, Michael, Dorian, Laura, Delta, Ida, Ian). Every record has been physically inspected by a FEMA inspector — eliminating rejection/fraud noise present in raw IHP registrations.
2. Joins community vulnerability features (ACS demographics, CDC SVI, USDA Food Access Atlas, SNAP retailers, FEMA flood zones) with storm characteristics (IBTrACS track, wind, category, geodesic distance-to-track).
3. Trains a multi-class classifier (Low / Medium / High / Severe damage per 1,000 residents) with a **temporal train / val / test split**: TRAIN=2017–2019, VAL=2020, TEST=2021–2022.
4. Multiplies `P(High ∪ Severe)` by a 4-component food-fragility score (food desert flag, SNAP retailer density, distance to nearest supermarket, % households without a vehicle) to produce a **Food Relief Priority Index**.
5. Includes an equity audit (Fairlearn) to check that recall for the Severe class does not drop in high-SVI communities.

## Quick start

```bash
# 1. create env & install
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. free Census API key (1 minute)
#    https://api.census.gov/data/key_signup.html
export CENSUS_API_KEY="your-key"   # Windows: set CENSUS_API_KEY=your-key

# 3. run notebooks in order
jupyter lab   # then run 01 → 08 sequentially

# 4. launch the dashboard after notebook 08 completes
streamlit run app/streamlit_app.py
```

Notebook 01 downloads ~several GB and takes 30–90 min. Notebooks 03 (geospatial fusion) and 06 (modeling + Optuna) are each ~15–30 min.

## Data sources

| # | Source | URL | Role |
|---|---|---|---|
| 1 | FEMA Housing Assistance — Owners v2 | https://www.fema.gov/api/open/v2/HousingAssistanceOwners | **TARGET** (inspector-verified) |
| 2 | FEMA Housing Assistance — Renters v2 | https://www.fema.gov/api/open/v2/HousingAssistanceRenters | **TARGET** |
| 3 | FEMA IHP Valid Registrations v2 | https://www.fema.gov/api/open/v2/IndividualsAndHouseholdsProgramValidRegistrations | Validation only |
| 4 | FEMA Disaster Declarations v2 | https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries | Metadata |
| 5 | USDA Food Access Research Atlas | https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data | Food-access features |
| 6 | USDA SNAP Retailer Locator (Historical) | https://www.fns.usda.gov/snap/retailer-locator/data | Food-access features |
| 7 | CDC SVI 2022 | https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html | Social vulnerability |
| 8 | Census ACS 5-Year (2021) | https://api.census.gov/data/2021/acs/acs5 | Demographics |
| 9 | NOAA IBTrACS v04r01 | https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv | Storm tracks |
| 10 | NOAA Storm Events Database | https://www.ncdc.noaa.gov/stormevents/ftp.jsp | Validation |
| 11 | FEMA National Flood Hazard Layer | https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query | Flood exposure |
| 12 | HUD USPS ZIP Crosswalk (TRACT_ZIP) | https://www.huduser.gov/portal/datasets/usps_crosswalk.html | Geographic join |
| 13 | Census TIGER/Line ZCTA shapefile | https://www.census.gov/cgi-bin/geo/shapefiles/index.php | Spatial joins |

## Project structure

```
hurricane-food-relief/
├── README.md                         ← you are here
├── requirements.txt
├── config.py                         ← all constants, hurricane metadata, API URLs
├── notebooks/
│   ├── 01_data_acquisition.ipynb     ← downloads all 13 sources
│   ├── 02_eda.ipynb                  ← target exploration + IHP/NOAA validation
│   ├── 03_data_fusion.ipynb          ← tract-to-zip, spatial joins, flood overlay
│   ├── 04_feature_engineering.ipynb  ← targets, severity bins, splits, ABT export
│   ├── 05_unsupervised_modeling.ipynb   ← K-Means, PCA, DBSCAN
│   ├── 06_supervised_modeling.ipynb  ← 5 classifiers + 2 regressors, tuning
│   ├── 07_evaluation_and_shap.ipynb  ← TEST set, SHAP, equity audit
│   └── 08_priority_index.ipynb       ← priority index + folium maps
├── src/
│   ├── data_acquisition.py           ← FEMA paginator, Census client, downloaders
│   ├── data_fusion.py                ← crosswalk, spatial joins, Geod distance, overlay
│   ├── feature_engineering.py        ← target computation, binning, imputation
│   ├── modeling.py                   ← imblearn Pipeline, tuners, CV
│   ├── evaluation.py                 ← SHAP helpers, Fairlearn equity audit
│   └── priority_index.py             ← fragility scalers + index computation
├── data/
│   ├── raw/ interim/ processed/
├── models/                           ← joblib pickles
├── outputs/                          ← figures, HTML maps, CSV exports
└── app/
    └── streamlit_app.py              ← dashboard
```

## Running the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Sidebar controls:
- **Hurricane**: Ida (LA) or Ian (FL) from the TEST set.
- **Minimum priority score**: 0–1.
- **Top-N**: number of zips to show.
- **Food deserts only** / **High-SVI (Q4) only** toggles.

The app renders the pre-generated folium choropleth, a sortable table, a per-zip SHAP waterfall, and a CSV download button.

## Key results

> Placeholder — fill in after running notebooks 06–08:
> - Test-set weighted F1 (classifier): __
> - Test-set RMSE (regressor on verified_damage_per_1000): __
> - Fairness: demographic parity ratio (Severe, by SVI quartile): __
> - Top-50 priority zips: food-desert share vs bottom-50: __

## Critical implementation notes (the traps)

1. **imblearn.pipeline.Pipeline** with SMOTE — never sklearn's (which leaks into predict()).
2. **pyproj.Geod** for all lat/lon distances. Euclidean on degrees is a bug.
3. **EPSG:5070** (Albers Equal Area) for every area calculation; EPSG:4326 for lat/lon joins.
4. CDC SVI uses **-999** as a missing-data sentinel — replace with NaN before any aggregation.
5. FEMA API caps responses at 10,000 records — must paginate with `$skip`.
6. Housing Assistance Owners + Renters are **summed**, not averaged — they are different populations.
7. `StratifiedGroupKFold` groups are passed to `.fit()`, not the constructor.
8. Food-fragility MinMaxScalers are fit on TRAIN only, then applied to val/test.
9. Process NFHL flood overlay **county-by-county** to keep memory under control.
10. All `random_state=42`.

## License & attribution

This is a course project. Data is used under the public-domain / open-data licenses of the respective providers (FEMA, USDA, CDC, Census, NOAA, HUD).
