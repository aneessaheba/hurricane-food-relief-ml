# Hurricane Food Relief Priority

Predict how badly each zip code will be damaged by an incoming hurricane — using community vulnerability and storm data known before landfall — so emergency managers can prioritize food relief for food-insecure communities before ground-truth damage reports arrive.

**Team:** Group 2, DATA 245, SJSU.

---

## What this project does

1. Pulls FEMA Housing Assistance (Owners + Renters) data for **14 Atlantic-basin hurricanes (2016–2024)**: Matthew, Harvey, Irma, Florence, Michael, Laura, Sally, Delta, Zeta, Ida, Ian, Idalia, Helene, Milton. Every record has been physically inspected by a FEMA inspector — eliminating rejection/fraud noise present in raw IHP registrations.
2. Joins community vulnerability features (ACS demographics, CDC SVI, USDA Food Access Atlas, SNAP retailers, FEMA NRI hazard scores) with storm characteristics (IBTrACS track, wind, category, geodesic distance-to-track).
3. Trains a multi-class classifier (Low / Medium / High / Severe damage per 1,000 residents) with a **temporal train / val / test split**: TRAIN=2016–2018 (5 storms), VAL=2020 (4 storms), TEST=2021–2024 (5 storms).
4. Multiplies `P(High ∪ Severe)` by a 4-component food-fragility score (food desert flag, SNAP retailer density, distance to nearest supermarket, % households without a vehicle) to produce a **Food Relief Priority Index**.
5. Includes an equity audit (Fairlearn) to verify recall for the Severe class is not lower in high-SVI communities.

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

Notebook 01 downloads ~5 GB and takes 1–3 hours (Irma's IHP file alone is 1.45 GB). Notebooks 03 (geospatial fusion) and 06 (modeling + Optuna) are each ~15–30 min.

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
| 11 | FEMA National Risk Index (Census Tracts) | https://www.arcgis.com/home/item.html?id=9da4eeb936544335a6db0cd7a8448a51 | Hazard scores (coastal flood, hurricane) |
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
│   ├── 03_data_fusion.ipynb          ← tract-to-zip, spatial joins, NRI hazard scores
│   ├── 04_feature_engineering.ipynb  ← targets, severity bins, splits, ABT export
│   ├── 05_unsupervised_modeling.ipynb   ← K-Means, PCA, DBSCAN
│   ├── 06_supervised_modeling.ipynb  ← 5 classifiers + 2 regressors, tuning
│   ├── 07_evaluation_and_shap.ipynb  ← TEST set, SHAP, equity audit
│   └── 08_priority_index.ipynb       ← priority index + folium maps
├── src/
│   ├── data_acquisition.py           ← FEMA paginator, Census client, downloaders
│   ├── data_fusion.py                ← crosswalk, spatial joins, Geod distance, NRI tract-to-zip
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
- **Hurricane**: any of the 5 TEST hurricanes (Ida, Ian, Idalia, Helene, Milton).
- **Minimum priority score**: 0–1.
- **Top-N**: number of zips to show.
- **Food deserts only** / **High-SVI (Q4) only** toggles.

The app renders the pre-generated folium choropleth, a sortable table, a per-zip SHAP waterfall, and a CSV download button.

## Key results

Best classifier: **XGBoost**. Best regressor: **Random Forest** (with `log1p` target transform).

### Classification on held-out TEST set (5 hurricanes, 2,587 zip × hurricane rows)
| metric | value |
|---|---|
| Accuracy | 0.57 |
| Weighted F1 | 0.53 |
| Macro ROC-AUC | **0.78** |
| Severe-class recall | **0.82** |
| Low-class F1 | 0.79 |

### Per-hurricane regression (`verified_damage_per_1000`)
| hurricane | RMSE | MAE | R² |
|---|---|---|---|
| Milton | 19.4 | 9.4 | **+0.35** |
| Ian | 48.5 | 20.9 | +0.19 |
| Idalia | 172.5 | 24.7 | +0.06 |
| Helene | 48.4 | 30.7 | -0.10 |
| Ida | 162.5 | 92.9 | -0.18 |

### Equity audit by SVI quartile (Fairlearn)
| quartile | accuracy | F1 | recall (Severe) |
|---|---|---|---|
| Q1 (low SVI) | 0.56 | 0.56 | 0.65 |
| Q2 | 0.56 | 0.51 | 0.83 |
| Q3 | 0.53 | 0.45 | 0.86 |
| **Q4 (high SVI)** | **0.63** | 0.55 | **0.89** |

Severe-class recall is monotonically higher for more vulnerable communities — the model identifies severely damaged vulnerable zips *better*, in line with the relief-targeting goal.

### Food relief priority index — top-50 vs bottom-50 zips
| metric | top 50 | bottom 50 |
|---|---|---|
| food-desert count | **50 of 50** | 0 of 50 |
| mean SVI | 0.76 | 0.36 |
| mean actual damage / 1k residents | **69.2** | 1.6 |

The top-50 priority zips are 100% food deserts, in the 76th percentile of social vulnerability nationally, and experience **43× the actual hurricane damage** of the bottom-50 — providing a strong operational ranking for relief logistics.

### Cluster ablation (K-Means)
With/without the unsupervised cluster_label feature: ≤0.013 weighted-F1 difference across all five classifiers (XGBoost is marginally *better* without it). Conclusion: SVI + demographic features already capture the vulnerability profile that K-Means recovers; the cluster feature is redundant.

## Critical implementation notes (the traps)

1. **imblearn.pipeline.Pipeline** with SMOTE — never sklearn's (which leaks into predict()).
2. **pyproj.Geod** for all lat/lon distances. Euclidean on degrees is a bug.
3. **EPSG:5070** (Albers Equal Area) for every area calculation; EPSG:4326 for lat/lon joins.
4. CDC SVI uses **-999** as a missing-data sentinel — replace with NaN before any aggregation.
5. FEMA API caps responses at 10,000 records — must paginate with `$skip` plus retry/backoff.
6. Housing Assistance Owners + Renters are aggregated to (disaster, zip) **before** the join, then summed across populations — to avoid Cartesian explosion when zips span multiple counties.
7. `StratifiedGroupKFold` groups are passed to `.fit()`, not the constructor; n_splits is capped at the number of distinct groups.
8. Food-fragility MinMaxScalers are fit on TRAIN only, then applied to val/test.
9. NRI hazard scores arrive at the census-tract level (zip-level was discontinued in 2025) and are aggregated to ZIP via the HUD crosswalk, same as SVI/Food Atlas.
10. Continuous regression target is right-tailed; we predict `log1p(damage)` and invert with `expm1` at predict time.
11. All `random_state=42`.

## License & attribution

This is a course project. Data is used under the public-domain / open-data licenses of the respective providers (FEMA, USDA, CDC, Census, NOAA, HUD).
