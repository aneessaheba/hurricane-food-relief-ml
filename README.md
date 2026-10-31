# Hurricane Food Relief Priority

Predict how badly each zip code will be damaged by an incoming hurricane — using community vulnerability and storm data known before landfall — so emergency managers can prioritize food relief for food-insecure communities before ground-truth damage reports arrive.

**Team:** Group 2, DATA 245, SJSU.

**Paper:** A. S. Guddi, C. Uppalapati, C. S. Elthuri, M. Kaur, and S.-Y. Chang, "Equitable Pre-Landfall Food Relief Prioritization via Multi-Source Machine Learning: A ZIP-Code Framework Across 14 Atlantic Hurricanes," *IEEE International Conference on Big Data (BigData 2026)*, accepted regular paper (BigD261). LaTeX source: [`ieee_paper.tex`](ieee_paper.tex). See [Reproducing the paper](#reproducing-the-paper).

---

## What this project does

1. Pulls FEMA Housing Assistance (Owners + Renters) data for **14 Atlantic-basin hurricanes (2016–2024)**: Matthew, Harvey, Irma, Florence, Michael, Laura, Sally, Delta, Zeta, Ida, Ian, Idalia, Helene, Milton. Every record has been physically inspected by a FEMA inspector — eliminating rejection/fraud noise present in raw IHP registrations.
2. Joins community vulnerability features (ACS demographics, CDC SVI, USDA Food Access Atlas, SNAP retailers, FEMA NRI hazard scores) with storm characteristics (IBTrACS track, wind, category, geodesic distance-to-track).
3. Trains a multi-class classifier (Low / Medium / High / Severe damage per 1,000 residents) with a **temporal train / val / test split**: TRAIN=2016–2018 (5 storms), VAL=2020 (4 storms), TEST=2021–2024 (5 storms).
4. Calibrates the classifier's probabilities (isotonic regression fit on VAL) and combines them with a food-fragility score into a **Food Relief Priority Index**: `FRPI = 0.6 · P(High ∪ Severe) + 0.4 · FFS`, where the FFS averages scaled food-desert flag, supermarket distance, SNAP retailer count and density (inverted), SNAP participation and SVI.
5. Includes an equity audit (Fairlearn) by SVI quartile, plus robustness checks requested by the paper's reviewers: leave-one-hurricane-out and expanding-window evaluation, a ZIP-disjoint control, feature ablation, year-matched ACS data, validation against FEMA applicants' reported food need, and a stress test with National Hurricane Center-scale track forecast errors.

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

## Team setup (skip the long downloads)

Teammates who just want to **inspect results, run the dashboard, or re-train on the prebuilt features** don't need to re-run notebook 01. Run these four commands once:

```bash
# 1. clone + env
git clone https://github.com/aneessaheba/hurricane-food-relief-ml.git
cd hurricane-food-relief-ml
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. pull the prebuilt artifacts bundle (~21 MB zipped, 54 MB extracted)
python scripts/fetch_share_bundle.py "https://drive.google.com/file/d/1d5kVJ-tmUWaTb7WS4ojIF4Bng0QAtabV/view?usp=sharing"

# 3. inspect results — pick one
streamlit run app/streamlit_app.py        # interactive dashboard with priority maps + SHAP
jupyter lab                                # re-run any notebook against the cached artifacts
```

The bundle contains:
- `data/processed/abt.csv` — analytic base table as published (5,522 zip × hurricane rows; see [Two versions of the ABT](#two-versions-of-the-abt))
- `data/processed/abt_with_clusters.csv` — ABT + K-Means cluster labels
- `data/processed/abt.xlsx` — color-coded 3-sheet workbook (data + dictionary + summary)
- `data/interim/fused_zip_hurricane.csv` — pre-feature-engineering fusion
- `models/best_classifier.pkl` — XGBoost (saved by notebook 06)
- `models/best_regressor.pkl` — Random Forest with log1p target
- `models/fragility_scalers.pkl` — MinMax scalers fit on TRAIN
- `outputs/priority_rankings.csv` — 2,587-zip priority index for the 5 TEST hurricanes

The fetch script auto-handles Google Drive's confirm-token dance via `gdown`, falls back to `curl` for non-Drive URLs, and prints a verification table at the end so you know each file landed.

### Need the raw 5 GB downloads?

Most workflows don't. But if you want to re-run notebook 01 (data acquisition) or notebook 03 (geospatial fusion) from scratch, ask the project owner to build a `--full` bundle:

```bash
# project owner only
python scripts/make_share_bundle.py --full     # ~5 GB
# upload share_bundle_full.zip to Drive, share new URL
```

Then teammates run `scripts/fetch_share_bundle.py` against the new URL.

## Building the data

```bash
python -m src.pipeline                   # download -> fuse -> features, skipping finished stages
python -m src.pipeline --from features   # rebuild the ABT from data/interim/fused_zip_hurricane.csv
python -m src.validation                 # check the result
```

The pipeline replaces running notebooks 01, 03 and 04 by hand. It writes `data/interim/fused_zip_hurricane.csv`, then `data/processed/abt_clean.csv` and `abt_dropped_rows.csv`, and refuses to write an ABT that fails validation (duplicate keys, ZIPs outside their record's state, non-positive population, wrong split labels, SVI outside [0, 1], severity classes that disagree with their bins, columns missing from `outputs/data_dictionary.csv`). Logs go to `data/interim/pipeline.log`.

### Two versions of the ABT

| file | what it is | used for |
|---|---|---|
| `data/processed/abt.csv` | the ABT as published (version 1, in the share bundle), 5,522 rows | every number in the paper; kept frozen |
| `data/processed/abt_clean.csv` | built by `src.pipeline`, 5,277 rows | the paper's "mis-keyed ZIPs removed" robustness row and future work |

A data audit found problems in version 1 that the pipeline now fixes:

- 245 rows (4.4%) carry a ZIP code outside the state of their FEMA record (e.g. New York ZIPs on Louisiana Ida records, Puerto Rico ZIPs on the Florida Irma declaration); 231 are Low-damage rows. They are set aside in `abt_dropped_rows.csv`.
- 251 rows hold the Census not-available code `-666666666` in `median_income`; it is now cleared and state-median imputed.
- 12 rows had `pct_mobile_homes` above 100% (population denominators in ZIPs with very few residents); shares are now capped at 100%.
- `tract_to_zip` counted a missing tract value as 0 while keeping its weight, and `impute_missing` zero-filled `snap_participation_pct`. Both are fixed in code; the first only affects tables rebuilt from the raw tract files.

On the corrected table XGBoost's TEST macro ROC-AUC is 0.778 and Severe recall 0.806 (published: 0.784 / 0.797). Logistic regression, which the income code had distorted, edges XGBoost on the development storms (0.697 vs 0.693, within the ±0.09 storm-to-storm spread). The paper reports the pre-specified analysis on version 1 and this check in Table VII.

## Reproducing the paper

Every table and figure in the paper comes from `scripts/camera_ready_experiments.py`. One command reruns it and checks each number against the committed `outputs/camera_ready/results.json`:

```bash
# needs data/processed/abt.csv (share bundle above); --fetch downloads the rest
export CENSUS_API_KEY="your-key"            # for the year-matched ACS tables
python -m src.pipeline --from features      # abt_clean.csv, for the `cleaned` stage
python scripts/reproduce_paper.py --fetch   # all stages, about 5 min after downloads
python scripts/reproduce_paper.py main loho # selected stages
python scripts/reproduce_paper.py --check-only
```

It checks that `abt.csv` is byte-for-byte the published ABT, validates `abt_clean.csv`, runs the stages, and exits non-zero if any value differs by more than 5e-4 (the paper reports at most three decimals; Random Forest's parallel training jitters some metrics around the sixth decimal). Stages map to the paper as follows:

| stage | paper |
|---|---|
| `main` | Tables III, V, VI, VIII, IX; Figs. 2, 4; calibration and Brier scores |
| `binning` | Table IV |
| `selection` | leave-one-storm-out model selection (Table V, "Dev") |
| `calibration` | Fig. 1, raw vs isotonic probabilities |
| `formulation` | multiclass vs ordinal vs regression |
| `loho`, `zip`, `ablation`, `acs` | Table VII |
| `cleaned` | Table VII "mis-keyed ZIPs removed" row, Section IV-B "Data cleaning" |
| `forecast` | Table VII forecast rows, Fig. 3 |
| `frpi` | Section IV-F (damage capture, food-need validation) |
| `figures` | Fig. 3 from results.json |

### Tests

```bash
pytest
```

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
│   ├── priority_index.py             ← fragility scalers, index computation, re-weighting
│   ├── validation.py                 ← ABT data checks + data-dictionary check
│   └── pipeline.py                   ← download -> fuse -> features, one command
├── scripts/
│   ├── fetch_camera_ready_data.py    ← FEMA food-need, year-matched ACS, tracks, ZCTAs
│   ├── camera_ready_experiments.py   ← every number in the paper
│   └── reproduce_paper.py            ← rerun + check against committed results
├── tests/                            ← pytest suite
├── ieee_paper.tex                    ← IEEE BigData 2026 paper
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
- **Weight on damage risk**: moves the FRPI between damage risk and food fragility (default 0.6); the table re-ranks and reports how many ZIPs are also in the default top N.
- **Minimum priority score**, **Top-N**, **Food deserts only**, **High-SVI (Q4) only**.
- **Forecast lead time**: shows the expected skill and ranking stability at 24, 48 and 72 h before landfall, from the forecast-error stress test. Rankings are most reliable within about 24 h.

The app renders the pre-generated folium choropleth (default weighting), a sortable table, per-ZIP SHAP attributions, and a CSV download button. Rankings are advisory; refresh them with each NHC advisory.

## Key results

Numbers from the camera-ready paper (`outputs/camera_ready/results.json`). Best classifier: **XGBoost**, chosen by leave-one-storm-out ROC-AUC on the nine TRAIN+VAL storms, with isotonic calibration fit on VAL.

### Classification on held-out TEST set (5 hurricanes, 2,587 zip × hurricane rows)
| metric | value |
|---|---|
| Accuracy | 0.57 |
| Weighted F1 | 0.54 |
| Macro ROC-AUC | **0.78** (storm-bootstrap 95% CI 0.71–0.83) |
| Severe-class recall | **0.80** |
| Leave-one-hurricane-out macro ROC-AUC (14 storms) | 0.76 ± 0.08 |

### Before landfall (forecast-quality storm inputs)
| lead time | macro ROC-AUC | Severe recall |
|---|---|---|
| observed track | 0.78 | 0.80 |
| 24 h | 0.73 | 0.69 |
| 48 h | 0.63 | 0.51 |
| 72 h | 0.64 | 0.41 |
| no storm features | 0.56 | — |

### Per-hurricane regression (`verified_damage_per_1000`, exploratory)
| hurricane | RMSE | MAE | R² |
|---|---|---|---|
| Milton | 19.3 | 9.2 | **+0.35** |
| Ian | 48.9 | 20.6 | +0.17 |
| Idalia | 172.0 | 24.5 | +0.06 |
| Helene | 48.1 | 30.3 | -0.09 |
| Ida | 162.2 | 92.8 | -0.17 |

### Equity audit by SVI quartile (Fairlearn)
| quartile | accuracy | recall (Severe) | true Severe rate | predicted Severe rate |
|---|---|---|---|---|
| Q1 (low SVI) | 0.56 | 0.63 | 0.22 | 0.27 |
| Q2 | 0.56 | 0.80 | 0.23 | 0.39 |
| Q3 | 0.54 | 0.83 | 0.32 | 0.52 |
| **Q4 (high SVI)** | **0.64** | **0.87** | 0.34 | 0.51 |

Severe recall rises with social vulnerability, but the predicted Severe rate also outpaces the true rate, and FEMA records may reflect unequal registration, so this does not by itself establish equitable allocation.

### Food Relief Priority Index
- Within each TEST storm, 92% of the FRPI top decile are ZIPs with a High or Severe FEMA inspection rate (per-storm base rate 50%); 55% of Severe ZIPs fall in the top quartile.
- FRPI rank correlation with applicant-reported food need: ρ = 0.67 (that count largely tracks damage, ρ = 0.94).
- Top 50 at 0.6/0.4: 52% food deserts, 75th SVI percentile, mean damage 73 per 1,000 (TEST mean 34.5). With uncalibrated probabilities the top 50 were 100% food deserts; calibration removed that artifact.

### Cluster ablation (K-Means)
With/without the unsupervised cluster_label feature: ≤0.013 weighted-F1 difference across all five classifiers (XGBoost is marginally *better* without it). Conclusion: SVI + demographic features already capture the vulnerability profile that K-Means recovers; the cluster feature is redundant.

## Critical implementation notes (the traps)

1. **imblearn.pipeline.Pipeline** with SMOTE — never sklearn's (which leaks into predict()).
2. **pyproj.Geod** for all lat/lon distances. Euclidean on degrees is a bug.
3. **EPSG:5070** (Albers Equal Area) for every area calculation; EPSG:4326 for lat/lon joins.
4. CDC SVI uses **-999** as a missing-data sentinel — replace with NaN before any aggregation.
5. FEMA API caps responses at 10,000 records — paginate by keyset (`id gt <last id>`, ordered by id), not deep `$skip` offsets, which time out on the large IHP tables; retry with backoff.
6. Housing Assistance Owners + Renters are aggregated to (disaster, zip) **before** the join, then summed across populations — to avoid Cartesian explosion when zips span multiple counties.
7. `StratifiedGroupKFold` groups are passed to `.fit()`, not the constructor; n_splits is capped at the number of distinct groups.
8. Food-fragility MinMaxScalers are fit on TRAIN only, then applied to val/test.
9. NRI hazard scores arrive at the census-tract level (zip-level was discontinued in 2025) and are aggregated to ZIP via the HUD crosswalk, same as SVI/Food Atlas.
10. Continuous regression target is right-tailed; we predict `log1p(damage)` and invert with `expm1` at predict time.
11. All `random_state=42`.

## License & attribution

This is a course project. Data is used under the public-domain / open-data licenses of the respective providers (FEMA, USDA, CDC, Census, NOAA, HUD).
