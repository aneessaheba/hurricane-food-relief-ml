"""
Food Relief Priority Dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py

Expects the following precomputed artifacts (produced by notebooks 06-08):
    outputs/priority_rankings.csv
    outputs/shap_values.npy
    models/best_classifier.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DATA_PATHS, TARGET_COL, TARGET_CLASS_COL  # noqa: E402

OUT = DATA_PATHS["outputs"]
MODELS = DATA_PATHS["models"]

st.set_page_config(page_title="Hurricane Food Relief Priority",
                   layout="wide", initial_sidebar_state="expanded")


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_rankings() -> pd.DataFrame:
    df = pd.read_csv(OUT / "priority_rankings.csv", dtype={"zip_code": str})
    df["zip_code"] = df["zip_code"].str.zfill(5)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODELS / "best_classifier.pkl")


@st.cache_data
def load_shap():
    path = OUT / "shap_values.npy"
    if path.exists():
        return np.load(path)
    return None


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Hurricane Food Relief Priority Dashboard")
st.caption("Predict damage severity before landfall, rank zip codes by food-relief "
           "urgency. DATA 245 · Group 2 · SJSU.")

try:
    df = load_rankings()
except FileNotFoundError:
    st.error("Priority rankings not found. Run notebook 08 first.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")
hurricane = st.sidebar.selectbox(
    "Hurricane", sorted(df["hurricane_name"].unique()),
)
min_score = st.sidebar.slider("Minimum priority score", 0.0, 1.0, 0.0, 0.05)
top_n = st.sidebar.slider("Top-N zip codes", 10, 100, 25, step=5)
only_desert = st.sidebar.checkbox("Food deserts only")
only_q4 = st.sidebar.checkbox("High-SVI (Q4) only")

sub = df[df["hurricane_name"] == hurricane].copy()
if "svi_overall" in sub.columns:
    sub["svi_q"] = pd.qcut(sub["svi_overall"], 4,
                           labels=["Q1", "Q2", "Q3", "Q4"])
sub = sub[sub["priority_index_norm"] >= min_score]
if only_desert and "food_desert_flag" in sub.columns:
    sub = sub[sub["food_desert_flag"] == 1]
if only_q4 and "svi_q" in sub.columns:
    sub = sub[sub["svi_q"] == "Q4"]
sub = sub.sort_values("priority_index_norm", ascending=False).head(top_n)

# -----------------------------------------------------------------------------
# KPI row
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Zip codes", f"{len(sub):,}")
c2.metric("Predicted High/Severe",
          int((sub.get("prob_high_or_severe", pd.Series([0]*len(sub))) > 0.5).sum()))
c3.metric("Food deserts",
          int(sub.get("food_desert_flag", pd.Series([0]*len(sub))).sum()))
c4.metric("Mean priority", f"{sub['priority_index_norm'].mean():.3f}")
if "svi_overall" in sub.columns:
    c5.metric("Mean SVI", f"{sub['svi_overall'].mean():.2f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Map
# -----------------------------------------------------------------------------
map_col, table_col = st.columns([3, 2])

with map_col:
    st.subheader("Priority map")
    map_file = OUT / f"priority_map_{hurricane.lower()}.html"
    if map_file.exists():
        html = map_file.read_text(encoding="utf-8")
        st.components.v1.html(html, height=520)
    else:
        st.info(f"No precomputed map for {hurricane} — run notebook 08 first.")

with table_col:
    st.subheader("Top priority zip codes")
    show_cols = [c for c in [
        "priority_rank", "zip_code", "state", "hurricane_name",
        "priority_index_norm", "prob_high_or_severe",
        "food_fragility_score", "svi_overall", TARGET_COL, TARGET_CLASS_COL,
    ] if c in sub.columns]
    st.dataframe(sub[show_cols], use_container_width=True, height=520)

    csv = sub[show_cols].to_csv(index=False).encode()
    st.download_button(
        "Download filtered CSV", csv,
        file_name=f"priority_{hurricane}.csv", mime="text/csv",
    )

# -----------------------------------------------------------------------------
# SHAP panel
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Per-zip SHAP explanation")
sv = load_shap()
if sv is None:
    st.info("SHAP values not precomputed. Run notebook 07.")
else:
    zip_pick = st.selectbox("Pick a zip from the table above",
                            sub["zip_code"].tolist())
    if zip_pick:
        row = df[df["zip_code"] == zip_pick].iloc[0]
        # Severe class SHAP (last axis, last class by encoder = 'Severe' alpha order)
        idx = df.index.get_loc(row.name) if row.name in df.index else 0
        try:
            severe_class = -1  # last class by label-encoder alpha order = Severe
            vals = sv[idx, :, severe_class]
            try:
                model_pkg = load_model()
                pre = model_pkg["pipe"].named_steps["pre"]
                feat_names = list(pre.get_feature_names_out())
            except Exception:
                feat_names = [f"f{i}" for i in range(vals.shape[0])]
            contrib = (pd.DataFrame({"feature": feat_names, "shap": vals})
                       .assign(abs=lambda d: d["shap"].abs())
                       .sort_values("abs", ascending=False).head(5))
            st.write(f"Top 5 features driving Severe-class prediction for **{zip_pick}**:")
            st.dataframe(contrib[["feature", "shap"]], use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP lookup failed: {e}")
