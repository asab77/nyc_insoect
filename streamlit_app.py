import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
DATA_PATH = Path("data_set/nyc_inspections_reduced.csv")
MODEL_PATH = Path("model/model.pkl")
METRICS_PATH = Path("reports/metrics.json")

FEATURES = [
    "prev_score", "hist_crit_rate_w3", "hist_visits_w3", "month", "year",
    "CUISINE DESCRIPTION", "BORO", "GRADE", "ZIPCODE"
]

st.set_page_config(page_title="InspectNYC — Critical Violation Risk", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def add_history_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild features to match training-time logic."""
    df = df.copy()
    df["INSPECTION DATE"] = pd.to_datetime(df["INSPECTION DATE"], errors="coerce")
    df = df.dropna(subset=["INSPECTION DATE", "CAMIS"])
    df = df.sort_values(["CAMIS", "INSPECTION DATE"])

    # current row flags (used to build rolling history)
    df["prev_is_critical"] = (df["CRITICAL FLAG"] == "Critical").astype(int)
    # rolling stats within each restaurant
    def _add_roll(g):
        g = g.sort_values("INSPECTION DATE")
        g["hist_crit_rate_w3"] = g["prev_is_critical"].rolling(3, min_periods=1).mean().shift(1)
        g["hist_visits_w3"] = np.arange(len(g))
        return g

    df = df.groupby("CAMIS", group_keys=False).apply(_add_roll)

    # score, time features
    df["prev_score"] = pd.to_numeric(df["SCORE"], errors="coerce")
    df["month"] = df["INSPECTION DATE"].dt.month
    df["year"] = df["INSPECTION DATE"].dt.year

    # tidy
    keep_cols = list(set(FEATURES + ["CAMIS", "DBA", "INSPECTION DATE"]))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["prev_score", "hist_crit_rate_w3", "month", "year"])

    return df

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None

def latest_snapshot_for_scoring(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Take the most recent inspection per restaurant as the basis for 'next inspection' prediction."""
    idx = df_feat.groupby("CAMIS")["INSPECTION DATE"].idxmax()
    snap = df_feat.loc[idx].copy()
    return snap

# ----------------------------
# App
# ----------------------------
st.title("InspectNYC — Predicting Critical Health Violations")
st.write("Rank restaurants by risk so inspectors can prioritize the next visit.")

# File checks
missing = []
if not DATA_PATH.exists(): missing.append(str(DATA_PATH))
if not MODEL_PATH.exists(): missing.append(str(MODEL_PATH))
if missing:
    st.error(f"Missing required files: {', '.join(missing)}")
    st.stop()

with st.spinner("Loading data and model..."):
    raw = load_data()
    model = load_model()
    metrics = load_metrics()

# Build features
with st.spinner("Building features..."):
    feat = add_history_and_features(raw)
    snap = latest_snapshot_for_scoring(feat)

# Sidebar controls
st.sidebar.header("Filters")
boroughs = ["All"] + sorted([b for b in snap["BORO"].dropna().unique()])
boro_choice = st.sidebar.selectbox("Borough", boroughs, index=0)

# limit cuisines to common ones for the UI
top_cuisines = (
    snap["CUISINE DESCRIPTION"]
    .value_counts()
    .head(30)
    .index.tolist()
)
cuisine_filter = st.sidebar.multiselect("Cuisine (top 30 by count)", top_cuisines, default=[])

zip_filter = st.sidebar.text_input("ZIP code (exact match or comma-separated)", "")

threshold = st.sidebar.slider("Risk threshold (probability)", 0.05, 0.95, 0.35, 0.01)
top_n = st.sidebar.number_input("Top N to display", min_value=10, max_value=1000, value=200, step=10)

# Apply filters
view = snap.copy()
if boro_choice != "All":
    view = view[view["BORO"] == boro_choice]
if cuisine_filter:
    view = view[view["CUISINE DESCRIPTION"].isin(cuisine_filter)]
if zip_filter.strip():
    zips = [z.strip() for z in zip_filter.split(",") if z.strip()]
    view = view[view["ZIPCODE"].astype(str).isin(zips)]

if view.empty:
    st.warning("No rows after filters. Relax filters or pick a different borough/cuisine.")
    st.stop()

# Score
X = view[FEATURES].copy()
proba = model.predict_proba(X)[:, 1]
view = view.assign(risk=proba)
view = view.sort_values("risk", ascending=False)

# Headline metrics
col1, col2 = st.columns(2)
col1.metric("Restaurants scored", f"{len(view):,}")
if metrics:
    col2.metric("Model ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
else:
    col2.write("No metrics.json found.")

# Table
st.subheader("Highest-risk venues")
risk_cut = view[view["risk"] >= threshold]
table = risk_cut.head(top_n).copy()
pretty_cols = ["CAMIS", "DBA", "BORO", "ZIPCODE", "CUISINE DESCRIPTION", "INSPECTION DATE", "risk"]
table = table[pretty_cols].rename(columns={
    "DBA": "Name",
    "BORO": "Borough",
    "ZIPCODE": "ZIP",
    "CUISINE DESCRIPTION": "Cuisine",
    "INSPECTION DATE": "Last Inspection",
    "risk": "Predicted Risk"
})
table["Predicted Risk"] = table["Predicted Risk"].round(3)

st.dataframe(table, use_container_width=True, height=480)

# Download
csv_bytes = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download table as CSV",
    data=csv_bytes,
    file_name="inspectnyc_high_risk.csv",
    mime="text/csv"
)

# Notes
with st.expander("Notes"):
    st.markdown(
        """
- **Prediction target:** “Will the **next** inspection be critical?”
- Scores reflect the model’s estimated probability based on a restaurant’s history, geography, cuisine, and seasonality.
- Use a **lower threshold** for high-recall (catch more risky venues) or a **higher threshold** for high precision.
        """
    )
