# nyc_inspect — Predicting Critical Health Violations 🧪🚨

**Live App:** https://nycinsoect-ahjagvrzt9x4kovkd8pzmt.streamlit.app/
**Dataset:** NYC DOHMH Restaurant Inspection Results (22MB, reduced version)  
**Goal:** Predict whether a restaurant’s **next** inspection will result in a critical violation.

---

## 🔍 Overview

This project analyzes NYC restaurant inspection history to estimate the risk of a future **critical health violation**, helping inspectors prioritize which venues to check first.

Instead of building a generic “ML classification example,” this project solves a real-world problem:

> _“Given a restaurant’s past inspections, cuisine, location, and seasonal patterns, can we estimate the chance that their next inspection will be critical?”_

The final output is a **Streamlit web app** that:
- Scores restaurants by risk
- Lets users filter by borough, cuisine, and ZIP code
- Shows the most recent inspection record + predicted probability
- Allows CSV download of high-risk locations

---

## 🛠 Features

| Area | What was done |
|------|---------------|
| Data Engineering | Reduced raw dataset from ~125MB to ~22MB, cleaned inconsistent formats, handled missing values |
| Feature Engineering | Rolling inspection history (last 3 visits), risk rate, time-based features (month/year), location/cuisine signals |
| Model | Logistic Regression inside a scikit-learn pipeline (OneHotEncoder + SimpleImputer) |
| Evaluation | Time-aware split (train on older inspections → test on newer ones) |
| Deployment | Streamlit + pinned environment for reproducibility |

---

## 📊 Model Results

Metrics computed from the time-aware test set:

| Metric | Value |
|--------|-------|
| ROC AUC | `0.XX` |
| PR AUC | `0.XX` |
| Best threshold found | `0.35` (recall-oriented) |

> The model favors **recall** to avoid missing risky restaurants.

---

## 🚀 Demo: Try It Yourself

To run the Streamlit app locally:

```bash
git clone https://github.com/<your-username>/nyc_inspect.git
cd InspectNYC
pip install -r requirements.txt
streamlit run app/inspect_nyc.py   # or streamlit_app.py depending on your entry point
