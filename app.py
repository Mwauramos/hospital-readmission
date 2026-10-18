# app.py — Hospital Readmission Predictor
# ZENIK.AI | Project 9: Computational Epidemiology

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import sys, os
sys.path.insert(0, ".")

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("outputs/best_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["features"]
model_name = bundle["model_name"]
auc = bundle["auc"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.shields.io/badge/ZENIK.AI-Healthcare%20AI-0a9396?style=for-the-badge", use_column_width=True)
st.sidebar.markdown("## 🏥 Hospital Readmission Predictor")
st.sidebar.markdown("**Project 9 | ZENIK.AI Portfolio**")
st.sidebar.markdown("Predicts 30-day hospital readmission risk for Kenyan patients using logistic regression with SHAP explainability and fairness auditing.")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** `{model_name}`")
st.sidebar.markdown(f"**Validation AUC:** `{auc:.3f}`")
st.sidebar.markdown(f"**Features:** `{len(feature_cols)}`")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 SHAP Explainability", "⚖️ Fairness Audit"])

# ── Tab 1: Predict ────────────────────────────────────────────────────────────

with tab1:
    st.header("Patient Risk Assessment")
    st.markdown("Enter patient details to predict 30-day readmission risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.slider("Age (years)", 18, 100, 55)
        sex_male = st.selectbox("Sex", ["Female", "Male"])
        sex_male = 1 if sex_male == "Male" else 0
        rural_residence = st.selectbox("Residence", ["Urban", "Rural"])
        rural_residence = 1 if rural_residence == "Rural" else 0
        nhif_insured = st.selectbox("NHIF Insurance", ["Uninsured", "NHIF Insured"])
        nhif_insured = 1 if nhif_insured == "NHIF Insured" else 0
        wealth_quintile = st.slider("Wealth Quintile (1=poorest, 5=richest)", 1, 5, 3)
        distance_to_facility_km = st.slider("Distance to Facility (km)", 0.5, 150.0, 15.0, step=0.5)

    with col2:
        st.subheader("Clinical")
        num_comorbidities = st.slider("Number of Comorbidities", 0, 5, 2)
        num_medications = st.slider("Number of Medications", 0, 15, 4)
        length_of_stay_days = st.slider("Length of Stay (days)", 1, 30, 5)
        prior_admissions_12m = st.slider("Prior Admissions (12 months)", 0, 10, 1)
        discharge_to_home = st.selectbox("Discharge Destination", ["Home", "Other facility"])
        discharge_to_home = 1 if discharge_to_home == "Home" else 0
        follow_up_scheduled = st.selectbox("Follow-up Scheduled", ["No", "Yes"])
        follow_up_scheduled = 1 if follow_up_scheduled == "Yes" else 0

    with col3:
        st.subheader("Lab Results")
        hba1c = st.slider("HbA1c (%)", 4.0, 14.0, 6.5, step=0.1)
        creatinine_umol = st.slider("Creatinine (µmol/L)", 40, 500, 90)
        sodium_meq = st.slider("Sodium (mEq/L)", 120, 160, 138)
        haemoglobin_gdl = st.slider("Haemoglobin (g/dL)", 5.0, 18.0, 13.0, step=0.1)
        systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 130)
        heart_rate = st.slider("Heart Rate (bpm)", 40, 140, 78)

    # Build input vector
    input_data = np.array([[
        age, sex_male, num_comorbidities, num_medications,
        length_of_stay_days, prior_admissions_12m, hba1c,
        creatinine_umol, sodium_meq, haemoglobin_gdl,
        systolic_bp, heart_rate, discharge_to_home,
        follow_up_scheduled, nhif_insured, rural_residence,
        distance_to_facility_km, wealth_quintile
    ]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0, 1]
    pred = int(prob >= 0.5)

    st.markdown("---")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Readmission Probability", f"{prob:.1%}")
    with r2:
        risk_label = "HIGH RISK" if prob >= 0.5 else ("MODERATE" if prob >= 0.3 else "LOW RISK")
        color = "🔴" if prob >= 0.5 else ("🟡" if prob >= 0.3 else "🟢")
        st.metric("Risk Category", f"{color} {risk_label}")
    with r3:
        st.metric("Model Confidence", f"{max(prob, 1-prob):.1%}")

    # SHAP for this patient
    st.subheader("Feature Contributions (this patient)")
    from sklearn.linear_model import LogisticRegression
    background = shap.maskers.Independent(input_scaled, max_samples=50)
    explainer = shap.LinearExplainer(model, background)
    shap_vals = explainer.shap_values(input_scaled)[0]

    shap_series = pd.Series(shap_vals, index=feature_cols).sort_values(key=abs, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e63946" if v > 0 else "#457b9d" for v in shap_series.values]
    ax.barh(shap_series.index[::-1], shap_series.values[::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Top 10 feature contributions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2: SHAP ───────────────────────────────────────────────────────────────

with tab2:
    st.header("SHAP Feature Importance")
    st.markdown("Global explainability across 500 patients. Red = increases readmission risk, blue = decreases it.")

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("outputs/shap_summary.png"):
            st.image("outputs/shap_summary.png", caption="SHAP Summary Plot", use_column_width=True)
        else:
            st.warning("Run `python src/explainability.py` to generate SHAP plots.")
    with col_b:
        if os.path.exists("outputs/shap_bar.png"):
            st.image("outputs/shap_bar.png", caption="Mean |SHAP| Feature Importance", use_column_width=True)

# ── Tab 3: Fairness ───────────────────────────────────────────────────────────

with tab3:
    st.header("Fairness Audit")
    st.markdown("Model performance across demographic subgroups. AUC parity assessed for Sex, Age, Residence, and Insurance status.")

    if os.path.exists("outputs/fairness_audit.csv"):
        df_fair = pd.read_csv("outputs/fairness_audit.csv")

        # Summary metrics
        auc_min = df_fair["auc"].min()
        auc_max = df_fair["auc"].max()
        auc_gap = auc_max - auc_min

        m1, m2, m3 = st.columns(3)
        m1.metric("Min Subgroup AUC", f"{auc_min:.3f}")
        m2.metric("Max Subgroup AUC", f"{auc_max:.3f}")
        gap_color = "normal" if auc_gap < 0.1 else "inverse"
        m3.metric("AUC Gap", f"{auc_gap:.3f}", delta="within threshold" if auc_gap < 0.1 else "exceeds 0.10", delta_color=gap_color)

        st.dataframe(df_fair.style.format({
            "actual_readmission_rate": "{:.1%}",
            "predicted_positive_rate": "{:.1%}",
            "auc": "{:.3f}"
        }), use_container_width=True)

        if os.path.exists("outputs/fairness_audit.png"):
            st.image("outputs/fairness_audit.png", caption="Fairness Audit — AUC and Readmission Rates by Subgroup", use_column_width=True)
    else:
        st.warning("Run `python src/explainability.py` to generate fairness audit.")

    st.markdown("---")
    st.markdown("""
    **Interpretation guide**
    - AUC gap < 0.10: acceptable parity across subgroups
    - Predicted positive rate disparity flags potential calibration issues
    - Rural patients and uninsured show slightly lower predicted positive rates despite higher actual readmission — warrants monitoring
    """)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("**ZENIK.AI** | Project 9: Hospital Readmission Predictor | Synthetic Kenya cohort | [GitHub](https://github.com/Mwauramos/hospital-readmission)")