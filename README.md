# Hospital Readmission Predictor

A machine learning system that predicts 30-day hospital readmission risk for Kenyan patients, with SHAP explainability and demographic fairness auditing.

---

## Live App

[hospital-readmission-ra8ibuwbegtmxypv76leik.streamlit.app](https://hospital-readmission-ra8ibuwbegtmxypv76leik.streamlit.app)

---

## Overview

Hospital readmission within 30 days is a key quality indicator in healthcare systems. This project builds a logistic regression classifier trained on a synthetic Kenya-context cohort, with full explainability and fairness auditing across demographic subgroups relevant to the Kenyan health system (sex, age band, rural/urban residence, NHIF insurance status).

---

## Features

- **Prediction UI** — Enter patient demographics, clinical parameters, and lab results to get a real-time readmission probability with risk category
- **Per-patient SHAP** — Feature contribution bar chart for every individual prediction
- **Global SHAP** — Summary and bar plots across 500 patients showing top predictors
- **Fairness Audit** — AUC parity analysis across Sex, Age Band, Residence, and Insurance subgroups

---

## Model

| Metric | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Validation AUC | 0.741 |
| Features | 18 |
| Training cohort | 2,000 synthetic patients |

**Top predictors (by mean SHAP):** prior admissions, length of stay, age, creatinine, HbA1c

---

## Fairness Audit Results

| Group | Subgroup | AUC |
|-------|----------|-----|
| Sex | Female | 0.752 |
| Sex | Male | 0.779 |
| Age Band | <30 | 0.804 |
| Age Band | 30-50 | 0.729 |
| Age Band | 50-70 | 0.740 |
| Age Band | 70+ | 0.747 |
| Residence | Urban | 0.715 |
| Residence | Rural | 0.767 |
| Insurance | Uninsured | 0.755 |
| Insurance | NHIF Insured | 0.771 |

AUC gap: 0.089 (within acceptable 0.10 threshold)

---

## Project Structure

hospital-readmission/
├── app.py # Streamlit app
├── requirements.txt
├── src/
│ ├── data_generator.py # Synthetic Kenya cohort generator
│ ├── model.py # Model training and evaluation
│ └── explainability.py # SHAP + fairness audit
├── outputs/
│ ├── best_model.pkl
│ ├── shap_summary.png
│ ├── shap_bar.png
│ ├── fairness_audit.csv
│ └── fairness_audit.png
└── data/
└── readmission_dataset.csv


---

## Setup

```bash
conda create -n readmission python=3.11
conda activate readmission
pip install -r requirements.txt
python src/model.py
python src/explainability.py
streamlit run app.py
```

---

## Context

Built as part of the ZENIK.AI 12-project Healthcare AI portfolio. Synthetic data modelled on Kenya-specific health system parameters including NHIF insurance coverage, rural/urban distribution, and disease burden patterns from KNBS and MOH Kenya data.

**Author:** Amos Mwaura | [github.com/Mwauramos](https://github.com/Mwauramos)
