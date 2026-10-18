# explainability.py
# SHAP explainability + fairness audit for Kenya Hospital Readmission Predictor
# ZENIK.AI — Computational Epidemiology Portfolio

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

print("Loading model and data...")

with open("outputs/best_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
model_name = bundle["model_name"]
feature_cols = bundle["features"]
scaler = bundle["scaler"]

import sys
sys.path.insert(0, ".")
from src.data_generator import generate_readmission_dataset

df = generate_readmission_dataset(n=2000)

# Keep only columns present in the dataframe
feature_cols = [c for c in feature_cols if c in df.columns]

X = scaler.transform(df[feature_cols].values)
y = df["readmitted_30d"].values

os.makedirs("outputs", exist_ok=True)

print(f"Model: {model_name}")
print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# ── SHAP ──────────────────────────────────────────────────────────────────────

print("\nComputing SHAP values...")

if isinstance(model, LogisticRegression):
    background = shap.maskers.Independent(X, max_samples=100)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X[:500])
    X_plot = X[:500]
elif isinstance(model, XGBClassifier):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    X_plot = X
else:
    background = shap.sample(X, 50, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X[:200])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    X_plot = X[:200]

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"SHAP values shape: {shap_values.shape}")

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_plot, feature_names=feature_cols, show=False, max_display=15)
plt.title("SHAP Feature Importance — Kenya Hospital Readmission", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/shap_summary.png")

# Bar plot
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_shap})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#e63946", height=0.65)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Top Predictors of 30-day Readmission\n(Kenya Synthetic Cohort)", fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/shap_bar.png")

# ── Fairness Audit ─────────────────────────────────────────────────────────────

print("\nRunning fairness audit...")

from sklearn.metrics import roc_auc_score

df_eval = df.copy().reset_index(drop=True)
probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)
df_eval["pred_prob"] = probs
df_eval["pred_label"] = preds

results = []

# Sex
sex_col = "sex_male" if "sex_male" in df_eval.columns else "sex"
for grp, label in [(0, "Female"), (1, "Male")]:
    sub = df_eval[df_eval[sex_col] == grp]
    if len(sub) < 20: continue
    results.append({"group": "Sex", "subgroup": label, "n": len(sub),
        "actual_readmission_rate": round(sub["readmitted_30d"].mean(), 3),
        "predicted_positive_rate": round(sub["pred_label"].mean(), 3),
        "auc": round(roc_auc_score(sub["readmitted_30d"], sub["pred_prob"]), 3)})

# Age bands
age_col = "age" if "age" in df_eval.columns else None
if age_col:
    df_eval["age_band"] = pd.cut(df_eval[age_col], bins=[0, 30, 50, 70, 100],
                                  labels=["<30", "30-50", "50-70", "70+"])
    for band in ["<30", "30-50", "50-70", "70+"]:
        sub = df_eval[df_eval["age_band"] == band]
        if len(sub) < 20: continue
        results.append({"group": "Age Band", "subgroup": band, "n": len(sub),
            "actual_readmission_rate": round(sub["readmitted_30d"].mean(), 3),
            "predicted_positive_rate": round(sub["pred_label"].mean(), 3),
            "auc": round(roc_auc_score(sub["readmitted_30d"], sub["pred_prob"]), 3)})

# Rural vs Urban
if "rural_residence" in df_eval.columns:
    for grp, label in [(0, "Urban"), (1, "Rural")]:
        sub = df_eval[df_eval["rural_residence"] == grp]
        if len(sub) < 20: continue
        results.append({"group": "Residence", "subgroup": label, "n": len(sub),
            "actual_readmission_rate": round(sub["readmitted_30d"].mean(), 3),
            "predicted_positive_rate": round(sub["pred_label"].mean(), 3),
            "auc": round(roc_auc_score(sub["readmitted_30d"], sub["pred_prob"]), 3)})

# Insurance
if "nhif_insured" in df_eval.columns:
    for grp, label in [(0, "Uninsured"), (1, "NHIF Insured")]:
        sub = df_eval[df_eval["nhif_insured"] == grp]
        if len(sub) < 20: continue
        results.append({"group": "Insurance", "subgroup": label, "n": len(sub),
            "actual_readmission_rate": round(sub["readmitted_30d"].mean(), 3),
            "predicted_positive_rate": round(sub["pred_label"].mean(), 3),
            "auc": round(roc_auc_score(sub["readmitted_30d"], sub["pred_prob"]), 3)})

fairness_df = pd.DataFrame(results)
print("\nFairness Audit Results:")
print(fairness_df.to_string(index=False))
fairness_df.to_csv("outputs/fairness_audit.csv", index=False)
print("Saved: outputs/fairness_audit.csv")

# Fairness plot
colors = ["#457b9d", "#e63946", "#2a9d8f", "#f4a261",
          "#6a4c93", "#1982c4", "#ff595e", "#8ac926"]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
for i, row in fairness_df.iterrows():
    ax.barh(f"{row['group']}: {row['subgroup']}", row["auc"],
            color=colors[i % len(colors)], height=0.6)
ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
ax.set_xlabel("AUC", fontsize=11)
ax.set_title("AUC by Subgroup", fontsize=12)
ax.set_xlim(0, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9)

ax = axes[1]
x = range(len(fairness_df))
width = 0.35
ax.bar([i - width/2 for i in x], fairness_df["actual_readmission_rate"],
       width, label="Actual Rate", color="#457b9d")
ax.bar([i + width/2 for i in x], fairness_df["predicted_positive_rate"],
       width, label="Predicted Rate", color="#e63946", alpha=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels([r["subgroup"] for _, r in fairness_df.iterrows()],
                   rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Rate", fontsize=11)
ax.set_title("Actual vs Predicted Readmission Rate\nby Subgroup", fontsize=12)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.suptitle("Fairness Audit — Kenya Hospital Readmission Model", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("outputs/fairness_audit.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/fairness_audit.png")

print("\nDone. Outputs: shap_summary.png, shap_bar.png, fairness_audit.csv, fairness_audit.png")