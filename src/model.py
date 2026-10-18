# model.py
# XGBoost and Logistic Regression for 30-day readmission prediction
# ZENIK.AI — Computational Epidemiology Portfolio

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                              roc_curve, confusion_matrix)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES = [
    "age", "sex_male", "num_comorbidities", "num_medications",
    "length_of_stay_days", "prior_admissions_12m", "hba1c",
    "creatinine_umol", "sodium_meq", "haemoglobin_gdl",
    "systolic_bp", "heart_rate", "discharge_to_home",
    "follow_up_scheduled", "nhif_insured", "rural_residence",
    "distance_to_facility_km", "wealth_quintile"
]

TARGET = "readmitted_30d"


def preprocess(df: pd.DataFrame):
    """Prepare dataset for modelling."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"Training: {len(X_train_bal)} samples (SMOTE balanced)")
    print(f"Test: {len(X_test)} samples")
    print(f"Readmission rate in test: {y_test.mean():.1%}")

    return X_train_bal, X_test, y_train_bal, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Train Logistic Regression and XGBoost models."""
    results = {}

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr.fit(X_train_sc, y_train)
    lr_probs = lr.predict_proba(X_test_sc)[:, 1]
    lr_preds = lr.predict(X_test_sc)
    lr_auc = roc_auc_score(y_test, lr_probs)
    print(f"Logistic Regression AUC: {lr_auc:.3f}")
    print(classification_report(y_test, lr_preds,
          target_names=["Not Readmitted", "Readmitted"]))

    results["logistic_regression"] = {
        "model": lr, "scaler": scaler,
        "auc": lr_auc, "probs": lr_probs
    }

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    print(f"XGBoost AUC: {xgb_auc:.3f}")
    print(classification_report(y_test, xgb_preds,
          target_names=["Not Readmitted", "Readmitted"]))

    results["xgboost"] = {
        "model": xgb_model, "scaler": None,
        "auc": xgb_auc, "probs": xgb_probs
    }

    return results, y_test, X_test


def plot_roc_curves(results, y_test, output_dir="outputs"):
    """Plot ROC curves for all models."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    colors = {"logistic_regression": "#1B4332", "xgboost": "#BC4A1A"}
    labels = {"logistic_regression": "Logistic Regression",
              "xgboost": "XGBoost"}

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["probs"])
        plt.plot(fpr, tpr, label=f"{labels[name]} (AUC={result['auc']:.3f})",
                 color=colors[name], linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — 30-Day Hospital Readmission\nKenya Patient Data")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved: {path}")


def save_best_model(results, output_dir="outputs"):
    """Save the best performing model."""
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "model": best["model"],
            "scaler": best["scaler"],
            "model_name": best_name,
            "auc": best["auc"],
            "features": FEATURES
        }, f)
    print(f"Best model ({best_name}, AUC={best['auc']:.3f}) saved: {path}")
    return best_name


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("data/readmission_dataset.csv")

    X_train, X_test, y_train, y_test = preprocess(df)
    results, y_test, X_test = train_models(X_train, X_test, y_train, y_test)
    plot_roc_curves(results, y_test)
    best = save_best_model(results)
    print(f"\nModel training complete. Best model: {best}")