# data_generator.py
# Synthetic hospital readmission dataset for Kenya context
# Based on published readmission research and Kenya health system structure
# ZENIK.AI — Computational Epidemiology Portfolio

import numpy as np
import pandas as pd

np.random.seed(42)

KENYA_COUNTIES = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Kisii", "Kakamega", "Machakos", "Nyeri", "Meru"
]

DIAGNOSES = [
    "Heart Failure", "Diabetes Mellitus", "Pneumonia",
    "Hypertension", "COPD", "Malaria", "TB", "Renal Failure",
    "Stroke", "Sepsis"
]

FACILITY_LEVELS = ["County Hospital", "Sub-County Hospital", "National Referral"]


def generate_readmission_dataset(n=2000):
    """
    Generate synthetic 30-day readmission dataset for Kenya.
    Features based on published readmission risk factors.
    """
    counties = np.random.choice(KENYA_COUNTIES, size=n)
    ages = np.random.normal(52, 18, size=n).clip(18, 90).astype(int)
    sex = np.random.choice([0, 1], size=n, p=[0.48, 0.52])
    diagnoses = np.random.choice(DIAGNOSES, size=n)
    facility = np.random.choice(FACILITY_LEVELS, size=n,
                                 p=[0.55, 0.35, 0.10])

    # Clinical features
    num_comorbidities = np.random.poisson(2.1, n).clip(0, 8)
    num_medications = np.random.poisson(4.5, n).clip(0, 15)
    length_of_stay = np.random.exponential(5, n).clip(1, 45).astype(int)
    prior_admissions = np.random.poisson(1.2, n).clip(0, 8)
    hba1c = np.random.normal(7.8, 2.1, n).clip(4.0, 15.0)
    creatinine = np.random.exponential(1.2, n).clip(0.5, 12.0)
    sodium = np.random.normal(138, 5, n).clip(120, 155)
    haemoglobin = np.random.normal(11.2, 2.8, n).clip(4.0, 18.0)
    systolic_bp = np.random.normal(138, 22, n).clip(80, 220)
    heart_rate = np.random.normal(88, 18, n).clip(45, 160)
    discharge_to_home = np.random.binomial(1, 0.72, n)
    follow_up_scheduled = np.random.binomial(1, 0.58, n)
    insurance_nhif = np.random.binomial(1, 0.45, n)
    rural_residence = np.array([
        1 if c not in ["Nairobi", "Mombasa", "Kisumu"] else 0
        for c in counties
    ])
    distance_to_facility_km = np.where(
        rural_residence,
        np.random.exponential(25, n).clip(1, 150),
        np.random.exponential(8, n).clip(1, 40)
    )
    wealth_quintile = np.random.choice([1, 2, 3, 4, 5], size=n,
                                        p=[0.28, 0.25, 0.22, 0.15, 0.10])

    # 30-day readmission outcome
    log_odds = (
        -2.5
        + 0.03 * (ages - 50)
        + 0.4 * num_comorbidities
        + 0.15 * num_medications
        - 0.08 * length_of_stay
        + 0.5 * prior_admissions
        + 0.3 * (hba1c > 9).astype(int)
        + 0.4 * (creatinine > 2).astype(int)
        + 0.3 * (sodium < 135).astype(int)
        - 0.3 * (haemoglobin > 10).astype(int)
        + 0.2 * (systolic_bp > 160).astype(int)
        - 0.5 * discharge_to_home
        - 0.6 * follow_up_scheduled
        - 0.3 * insurance_nhif
        + 0.4 * rural_residence
        + 0.02 * distance_to_facility_km
        - 0.2 * (wealth_quintile >= 4).astype(int)
    )
    prob_readmission = 1 / (1 + np.exp(-log_odds))
    readmitted_30d = np.random.binomial(1, prob_readmission)

    df = pd.DataFrame({
        "patient_id": [f"KE{i:05d}" for i in range(n)],
        "county": counties,
        "age": ages,
        "sex_male": sex,
        "diagnosis": diagnoses,
        "facility_level": facility,
        "num_comorbidities": num_comorbidities,
        "num_medications": num_medications,
        "length_of_stay_days": length_of_stay,
        "prior_admissions_12m": prior_admissions,
        "hba1c": hba1c.round(1),
        "creatinine_umol": creatinine.round(2),
        "sodium_meq": sodium.round(1),
        "haemoglobin_gdl": haemoglobin.round(1),
        "systolic_bp": systolic_bp.round(0).astype(int),
        "heart_rate": heart_rate.round(0).astype(int),
        "discharge_to_home": discharge_to_home,
        "follow_up_scheduled": follow_up_scheduled,
        "nhif_insured": insurance_nhif,
        "rural_residence": rural_residence,
        "distance_to_facility_km": distance_to_facility_km.round(1),
        "wealth_quintile": wealth_quintile,
        "readmitted_30d": readmitted_30d
    })

    return df


if __name__ == "__main__":
    df = generate_readmission_dataset(2000)
    df.to_csv("data/readmission_dataset.csv", index=False)

    print(f"Dataset generated: {len(df)} patients")
    print(f"30-day readmission rate: {df['readmitted_30d'].mean():.1%}")
    print(f"Counties: {df['county'].nunique()}")
    print(f"Diagnoses: {df['diagnosis'].nunique()}")
    print(f"\nTop readmission risk factors (by correlation):")
    features = ["prior_admissions_12m", "num_comorbidities", "follow_up_scheduled",
                "rural_residence", "distance_to_facility_km", "nhif_insured"]
    for f in features:
        corr = df[f].corr(df["readmitted_30d"])
        print(f"  {f}: {corr:.3f}")
    print("\nDataset saved to data/readmission_dataset.csv")