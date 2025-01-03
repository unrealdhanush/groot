# src/data_processing/preprocess.py

"""
This script takes raw MIMIC-IV tables and preprocesses them into a cleaned,
analysis-ready dataset for 30-day readmission prediction.
"""

import pandas as pd
import os
from datetime import timedelta
from src.config.base_config import (
    PROCESSED_DATA_DIR,
    TARGET_READMISSION_DAYS,
    RANDOM_SEED
)
from src.data_processing.data_loader import load_admissions, load_patients, load_diagnoses

def create_cohort() -> pd.DataFrame:
    """
    Create a cohort of patients and admissions suitable for predicting 30-day readmission.
    
    Steps:
    1. Load admissions, patients, and diagnoses data (MIMIC-IV).
    2. Merge into a single DataFrame keyed by admission (hadm_id).
    3. Sort admissions by admittime per subject_id to find subsequent admissions.
    4. Determine if a given admission is followed by another admission within 30 days.
    5. Return a cleaned DataFrame with a binary target for readmission.
    
    Returns:
        pd.DataFrame: DataFrame with patient/admission-level features and 'readmitted_within_30d' target.
    """
    # 1. Load data
    admissions = load_admissions()  # must have columns ['subject_id', 'hadm_id', 'admittime', 'dischtime']
    patients = load_patients()      # must have columns like ['subject_id', 'anchor_age', 'gender']
    diagnoses = load_diagnoses()    # must have columns ['subject_id', 'hadm_id', 'icd_code', ...]

    # 2. Convert admission times to datetime
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

    # 3. Merge patient info (we'll keep anchor_age and gender as example features)
    # In MIMIC-IV, 'anchor_age' is the approximate age at first admission.
    # 'gender' can be 'M' or 'F'.
    merged = admissions.merge(
        patients[['subject_id', 'anchor_age', 'gender']], 
        on='subject_id', 
        how='left'
    )

    # 4. Incorporate diagnoses
    # diagnoses_icd.csv.gz in MIMIC-IV often uses 'icd_code' (and 'icd_version')
    # If your file uses 'icd9_code', adjust the line below to match.
    diag_counts = diagnoses.groupby('hadm_id')['icd_code'].count().reset_index()
    diag_counts.rename(columns={'icd_code': 'num_diagnoses'}, inplace=True)
    merged = merged.merge(diag_counts, on='hadm_id', how='left')

    # 5. Sort to find subsequent admissions
    merged = merged.sort_values(by=['subject_id', 'admittime'])

    # 6. Create the 30-day readmission target
    merged['next_admittime'] = merged.groupby('subject_id')['admittime'].shift(-1)
    merged['readmitted_within_30d'] = (
        (merged['next_admittime'] - merged['dischtime']) <= timedelta(days=TARGET_READMISSION_DAYS)
    ).astype(int)

    # For the last admission of each patient, there's no "next_admittime"; fill with 0
    merged['readmitted_within_30d'].fillna(0, inplace=True)

    return merged

def save_processed_data(df: pd.DataFrame, filename="processed_cohort.csv"):
    """
    Save the processed DataFrame to the processed directory.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Running this script directly will create and save the cohort.
    df = create_cohort()
    print("Target Distribution:\n", df['readmitted_within_30d'].value_counts())
    save_processed_data(df)
