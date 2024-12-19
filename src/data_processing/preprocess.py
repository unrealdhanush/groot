# src/data_processing/preprocess.py

"""
This script takes raw MIMIC tables and preprocesses them into a cleaned, 
analysis-ready dataset. The preprocessing might include:
- Merging admissions, patients, diagnoses tables.
- Creating the target variable (readmission within 30 days).
- Filtering out certain populations if needed.
- Saving the processed dataset for modeling.
"""

import pandas as pd
import os
from datetime import timedelta
from src.config.base_config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_READMISSION_DAYS,
    RANDOM_SEED
)
from src.data_processing.data_loader import load_admissions, load_patients, load_diagnoses

def create_cohort() -> pd.DataFrame:
    """
    Create a cohort of patients and admissions suitable for predicting 30-day readmission.
    
    Steps:
    1. Load admissions, patients, and diagnoses data.
    2. Merge them into a single DataFrame keyed by admission.
    3. Sort admissions by admission time per patient to find subsequent admissions.
    4. Determine if a given admission was followed by another admission within 30 days.
    5. Return a cleaned DataFrame with target labels.
    
    Returns:
        pd.DataFrame: A DataFrame with patient and admission-level features and a binary target for readmission.
    """
    # 1. Load data
    admissions = load_admissions()
    patients = load_patients()
    diagnoses = load_diagnoses()
    
    # 2. Merge data
    # A common key is SUBJECT_ID (patient) and HADM_ID (admission).
    # We'll start simple: keep only a subset of columns.
    # Let's say we keep ADMITTIME, DISCHTIME from admissions, age/gender from patients, ICD codes from diagnoses.
    # (You would tailor this to your analysis)
    
    # Convert date columns to datetime
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
    
    # Merge patients info
    # Patients table has SUBJECT_ID unique per patient. Let's pick DOB, GENDER.
    # Convert DOB to datetime if needed.
    patients['DOB'] = pd.to_datetime(patients['DOB'], errors='coerce')
    # Merge admissions with patients on SUBJECT_ID
    merged = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID', how='left')
    
    # Incorporate diagnoses
    # Diagnoses table typically has multiple rows per admission. Let's get a count of distinct ICD9 codes per admission.
    diag_counts = diagnoses.groupby('HADM_ID')['ICD9_CODE'].count().reset_index()
    diag_counts.rename(columns={'ICD9_CODE': 'NUM_DIAGNOSES'}, inplace=True)
    merged = merged.merge(diag_counts, on='HADM_ID', how='left')
    
    # 3. Sort admissions by time to find subsequent admissions
    merged = merged.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
    
    # 4. Create the readmission target
    # For each admission, we look for the next admission for the same patient and check if it occurred within 30 days.
    # We'll do a groupby SUBJECT_ID and shift the next admission time.
    merged['NEXT_ADMITTIME'] = merged.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
    merged['READMITTED_WITHIN_30D'] = (
        (merged['NEXT_ADMITTIME'] - merged['DISCHTIME']) <= timedelta(days=TARGET_READMISSION_DAYS)
    ).astype(int)
    
    # The last admission for each patient won't have a next admission, so it becomes NaN. Fill NaN with 0.
    merged['READMITTED_WITHIN_30D'].fillna(0, inplace=True)
    
    # Optional: filter out admissions that happen in NICU or certain wards, or patients under 18, etc.
    # For demonstration, we’ll assume no extra filtering now.
    
    # 5. Return the final DataFrame.
    # This DataFrame now has a binary column 'READMITTED_WITHIN_30D' that we can predict.
    
    return merged

def save_processed_data(df: pd.DataFrame, filename="processed_cohort.csv"):
    """
    Save the processed DataFrame to the processed directory.
    """
    # Ensure directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Running this script directly will create and save the cohort.
    df = create_cohort()
    # You might want to do some basic checks here, like ensuring the target distribution is reasonable.
    print("Target Distribution:\n", df['READMITTED_WITHIN_30D'].value_counts())
    save_processed_data(df)
