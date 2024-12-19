# src/data_processing/feature_engineering.py

"""
This script applies feature transformations determined during EDA.
We assume we have a processed_cohort.csv with basic columns:
  ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'GENDER', 'DOB', 'NUM_DIAGNOSES', 'READMITTED_WITHIN_30D']
We will:
1. Create an AGE_AT_ADMISSION feature.
2. Handle missing values in NUM_DIAGNOSES by filling with 0 (as an example).
3. (Optional) Encode GENDER as binary (0/1).
4. Save a final feature set ready for modeling.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.config.base_config import PROCESSED_DATA_DIR

def engineer_features(input_filename="processed_cohort.csv", output_filename="final_features.csv"):
    input_path = os.path.join(PROCESSED_DATA_DIR, input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_filename} not found in {PROCESSED_DATA_DIR}")

    df = pd.read_csv(input_path, parse_dates=['ADMITTIME', 'DISCHTIME', 'DOB'])

    # Feature 1: Age at admission
    df['AGE_AT_ADMISSION'] = (df['ADMITTIME'] - df['DOB']).dt.days / 365.25
    # Handle unrealistic ages (some MIMIC patients have masked DOBs leading to extreme ages)
    # Replace ages > 90 with 90 (censoring age)
    df.loc[df['AGE_AT_ADMISSION'] > 90, 'AGE_AT_ADMISSION'] = 90

    # Feature 2: Missing values handling for NUM_DIAGNOSES
    df['NUM_DIAGNOSES'].fillna(0, inplace=True)

    # Feature 3: Encode GENDER
    # Map M, F to 0,1 for modeling convenience.
    df['GENDER_ENCODED'] = df['GENDER'].map({'M': 0, 'F': 1})
    df.drop(columns=['GENDER'], inplace=True)

    # Drop columns that won't be used for modeling (like raw times, DOB)
    # We keep ADMITTIME, DISCHTIME if we want time-based features. Otherwise, we might remove them.
    # For now, let's remove them to keep a simpler dataset for baseline:
    df.drop(columns=['ADMITTIME', 'DISCHTIME', 'DOB', 'NEXT_ADMITTIME'], errors='ignore', inplace=True)

    # Save the engineered features
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved to {output_path}")

if __name__ == "__main__":
    engineer_features()
