# src/data_processing/feature_engineering.py

"""
This script applies feature transformations for MIMIC-IV data.
We assume we have a processed_cohort.csv with columns like:
  [
    'subject_id', 'hadm_id', 'admittime', 'dischtime', 'anchor_age',
    'gender', 'num_diagnoses', 'readmitted_within_30d', 'next_admittime'
  ]

Steps:
1. Create an AGE_AT_ADMISSION feature (simplified to anchor_age).
2. Handle missing values in NUM_DIAGNOSES by filling with 0.
3. Encode gender as binary (0/1).
4. Save a final feature set ready for modeling.
"""

import os
import pandas as pd
import numpy as np
from src.config.base_config import PROCESSED_DATA_DIR

def engineer_features(input_filename="processed_cohort.csv", output_filename="final_features.csv"):
    input_path = os.path.join(PROCESSED_DATA_DIR, input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_filename} not found in {PROCESSED_DATA_DIR}")

    # For MIMIC-IV, 'admittime' and 'dischtime' are typically already in datetime if you created them in preprocess.py
    # We'll parse them again just in case, ignoring errors if they're already datetimes.
    df = pd.read_csv(input_path, parse_dates=['admittime', 'dischtime', 'next_admittime'], keep_default_na=True)

    # --- 1. AGE_AT_ADMISSION Feature ---
    # MIMIC-IV does not have a direct DOB. Instead, we have anchor_age (approximate age at first admission).
    # For a simple approach, just use anchor_age as the age feature.
    # If you need a more precise age per admission, you could approximate by anchor_year vs admittime year, etc.
    df['age_at_admission'] = df['anchor_age']

    # Handle unrealistic ages (some might be > 90 due to deidentification policy).
    # This step is optional. You might choose to leave them as-is or clamp them.
    df.loc[df['age_at_admission'] > 90, 'age_at_admission'] = 90

    # --- 2. Missing Values for num_diagnoses ---
    # In MIMIC-IV, the script may have named it 'num_diagnoses' already.
    # If your code uses a different name, adjust accordingly.
    if 'num_diagnoses' in df.columns:
        df['num_diagnoses'] = df['num_diagnoses'].fillna(0)
    else:
        df['num_diagnoses'] = 0  # fallback if not present

    # --- 3. Encode Gender (M/F -> 0/1) ---
    # MIMIC-IV typically uses 'gender' with values "M" or "F".
    if 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})
        df.drop(columns=['gender'], inplace=True)
    else:
        df['gender_encoded'] = 0

    # --- 4. Remove Unneeded Columns ---
    # We remove columns we won't use in modeling, like raw times or anchor_age itself
    # (you can keep them if you want to do time-based features).
    # We'll keep 'hadm_id', 'subject_id', and 'readmitted_within_30d' for reference.
    drop_cols = [
        'admittime', 
        'dischtime', 
        'next_admittime', 
        'anchor_age'
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Save the engineered features
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved to {output_path}")

if __name__ == "__main__":
    engineer_features()
