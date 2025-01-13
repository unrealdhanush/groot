# src/config/base_config.py

"""
This configuration file centralizes all constants, file paths, and parameters used across the project.
By changing these values here, we can easily adjust our pipeline without editing multiple scripts.
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
ADMISSIONS_FILE = os.path.join(RAW_DATA_DIR, "mimiciv/3.1/hosp/admissions.csv.gz")
PATIENTS_FILE = os.path.join(RAW_DATA_DIR, "mimiciv/3.1/hosp/patients.csv.gz")
DIAGNOSES_FILE = os.path.join(RAW_DATA_DIR, "mimiciv/3.1/hosp/diagnoses_icd.csv.gz")
NOTES_FILE = os.path.join(RAW_DATA_DIR, "mimic-iv-note/2.2/note/discharge.csv.gz")

TARGET_READMISSION_DAYS = 30  
TEST_SIZE = 0.2  
RANDOM_SEED = 42