# src/config/base_config.py

"""
This configuration file centralizes all constants, file paths, and parameters used across the project.
By changing these values here, we can easily adjust our pipeline without editing multiple scripts.
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(ROOT_DIR, "data", "interim")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

ADMISSIONS_FILE = os.path.join(RAW_DATA_DIR, "ADMISSIONS.csv.gz")  
PATIENTS_FILE = os.path.join(RAW_DATA_DIR, "PATIENTS.csv.gz")
DIAGNOSES_FILE = os.path.join(RAW_DATA_DIR, "DIAGNOSES_ICD.csv.gz")
NOTES_FILE = os.path.join(RAW_DATA_DIR, "NOTEEVENTS.csv.gz")

TARGET_READMISSION_DAYS = 30  
TEST_SIZE = 0.2  
RANDOM_SEED = 42