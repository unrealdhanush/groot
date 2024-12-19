# src/data_processing/data_loader.py

"""
This script is responsible for loading raw MIMIC data into memory.
It provides functions to load admissions, patients, diagnoses, and notes data.
We want to ensure that each function just focuses on loading and returning raw DataFrames.
All heavy preprocessing will be done in separate scripts to keep responsibilities separate.
"""

import os
import pandas as pd
from src.config.base_config import (
    ADMISSIONS_FILE,
    PATIENTS_FILE,
    DIAGNOSES_FILE,
    NOTES_FILE
)

def load_admissions() -> pd.DataFrame:
    """
    Load the admissions table from MIMIC dataset.
    
    Returns:
        pd.DataFrame: The admissions data.
    """
    # Check if file exists before loading
    if not os.path.exists(ADMISSIONS_FILE):
        raise FileNotFoundError(f"Admissions file not found at {ADMISSIONS_FILE}")
    
    # Load using pandas. The file may be gzipped (common for large MIMIC files).
    # compression='gzip' will handle that automatically.
    df = pd.read_csv(ADMISSIONS_FILE, compression='gzip')
    return df

def load_patients() -> pd.DataFrame:
    """
    Load the patients table from MIMIC dataset.
    
    Returns:
        pd.DataFrame: The patients data.
    """
    if not os.path.exists(PATIENTS_FILE):
        raise FileNotFoundError(f"Patients file not found at {PATIENTS_FILE}")
    
    df = pd.read_csv(PATIENTS_FILE, compression='gzip')
    return df

def load_diagnoses() -> pd.DataFrame:
    """
    Load the diagnoses table from MIMIC dataset.
    
    Returns:
        pd.DataFrame: The diagnoses data.
    """
    if not os.path.exists(DIAGNOSES_FILE):
        raise FileNotFoundError(f"Diagnoses file not found at {DIAGNOSES_FILE}")
    
    df = pd.read_csv(DIAGNOSES_FILE, compression='gzip')
    return df

def load_notes() -> pd.DataFrame:
    """
    Load the notes (clinical notes) from MIMIC dataset.
    
    Returns:
        pd.DataFrame: The notes data.
    """
    if not os.path.exists(NOTES_FILE):
        raise FileNotFoundError(f"Notes file not found at {NOTES_FILE}")
    
    df = pd.read_csv(NOTES_FILE, compression='gzip')
    return df