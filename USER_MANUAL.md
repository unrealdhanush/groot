# **User Manual: G.R.O.O.T (Guided Readmission & Orchestrated Observation Text)**

Welcome to the comprehensive user manual for our **G.R.O.O.T (Guided Readmission & Orchestrated Observation Text)** and **Retrieval-Augmented Generation (RAG) Summaries** project. This manual aims to explain:

1. **Why** this project exists.  
2. **Who** can use it.  
3. **Why** we chose the MIMIC dataset.  
4. **Where** the data came from.  
5. **What** the entire code base does, **how** it does it, and **when** you might want to use certain features.  

We’ll keep the language straightforward and highlight motivations, folder structures, and code commentary extensively.

---

## **Contents**

1. Overview
2. Data
3. Folder Structure
4. Codex Manual
5. How to Use

---

## **1. Overview**

### **1.1 What Is This Project?**

- We want to **predict 30-day readmission risk** for patients by using a real-world medical dataset called **MIMIC** (Medical Information Mart for Intensive Care).
- We also incorporate a **RAG (Retrieval-Augmented Generation)** pipeline to provide **short text summaries** of the risk factors and best practices, leveraging **Local LLM** or **Hugging Face** models.

### **1.2 Why We Need This Project**

- **Hospitals** often care about which patients are more likely to come back (“readmitted”) within 30 days because it affects costs, patient health outcomes, and resource planning.
- By **predicting** who’s at high risk, doctors/nurses can intervene earlier, reduce complications, and **focus** on better care transitions.
- The **RAG Summaries** help automatically write a short note or plan about these risk factors, so clinicians can quickly see the main issues.

### **1.3 Who Can Use This Project**

1. **Medical Researchers**: They can try custom features, test readmission models, and refine knowledge-based summarization.
2. **Data Scientists**: They can learn how to do end-to-end ingestion, modeling, retrieval, and summarization with large language models (LLMs).
3. **Clinicians or IT Staff**: Potentially adapt the pipeline to see risk predictions for real patients (with the correct compliance and anonymization).
4. **Curious Learners**: Even someone new can run the pipeline, pick conditions, and see how the model responds.

### **1.4 Motivation for the Project**

- Real hospitals pay close attention to 30-day readmission rates (like for heart failure, sepsis, etc.). We want a **clinical decision support tool** that:
  1. Predicts readmission risk.  
  2. Summarizes key risk factors & guidelines to help clinicians or data scientists see **why** a patient might be readmitted.

---

## **2. Data**

### **2.1 What Is MIMIC?**

**MIMIC (Medical Information Mart for Intensive Care)** is a large, publicly available, de-identified clinical database developed by the MIT Lab for Computational Physiology. It contains detailed data from critical care units of the Beth Israel Deaconess Medical Center. Because the data is rigorously de-identified, researchers and data scientists worldwide use it to develop and evaluate algorithms in healthcare machine learning.

### **2.2 Origin of MIMIC**

- **MIMIC** is hosted on [PhysioNet.org](https://physionet.org/).  
- Anyone can sign up for an account, complete a short certification course on data usage, and then request access to the MIMIC dataset for research or educational purposes.

### **2.3 Why MIMIC Dataset?**

- **Breadth and Depth**: MIMIC includes **patient admissions**, **discharge summaries**, ICU chart events, lab data, and more, making it ideal for modeling real-world hospital scenarios.  
- **Free and Open**: Although you need credentialing, there’s no cost to access MIMIC once you’ve met the requirements, and the data is widely used in academic research.  
- **Rich in Clinical Variables**: The dataset spans multiple ICU stays, enabling the creation of advanced prediction tasks such as **30-day readmission** and **clinical summarization**.  
- **NLP-Friendly**: MIMIC provides **clinical notes** for tasks like discharge summary analysis, which we leverage to build an **NLP** pipeline for summarizing risk factors and recommended guidelines.

---

## **3. Folder & File Structure**

Here is a typical layout:

```bash
groot
├── data
│   ├── knowledge_base
│   │   └── medical_knowledge.csv
│   ├── processed
│   └── raw
├── models
│   ├── baseline_xgb_model.json
│   └── fusion_model.pt
├── notebooks
│   └── EDA.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── config
│   │   ├── __init__.py
│   │   └── base_config.py
│   ├── data_processing
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── merge_embeddings.py
│   │   └── preprocess.py
│   ├── explainability
│   │   ├── __init__.py
│   │   └── explain.py
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── baseline_model.py
│   │   ├── fusion_model.py
│   │   └── trainer.py
│   ├── nlp
│   │   ├── __init__.py
│   │   ├── domain_adaption.py
│   │   └── embed_clinical_texts.py
│   └── rag
│       ├── __init__.py
│       ├── build_index.py
│       └── generate_summaries.py
├── .gitignore
├── README.md
├── USER_MANUAL.md
├── app_fm.py
├── app_xgb.py
├── requirements.txt
├── base_structure.txt
└── structure.txt
```

After obtaining access and downloading the MIMIC-IV dataset, the typical dataset structure would look something like this:

```bash
data
└── raw
│       ├── mimic-iv-note
│       │   └── 2.2
│       │       ├── LICENSE.txt
│       │       ├── SHA256SUMS.txt
│       │       ├── index.html
│       │       └── note
│       │           ├── discharge.csv.gz
│       │           ├── discharge_detail.csv.gz
│       │           ├── index.html
│       │           ├── radiology.csv.gz
│       │           └── radiology_detail.csv.gz
│       ├── mimiciv
│       │   └── 3.1
│       │       ├── CHANGELOG.txt
│       │       ├── LICENSE.txt
│       │       ├── SHA256SUMS.txt
│       │       ├── hosp
│       │       │   ├── admissions.csv.gz
│       │       │   ├── d_hcpcs.csv.gz
│       │       │   ├── d_icd_diagnoses.csv.gz
│       │       │   ├── d_icd_procedures.csv.gz
│       │       │   ├── d_labitems.csv.gz
│       │       │   ├── diagnoses_icd.csv.gz
│       │       │   ├── drgcodes.csv.gz
│       │       │   ├── emar.csv.gz
│       │       │   ├── emar_detail.csv.gz
│       │       │   ├── hcpcsevents.csv.gz
│       │       │   ├── index.html
│       │       │   ├── labevents.csv.gz
│       │       │   ├── microbiologyevents.csv.gz
│       │       │   ├── omr.csv.gz
│       │       │   ├── patients.csv.gz
│       │       │   ├── pharmacy.csv.gz
│       │       │   ├── poe.csv.gz
│       │       │   ├── poe_detail.csv.gz
│       │       │   ├── prescriptions.csv.gz
│       │       │   ├── procedures_icd.csv.gz
│       │       │   ├── provider.csv.gz
│       │       │   ├── services.csv.gz
│       │       │   └── transfers.csv.gz
│       │       ├── icu
│       │       │   ├── caregiver.csv.gz
│       │       │   ├── chartevents.csv.gz
│       │       │   ├── d_items.csv.gz
│       │       │   ├── datetimeevents.csv.gz
│       │       │   ├── icustays.csv.gz
│       │       │   ├── index.html
│       │       │   ├── ingredientevents.csv.gz
│       │       │   ├── inputevents.csv.gz
│       │       │   ├── outputevents.csv.gz
│       │       │   └── procedureevents.csv.gz
│       │       └── index.html
│       └── robots.txt
```

---

## **4. Codex Manual**

Now let's go through the codebase and how it was framed.

### **4.1 Environment and Configuration**

We can start by creating and activating a virtual environment ```venv``` in Python. To do this you can execute the following bash script command in your terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After the above command is executed, we are ready to install all the necessary libraries for this project using the requirements text file in the root of the repository. That can be initiated by executing the following command:

```bash
pip install -r requirements.txt
```

Next, let's focus on initializing a configuration file, which handles paths and constants that will be used throughout the project. This helps maintaining consistency and avoids hardcoding paths in multiple files.

File: ```src/config/base_config.py```

Here is how the code looks:

```bash
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
```

Explanation:

- ```ROOT_DIR``` is computed dynamically, ensuring we can run scripts from anywhere.

- ```RAW_DATA_DIR``` and ```PROCESSED_DATA_DIR``` define where we store data at various stages.

- Paths like ```ADMISSIONS_FILE``` and ```PATIENTS_FILE``` point to actual MIMIC CSV files. You must have MIMIC data placed in ```data/raw/``` as instructed in the previous section.

- Constants like ```TARGET_READMISSION_DAYS``` are central to our problem definition.

- ```TEST_SIZE``` and ```RANDOM_SEED``` will be used for splitting data and ensuring reproducibility.

### **4.2 Data Acquisition and Preprocessing**

#### **4.2.1 Data Loader**

Here is the data loader script, this scripts basically:

1. Check if the MIMIC data files exist
2. Load the CSV files into the memory using ```pandas```.
3. Does minimal preprocessing (like selecting needed columns).

File: ```src/data_processing/data_loader.py```

```bash
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
```

Explanation:

- Each loader function checks if the file exists. This is usually a good practice for error handling.

- ```pd.read_csv(..., compression='gzip')```: MIMIC data often is gzipped. This handle is automatic.

- Each function returns a pandas Dataframe. We do no processing here, just loading.

#### **4.2.2 Preprocessing Script**

Here we define the logic to merge tables, create the outcome variable (30-day readmission), and do initial cleaning. For instance here, we create a cohort from patirnts admitted for certain conditions, or just select the relevant columns.

File: ```src/data_processing/preprocess.py```

```bash
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
```

Explanation:

- ```def create_cohort()```: Main function to create the aalysis-ready data.

- ```load_admissions()```, ```load_patients()```, ```load_diagnoses()``` are all caled to bring in raw data.

- 