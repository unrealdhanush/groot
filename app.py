# app.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from src.config.base_config import PROCESSED_DATA_DIR
from src.rag.generate_summaries import generate_summary

# Define all feature names expected by the model
FEATURE_NAMES = [
    'subject_id', 'hadm_id', 'hospital_expire_flag', 'num_diagnoses', 'age_at_admission',
    'gender_encoded', 'admission_type_DIRECT EMER.', 'admission_type_DIRECT OBSERVATION',
    'admission_type_ELECTIVE', 'admission_type_EU OBSERVATION', 'admission_type_EW EMER.',
    'admission_type_OBSERVATION ADMIT', 'admission_type_SURGICAL SAME DAY ADMISSION',
    'admission_type_URGENT', 'admission_location_CLINIC REFERRAL',
    'admission_location_EMERGENCY ROOM', 'admission_location_INFORMATION NOT AVAILABLE',
    'admission_location_INTERNAL TRANSFER TO OR FROM PSYCH', 'admission_location_PACU',
    'admission_location_PHYSICIAN REFERRAL', 'admission_location_PROCEDURE SITE',
    'admission_location_TRANSFER FROM HOSPITAL', 'admission_location_TRANSFER FROM SKILLED NURSING FACILITY',
    'admission_location_WALK-IN/SELF REFERRAL', 'insurance_Medicare', 'insurance_No charge',
    'insurance_Other', 'insurance_Private', 'language_Amharic', 'language_Arabic',
    'language_Armenian', 'language_Bengali', 'language_Chinese', 'language_English',
    'language_French', 'language_Haitian', 'language_Hindi', 'language_Italian',
    'language_Japanese', 'language_Kabuverdianu', 'language_Khmer', 'language_Korean',
    'language_Modern Greek (1453-)', 'language_Other', 'language_Persian', 'language_Polish',
    'language_Portuguese', 'language_Russian', 'language_Somali', 'language_Spanish',
    'language_Thai', 'language_Vietnamese', 'marital_status_MARRIED',
    'marital_status_SINGLE', 'marital_status_WIDOWED', 'race_ASIAN',
    'race_ASIAN - ASIAN INDIAN', 'race_ASIAN - CHINESE', 'race_ASIAN - KOREAN',
    'race_ASIAN - SOUTH EAST ASIAN', 'race_BLACK/AFRICAN', 'race_BLACK/AFRICAN AMERICAN',
    'race_BLACK/CAPE VERDEAN', 'race_BLACK/CARIBBEAN ISLAND', 'race_HISPANIC OR LATINO',
    'race_HISPANIC/LATINO - CENTRAL AMERICAN', 'race_HISPANIC/LATINO - COLUMBIAN',
    'race_HISPANIC/LATINO - CUBAN', 'race_HISPANIC/LATINO - DOMINICAN',
    'race_HISPANIC/LATINO - GUATEMALAN', 'race_HISPANIC/LATINO - HONDURAN',  'race_HISPANIC/LATINO - MEXICAN',
    'race_HISPANIC/LATINO - PUERTO RICAN', 'race_HISPANIC/LATINO - SALVADORAN',
    'race_MULTIPLE RACE/ETHNICITY', 'race_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
    'race_OTHER', 'race_PATIENT DECLINED TO ANSWER', 'race_PORTUGUESE',
    'race_SOUTH AMERICAN', 'race_UNABLE TO OBTAIN', 'race_UNKNOWN', 'race_WHITE',
    'race_WHITE - BRAZILIAN', 'race_WHITE - EASTERN EUROPEAN', 'race_WHITE - OTHER EUROPEAN',
    'race_WHITE - RUSSIAN'
]

MEDICAL_CONDITIONS = [
    "heart failure", "diabetes mellitus", "chronic kidney disease", "hypertension",
    "COPD", "pneumonia", "sepsis", "myocardial infarction", "stroke",
    "depression", "anxiety", "asthma", "obesity", "cancer", "arthritis"
]

# Initialize all features to 0
def initialize_features():
    data = {feature: 0 for feature in FEATURE_NAMES}
    return pd.DataFrame([data])

# Load model with caching
@st.cache_resource
def load_xgb_model(model_path):
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        st.success("XGBoost model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None

# Load XGBoost model once
model_path = os.path.join(PROCESSED_DATA_DIR, "baseline_xgb_model.json")
model = load_xgb_model(model_path)

st.title("MIMIC 30-Day Readmission Risk Predictor")

# User inputs
age = st.number_input("Age at Admission", min_value=0, max_value=100, value=65)
num_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=50, value=5)
gender = st.selectbox("Gender", ["M", "F"])

# Admission Type (One-Hot Encoded)
admission_types = [
    "DIRECT EMER.", "DIRECT OBSERVATION", "ELECTIVE", "EU OBSERVATION",
    "EW EMER.", "OBSERVATION ADMIT", "SURGICAL SAME DAY ADMISSION", "URGENT"
]
selected_admission_type = st.selectbox("Admission Type", admission_types)

# Admission Location (One-Hot Encoded)
admission_locations = [
    "CLINIC REFERRAL", "EMERGENCY ROOM", "INFORMATION NOT AVAILABLE",
    "INTERNAL TRANSFER TO OR FROM PSYCH", "PACU", "PHYSICIAN REFERRAL",
    "PROCEDURE SITE", "TRANSFER FROM HOSPITAL", "TRANSFER FROM SKILLED NURSING FACILITY",
    "WALK-IN/SELF REFERRAL"
]
selected_admission_location = st.selectbox("Admission Location", admission_locations)

# Insurance (One-Hot Encoded)
insurances = ["Medicare", "No charge", "Other", "Private"]
selected_insurance = st.selectbox("Insurance", insurances)

# Language (One-Hot Encoded)
languages = [
    "Amharic", "Arabic", "Armenian", "Bengali", "Chinese", "English",
    "French", "Haitian", "Hindi", "Italian", "Japanese", "Kabuverdianu",
    "Khmer", "Korean", "Modern Greek (1453-)", "Other", "Persian",
    "Polish", "Portuguese", "Russian", "Somali", "Spanish", "Thai",
    "Vietnamese"
]
selected_language = st.selectbox("Language", languages)

# Marital Status (One-Hot Encoded)
marital_statuses = ["MARRIED", "SINGLE", "WIDOWED"]
selected_marital_status = st.selectbox("Marital Status", marital_statuses)

# Race (One-Hot Encoded)
races = [
    "ASIAN", "ASIAN - ASIAN INDIAN", "ASIAN - CHINESE", "ASIAN - KOREAN",
    "ASIAN - SOUTH EAST ASIAN", "BLACK/AFRICAN", "BLACK/AFRICAN AMERICAN",
    "BLACK/CAPE VERDEAN", "BLACK/CARIBBEAN ISLAND", "HISPANIC OR LATINO",
    "HISPANIC/LATINO - CENTRAL AMERICAN", "HISPANIC/LATINO - COLUMBIAN",
    "HISPANIC/LATINO - CUBAN", "HISPANIC/LATINO - DOMINICAN",
    "HISPANIC/LATINO - GUATEMALAN", "HISPANIC/LATINO - HONDURAN",
    "HISPANIC/LATINO - MEXICAN", "HISPANIC/LATINO - PUERTO RICAN",
    "HISPANIC/LATINO - SALVADORAN", "MULTIPLE RACE/ETHNICITY",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER", "OTHER",
    "PATIENT DECLINED TO ANSWER", "PORTUGUESE", "SOUTH AMERICAN",
    "UNABLE TO OBTAIN", "UNKNOWN", "WHITE",
    "WHITE - BRAZILIAN", "WHITE - EASTERN EUROPEAN",
    "WHITE - OTHER EUROPEAN", "WHITE - RUSSIAN"
]
selected_race = st.selectbox("Race", races)

# Medical Conditions (Dynamic Input)
selected_conditions = st.multiselect(
    "Select Relevant Medical Conditions",
    options=MEDICAL_CONDITIONS,
    default=None
)

# Allow users to add additional conditions
additional_conditions = st.text_input("Add Additional Medical Conditions (comma-separated)")
if additional_conditions:
    additional_conditions = [cond.strip() for cond in additional_conditions.split(",") if cond.strip()]
    if selected_conditions is None:
        selected_conditions = additional_conditions
    else:
        selected_conditions = selected_conditions + additional_conditions

if st.button("Predict Readmission Risk"):
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
    else:
        try:
            # Initialize all features to 0
            X_input = initialize_features()

            # Update user-provided features
            X_input.at[0, 'age_at_admission'] = age
            X_input.at[0, 'num_diagnoses'] = num_diagnoses
            X_input.at[0, 'gender_encoded'] = 1 if gender == 'F' else 0

            # Update Admission Type
            admission_type_column = f"admission_type_{selected_admission_type}"
            if admission_type_column in FEATURE_NAMES:
                X_input.at[0, admission_type_column] = 1

            # Update Admission Location
            admission_location_column = f"admission_location_{selected_admission_location}"
            if admission_location_column in FEATURE_NAMES:
                X_input.at[0, admission_location_column] = 1

            # Update Insurance
            insurance_column = f"insurance_{selected_insurance}"
            if insurance_column in FEATURE_NAMES:
                X_input.at[0, insurance_column] = 1

            # Update Language
            language_column = f"language_{selected_language}"
            if language_column in FEATURE_NAMES:
                X_input.at[0, language_column] = 1

            # Update Marital Status
            marital_status_column = f"marital_status_{selected_marital_status}"
            if marital_status_column in FEATURE_NAMES:
                X_input.at[0, marital_status_column] = 1

            # Update Race
            race_column = f"race_{selected_race}"
            if race_column in FEATURE_NAMES:
                X_input.at[0, race_column] = 1

            # Optional: Handle 'subject_id' and 'hadm_id' if necessary
            # For prediction purposes, you might set them to a default value like 0
            X_input.at[0, 'subject_id'] = 0
            X_input.at[0, 'hadm_id'] = 0
            X_input.at[0, 'hospital_expire_flag'] = 0  # Assuming this is a feature, not the target

            # Convert to DMatrix
            dmatrix = xgb.DMatrix(X_input)

            # Make prediction
            y_pred = model.predict(dmatrix)
            readmission_prob = float(y_pred[0])

            st.write(f"Predicted Probability of Readmission: {readmission_prob*100:.2f}%")

            # Generate summary with RAG
            patient_context = f"A {age}-year-old patient with {num_diagnoses} diagnoses."
            conditions = selected_conditions if selected_conditions else []
            
            if not conditions:
                st.warning("No medical conditions selected. Please select or add relevant medical conditions to generate a summary.")
            else:
                with st.spinner('Generating summary...'):
                    summary = generate_summary(patient_context, conditions)
                if summary:
                    st.write("RAG-based Summary:")
                    st.write(summary)
                else:
                    st.error("Failed to generate summary.")
        except Exception as e:
            st.error(f"An error occurred during prediction or summary generation: {e}")
