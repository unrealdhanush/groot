import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import xgboost as xgb
from src.config.base_config import PROCESSED_DATA_DIR
from src.rag.generate_summaries import generate_summary
import streamlit as st

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

def initialize_features():
    data = {feature: 0 for feature in FEATURE_NAMES}
    return pd.DataFrame([data])

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

model_path = os.path.join(PROCESSED_DATA_DIR, "baseline_xgb_model.json")
model = load_xgb_model(model_path)

st.title("G.R.O.O.T (Guided Readmission & Orchestrated Observation Text)")

#################################################
# Basic Inputs
#################################################
age = st.number_input("Age at Admission", min_value=0, max_value=100, value=65)

if age < 12:
    # Initialize or check session_state
    if "parental_mode_accepted" not in st.session_state:
        st.session_state["parental_mode_accepted"] = False
    
    if not st.session_state["parental_mode_accepted"]:
        # Show short warning + accept button
        st.warning("Parental Mode activated! This pipeline is primarily adult-focused. "
                   "Pediatric predictions may be less reliable. Please accept to proceed.")
        if st.button("Accept Parental Mode"):
            st.session_state["parental_mode_accepted"] = True
            st.rerun()
        else:
            st.stop()
    else:
        st.warning("Parental Mode activated! This pipeline is primarily adult-focused. "
                   "Pediatric predictions may be less reliable. Please consult a pediatric "
                   "specialist for accurate results.")

num_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=50, value=5)
if num_diagnoses > 20:
    # Initialize or check session_state
    if "more_diagnoses_accepted" not in st.session_state:
        st.session_state["more_diagnoses_accepted"] = False
    
    if not st.session_state["more_diagnoses_accepted"]:
        # Show short warning + accept button
        st.warning("Are you sure about the number of diagnoses?")
        if st.button("Yes"):
            st.session_state["more_diagnoses_accepted"] = True
            st.rerun()
        else:
            st.stop()
gender = st.selectbox("Gender", ["M", "F"])

admission_types = [
    "DIRECT EMER.", "DIRECT OBSERVATION", "ELECTIVE", "EU OBSERVATION",
    "EW EMER.", "OBSERVATION ADMIT", "SURGICAL SAME DAY ADMISSION", "URGENT"
]
selected_admission_type = st.selectbox("Admission Type", admission_types)

admission_locations = [
    "CLINIC REFERRAL", "EMERGENCY ROOM", "INFORMATION NOT AVAILABLE",
    "INTERNAL TRANSFER TO OR FROM PSYCH", "PACU", "PHYSICIAN REFERRAL",
    "PROCEDURE SITE", "TRANSFER FROM HOSPITAL", "TRANSFER FROM SKILLED NURSING FACILITY",
    "WALK-IN/SELF REFERRAL"
]
selected_admission_location = st.selectbox("Admission Location", admission_locations)

insurances = ["Medicare", "No charge", "Other", "Private"]
selected_insurance = st.selectbox("Insurance", insurances)

languages = [
    "Amharic", "Arabic", "Armenian", "Bengali", "Chinese", "English",
    "French", "Haitian", "Hindi", "Italian", "Japanese", "Kabuverdianu",
    "Khmer", "Korean", "Modern Greek (1453-)", "Other", "Persian",
    "Polish", "Portuguese", "Russian", "Somali", "Spanish", "Thai",
    "Vietnamese"
]
selected_language = st.selectbox("Language", languages)

marital_statuses = ["MARRIED", "SINGLE", "WIDOWED"]
selected_marital_status = st.selectbox("Marital Status", marital_statuses)

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

selected_conditions = st.multiselect(
    "Select Relevant Medical Conditions",
    options=MEDICAL_CONDITIONS,
    default=None
)

additional_conditions = st.text_input("Add Additional Medical Conditions (comma-separated)")
if additional_conditions.strip():
    extra_conds = [cond.strip() for cond in additional_conditions.split(",") if cond.strip()]
    if selected_conditions is None:
        selected_conditions = []
    selected_conditions += extra_conds

if num_diagnoses > 25:
    st.warning("Are you sure about so many diagnoses?")
    if "diagnoses_confirmed" not in st.session_state:
        st.session_state["diagnoses_confirmed"] = False

    if not st.session_state["diagnoses_confirmed"]:
        confirm_choice = st.radio(
            "Confirm to proceed with this large number of diagnoses?",
            ["No", "Yes"],
            index=0
        )
        if st.button("Submit"):
            if confirm_choice == "Yes":
                st.session_state["diagnoses_confirmed"] = True
                st.rerun()
            else:
                st.warning("Diagnosis entry not confirmed. Exiting.")
                st.stop()
    else:
        st.warning("Large number of diagnoses confirmed. Proceed with caution.")

############################################
# PREDICT BUTTON
############################################
if st.button("Predict Readmission Risk"):
    if not selected_conditions:
        st.warning("Please select or add at least one medical condition to proceed.")
        st.stop()
        
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
        st.stop()

    try:
        X_input = initialize_features()
        X_input.at[0, 'age_at_admission'] = age
        X_input.at[0, 'num_diagnoses'] = num_diagnoses
        X_input.at[0, 'gender_encoded'] = 1 if gender == 'F' else 0

        admission_type_column = f"admission_type_{selected_admission_type}"
        if admission_type_column in FEATURE_NAMES:
            X_input.at[0, admission_type_column] = 1

        admission_location_column = f"admission_location_{selected_admission_location}"
        if admission_location_column in FEATURE_NAMES:
            X_input.at[0, admission_location_column] = 1

        insurance_column = f"insurance_{selected_insurance}"
        if insurance_column in FEATURE_NAMES:
            X_input.at[0, insurance_column] = 1

        language_column = f"language_{selected_language}"
        if language_column in FEATURE_NAMES:
            X_input.at[0, language_column] = 1

        marital_status_column = f"marital_status_{selected_marital_status}"
        if marital_status_column in FEATURE_NAMES:
            X_input.at[0, marital_status_column] = 1

        race_column = f"race_{selected_race}"
        if race_column in FEATURE_NAMES:
            X_input.at[0, race_column] = 1

        X_input.at[0, 'subject_id'] = 0
        X_input.at[0, 'hadm_id'] = 0
        X_input.at[0, 'hospital_expire_flag'] = 0

        dmatrix = xgb.DMatrix(X_input)
        y_pred = model.predict(dmatrix)
        readmission_prob = float(y_pred[0])

        st.write(f"**Predicted Probability of Readmission: {readmission_prob*100:.2f}%**")
        
        patient_context = f"A {age}-year-old patient with {num_diagnoses} diagnoses."
        print(selected_conditions)
        with st.spinner('Generating summary...'):
            summary = generate_summary(patient_context, selected_conditions)
        if summary:
            st.write("### RAG-based Summary")
            with st.expander("View Detailed Summary"):
                st.markdown(summary)
        else:
            st.error("No summary could be generated.")
        
        st.write("---")
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Use Again"):
                st.rerun()

        with col2:
            if st.button("Close Tool"):
                st.write("Session will be closed. Goodbye!")
                st.stop()

    except Exception as e:
        st.error(f"An error occurred during prediction or summary generation: {e}")
