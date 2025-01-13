# app_fm.py

import sys
print(f"Streamlit running on Python {sys.executable}")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel

from src.config.base_config import MODELS_DIR
from src.rag.generate_summaries import generate_summary

############################################################
# Fusion Model Import
############################################################
from src.modeling.fusion_model import FusionModel

############################################################
# Basic Constants
############################################################

# Structured columns from your training dataset (excluding label).
# These are 15 columns of structured data (no placeholders).
STRUCTURED_COLUMNS = [
    "deathtime",
    "admission_type",
    "admit_provider_id",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "edregtime",
    "edouttime",
    "hospital_expire_flag",
    "num_diagnoses",
    "age_at_admission",
    "gender_encoded"
]

# The text embedding dimension expected by your fusion model.
# If you used a BERT-like model, typically 768. 
EMBED_DIM = 768

############################################################
# Additional Tools
############################################################

@st.cache_resource
def load_embedding_model(model_name="emilyalsentzer/Bio_ClinicalBERT"):
    """
    Load a sentence-transformer model for on-the-fly embeddings of the user-provided text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

def embed_text(
    text: str, 
    tokenizer, 
    model, 
    device, 
    max_length=512
) -> torch.Tensor:
    """
    Convert a piece of text into an embedding vector using mean pooling.
    Returns a tensor of shape [EMBED_DIM].
    If no text is provided, returns zeros.
    """
    if not text.strip():
        # If empty, return zeros.
        return torch.zeros((EMBED_DIM,), dtype=torch.float32)

    inputs = tokenizer(
        [text], 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # shape [hidden_size]
    return embedding.cpu()

############################################################
# Load the Fusion Model
############################################################

@st.cache_resource
def load_fusion_model(model_path):
    """
    Load the saved fusion model for inference.
    """
    try:
        # Must match training dims
        structured_dim = len(STRUCTURED_COLUMNS)   # 15
        embedding_dim = EMBED_DIM                 # e.g. 768

        # Suppose your hidden layers are [64, 32]
        model = FusionModel(
            structured_input_dim=structured_dim,
            embedding_dim=embedding_dim,
            hidden_dims=[64, 32]
        )

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        st.success("Fusion Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading fusion model: {e}")
        return None

############################################################
# Initialize Single-Row DataFrame
############################################################

def initialize_features() -> pd.DataFrame:
    """
    Create a single-row DataFrame for structured columns. 
    Note: We won't pre-allocate embedding columns, since we will do 
    on-the-fly text embedding. We'll pass embeddings as a separate tensor.
    """
    data = {}
    for col in STRUCTURED_COLUMNS:
        data[col] = 0  # numeric or empty
    return pd.DataFrame([data])

############################################################
# Streamlit UI
############################################################

model_path = os.path.join(MODELS_DIR, "fusion_model.pt")
model = load_fusion_model(model_path)

# Also load the text embedding model
tokenizer, embed_model, embed_device = load_embedding_model()

st.title("G.R.O.O.T (Guided Readmission & Orchestrated Observation Text)")
st.subheader("Fusion Model with On-the-Fly Embeddings")

############################################
# Basic Inputs for Structured Data
############################################
age = st.number_input("Age at Admission", min_value=0, max_value=120, value=65)
num_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=50, value=5)
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

discharge_locations = [
    "HOME", "SKILLED NURSING FACILITY", "REHAB", "ICF", "LONG TERM CARE",
    "DEAD/EXPIRED", "OTHER FACILITY", "AGAINST MEDICAL ADVICE"
]
selected_discharge_location = st.selectbox("Discharge Location", discharge_locations)

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

marital_statuses = ["MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "SEPARATED"]
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

# Additional text inputs for columns that aren't easily enumerated
admit_provider_id = st.text_input("Admitting Provider ID", "PROV1234")
edregtime = st.text_input("ED Registration Time", "2185-01-01 08:00:00")
edouttime = st.text_input("ED Out Time", "2185-01-01 10:30:00")

############################################
# Discharge Summary for On-the-Fly Embedding
############################################
discharge_summary = st.text_area(
    "Discharge Summary Text", 
    "Patient was admitted for chest pain, treated for possible myocardial infarction..."
)

############################################
# Pediatric & High-Diagnoses Warnings
############################################
if age < 12:
    if "parental_mode_accepted" not in st.session_state:
        st.session_state["parental_mode_accepted"] = False
    if not st.session_state["parental_mode_accepted"]:
        st.warning("Parental Mode activated! Pipeline is primarily adult-focused. Accept to proceed.")
        if st.button("Accept Parental Mode"):
            st.session_state["parental_mode_accepted"] = True
            st.rerun()
        else:
            st.stop()
    else:
        st.warning("Parental Mode active! Pediatric predictions may be less reliable.")

if num_diagnoses > 25:
    st.warning("Are you sure about so many diagnoses?")
    if "diagnoses_confirmed" not in st.session_state:
        st.session_state["diagnoses_confirmed"] = False
    if not st.session_state["diagnoses_confirmed"]:
        confirm_choice = st.radio("Confirm high diagnoses count?", ["No", "Yes"], index=0)
        if st.button("Submit"):
            if confirm_choice == "Yes":
                st.session_state["diagnoses_confirmed"] = True
                st.rerun()
            else:
                st.warning("Diagnosis entry not confirmed. Exiting.")
                st.stop()
    else:
        st.warning("High diagnoses count confirmed. Proceed with caution.")

############################################
# Conditions for RAG Summaries
############################################
MEDICAL_CONDITIONS = [
    "heart failure", "diabetes mellitus", "chronic kidney disease", "hypertension",
    "COPD", "pneumonia", "sepsis", "myocardial infarction", "stroke",
    "depression", "anxiety", "asthma", "obesity", "cancer", "arthritis"
]

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


############################################
# Predict Button
############################################
if st.button("Predict Readmission Risk"):
    if model is None:
        st.error("Fusion Model is not loaded. Cannot make predictions.")
        st.stop()

    # Build structured input row
    X_input = initialize_features()

    X_input.at[0, "deathtime"] = ""
    X_input.at[0, "admission_type"] = selected_admission_type
    X_input.at[0, "admit_provider_id"] = admit_provider_id
    X_input.at[0, "admission_location"] = selected_admission_location
    X_input.at[0, "discharge_location"] = selected_discharge_location
    X_input.at[0, "insurance"] = selected_insurance
    X_input.at[0, "language"] = selected_language
    X_input.at[0, "marital_status"] = selected_marital_status
    X_input.at[0, "race"] = selected_race
    X_input.at[0, "edregtime"] = edregtime
    X_input.at[0, "edouttime"] = edouttime
    X_input.at[0, "hospital_expire_flag"] = 0
    X_input.at[0, "num_diagnoses"] = num_diagnoses
    X_input.at[0, "age_at_admission"] = age
    X_input.at[0, "gender_encoded"] = 1 if gender == 'F' else 0

    # Convert categorical to numeric codes (quick fix)
    structured_df = X_input[STRUCTURED_COLUMNS].copy()
    for col in structured_df.columns:
        if structured_df[col].dtype == 'object':
            structured_df[col] = structured_df[col].astype('category').cat.codes
    
    try:
        # Convert structured to Torch
        structured_tensor = torch.tensor(structured_df.values, dtype=torch.float32)

        # On-the-fly embed the discharge summary
        text_emb = embed_text(discharge_summary, tokenizer, embed_model, embed_device)
        # Unsqueeze to shape [1, EMBED_DIM]
        text_emb_tensor = text_emb.unsqueeze(0)

        with torch.no_grad():
            logits = model(structured_tensor, text_emb_tensor).squeeze().item()
            readmission_prob = torch.sigmoid(torch.tensor(logits)).item()

        st.write(f"**Predicted Probability of Readmission**: {readmission_prob*100:.2f}%")

        # RAG-based summary
        if not selected_conditions:
            st.info("No medical conditions selected for summary generation.")
        else:
            with st.spinner("Generating summary..."):
                patient_context = f"A {age}-year-old patient with {num_diagnoses} diagnoses."
                summary = generate_summary(patient_context, selected_conditions)
            if summary:
                st.write("### RAG-based Summary")
                with st.expander("View Detailed Summary"):
                    st.markdown(summary)
            else:
                st.info("No summary generated.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
