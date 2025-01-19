# G.R.O.O.T (Guided Readmission & Orchestrated Observation Text)

A comprehensive, **end-to-end** clinical decision support pipeline that:

1. **Predicts 30-day readmission risk** using both **structured** data (demographics, labs, diagnoses, etc.) and **text embeddings** from clinical notes.
2. Provides **Retrieval-Augmented Generation (RAG)** summaries to deliver short, user-friendly write-ups about patient risk factors, best practices, and guidelines.

## Table of Contents

- [G.R.O.O.T (Guided Readmission \& Orchestrated Observation Text)](#groot-guided-readmission--orchestrated-observation-text)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [What Is G.R.O.O.T?](#what-is-groot)
    - [Why This Matters](#why-this-matters)
  - [Data Requirements](#data-requirements)
  - [Project Structure](#project-structure)
  - [Quick Start](#quick-start)
  - [Key Scripts \& Pipelines](#key-scripts--pipelines)
  - [Streamlit Apps](#streamlit-apps)
  - [Contact \& Acknowledgments](#contact--acknowledgments)

---

## Overview

### What Is G.R.O.O.T?

- A toolkit for **hospital readmission prediction** based on real medical data (MIMIC-IV).  
- Integrates **structured features** and **clinical text** embeddings in either an XGBoost model or a **FusionModel** (PyTorch).  
- Employs a **RAG** pipeline to retrieve knowledge base snippets relevant to the patient’s conditions, then uses a language model to generate a **short summary** or guidance.

### Why This Matters

- **Hospitals** care about 30-day readmission metrics for cost reduction, improved patient care, and compliance (e.g., CMS guidelines).  
- **Combining structured data + text** from discharge summaries often yields more robust predictions and can highlight subtle risk factors.  
- **RAG Summaries** help interpret these risks, pointing to medical guidelines or best practices in an automated, user-friendly note.

---

## Data Requirements

- **MIMIC-IV** (version 3.1 or newer) downloaded from [PhysioNet](https://physionet.org/). After obtaining credentialed access, place CSV files under `data/raw/` in a directory structure consistent with MIMIC standards.  
- Optional: any **knowledge base** CSV for the RAG pipeline, such as `medical_knowledge.csv` with short guideline entries about conditions.

---

## Project Structure

Below is a high-level layout:

```plaintext
groot/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ knowledge_base/
├─ models/
│  ├─ baseline_xgb_model.json
│  └─ fusion_model.pt
├─ notebooks/
│  └─ EDA.ipynb
├─ src/
│  ├─ config/
│  │  └─ base_config.py
│  ├─ data_processing/
│  │  ├─ data_loader.py
│  │  ├─ preprocess.py
│  │  ├─ feature_engineering.py
│  │  └─ merge_embeddings.py
│  ├─ modeling/
│  │  ├─ baseline_model.py
│  │  ├─ fusion_model.py
│  │  └─ trainer.py
│  ├─ nlp/
│  │  ├─ embed_clinical_texts.py
│  │  └─ domain_adaptation.py
│  ├─ rag/
│  │  ├─ build_index.py
│  │  └─ generate_summaries.py
│  └─ explainability/
│     └─ explain.py
├─ .gitignore
├─ app_xgb.py
├─ app_fm.py
├─ requirements.txt
└─ README.md
```

**Key directories**:

- **data_processing**: Scripts to load and transform raw MIMIC data, create a labeled cohort, and engineer features.
- **modeling**: Code for building/training models:
  - `baseline_model.py` for XGBoost.
  - `fusion_model.py` + `trainer.py` for a neural net that fuses structured + text embeddings.
- **nlp**: Scripts to generate text embeddings, or adapt an NLP model to the domain (PEFT, domain adaptation).
- **rag**: Scripts that build a FAISS index on a knowledge base, then retrieve relevant entries for summarization (`generate_summaries.py`).
- **app_xgb.py** and **app_fm.py**: Two separate Streamlit apps:
  - One uses the XGBoost model on structured features.
  - The other uses a FusionModel that includes on-the-fly text embeddings (local LLM or huggingface models).

---

## Quick Start

1. **Create and Activate a Virtual Environment**  

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**  

   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain MIMIC-IV Data**  
   - Place MIMIC CSV files in `data/raw/` following the MIMIC folder structure.
4. **Preprocess & Engineer**  

   ```bash
   python src/data_processing/preprocess.py
   python src/data_processing/feature_engineering.py
   ```

   This yields `final_features.csv` in `data/processed`.
5. **Train a Baseline**  

   ```bash
   python src/modeling/baseline_model.py
   ```

   Or embed text + merge, then train the fusion model:

   ```bash
   python src/nlp/embed_clinical_texts.py
   python src/data_processing/merge_embeddings.py
   python src/modeling/trainer.py
   ```

6. **Run the App**  
   - For XGBoost:

     ```bash
     streamlit run app_xgb.py
     ```

   - For Fusion:

     ```bash
     streamlit run app_fm.py
     ```

Open your browser at the indicated URL (often `localhost:8501`).

---

## Key Scripts & Pipelines

1. **Data Preprocessing**  
   - `preprocess.py`: Creates a readmission label by checking if a next admission is within 30 days.
   - `feature_engineering.py`: Creates age/gender-encoded columns and cleans missing data.
2. **NLP Embedding**  
   - `embed_clinical_texts.py`: Grabs discharge notes from MIMIC and produces embeddings with e.g. ClinicalBERT or domain-adapted BERT.
   - `merge_embeddings.py`: Merges these embeddings with `final_features.csv` → `features_with_embeddings.csv`.
3. **Model Training**  
   - `baseline_model.py`: Trains an XGBoost classifier on structured data alone.
   - `trainer.py`: Trains a PyTorch-based `FusionModel` that fuses structured features + embeddings.
4. **RAG**  
   - `build_index.py`: Uses a `medical_knowledge.csv` to embed each row, then store in a FAISS vector index for quick retrieval.
   - `generate_summaries.py`: Embeds a user query (e.g. conditions), retrieves top matches, and uses a small LLM to produce a short textual summary of relevant guidelines.
5. **Explainability**  
   - `explain.py`: Uses SHAP for the baseline XGBoost model (and can be adapted to the Fusion model if needed).

---

## Streamlit Apps

- **`app_xgb.py`**:  
  1. Loads the XGBoost model (`baseline_xgb_model.json`)  
  2. Asks for basic structured inputs (age, diagnoses, etc.)  
  3. Predicts readmission probability  
  4. Optionally calls `generate_summaries` for a short RAG-based summary  

- **`app_fm.py`**:  
  1. Loads the `fusion_model.pt`  
  2. Asks for structured inputs + a discharge summary text area  
  3. Embeds the discharge summary on the fly (via `AutoTokenizer` + `AutoModel` from `transformers`)  
  4. Passes both structured + text embeddings into the FusionModel  
  5. Displays the predicted readmission probability + RAG summary  

---

## Contact & Acknowledgments

- **Authors**: [Dhanush Balakrishna](https://unrealdhanush.com/)
- **License**: MIT (or whichever license you prefer).
- **Acknowledgments**:
  - PhysioNet / MIT for providing the MIMIC-IV dataset.
  - Hugging Face for Transformers and tokenizers.
  - The PyTorch and XGBoost communities.
