# src/rag/generate_summaries.py

"""
Given a patient's data:
1. Identify key conditions (e.g., top diagnoses or features from the model).
2. Use FAISS to retrieve the most relevant KB entries.
3. Use a small LLM to generate a summary explaining risk factors and conditions.

For simplicity, let's say we already know the patient's conditions (top ICD codes or from SHAP analysis).
We:
- Convert conditions to embedding and query FAISS.
- Retrieve top K results.
- Prompt a small LLM (or a locally fine-tuned LLaMA2 / GPT model if available) to produce a summary.
"""

import faulthandler
faulthandler.enable()

import os
import pandas as pd
import torch
import faiss
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from src.config.base_config import ROOT_DIR

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def load_faiss_index(index_file="faiss_index.bin", mapping_file="kb_mapping.csv"):
    """
    Load the FAISS index and the corresponding mapping CSV.
    """
    try:
        index_path = os.path.join(ROOT_DIR, "data", "knowledge_base", index_file)
        mapping_path = os.path.join(ROOT_DIR, "data", "knowledge_base", mapping_file)
        index = faiss.read_index(index_path)
        df_map = pd.read_csv(mapping_path)
        logger.info(f"Loaded FAISS index from {index_path} and mapping from {mapping_path}.")
        return index, df_map
    except Exception as e:
        logger.error(f"Error loading FAISS index or mapping: {e}")
        raise

def embed_query(query, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Embed the query using a sentence transformer model.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info(f"Using device: {device} for embedding.")
        
        encoded = tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
        
        # For sentence-transformers models, the [CLS] token or mean pooling can be used
        # Here, we'll use mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        logger.info("Generated query embedding.")
        return embedding
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise

def retrieve_knowledge(query, top_k=3):
    """
    Retrieve top_k knowledge base entries relevant to the query.
    """
    try:
        index, df_map = load_faiss_index()
        embedding = embed_query(query)
        D, I = index.search(embedding, top_k)
        # I are the indices of retrieved docs
        retrieved_docs = df_map.iloc[I[0]]
        logger.info(f"Retrieved top {top_k} documents.")
        return retrieved_docs
    except Exception as e:
        logger.error(f"Error during knowledge retrieval: {e}")
        raise

def generate_summary(patient_context, conditions):
    """
    Generate a summary based on patient context and conditions.
    """
    try:
        # Create a query by joining conditions
        query = " ".join(conditions)
        logger.info(f"Generated query from conditions: {query}")
        
        # Retrieve relevant knowledge base entries
        retrieved = retrieve_knowledge(query)
        retrieved_texts = "\n".join(retrieved['text'].values)
        logger.info("Retrieved knowledge base texts for summary generation.")
        
        # Construct the prompt
        prompt = f"""
        Given the patient's context: {patient_context}
        And the following medical knowledge:
        {retrieved_texts}

        Provide a short, user-friendly summary explaining the key risk factors and their significance.
        """
        
        # Choose an instruction-tuned model for better summaries
        model_name = "google/flan-t5-small"  # You can choose other models like "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        model.eval()
        logger.info(f"Loaded generative model '{model_name}' on device {device}.")

        # Encode the prompt
        inputs = tokenizer.encode(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)
        logger.info("Encoded the prompt for generation.")

        # Generate the summary
        outputs = model.generate(
            inputs,
            max_length=512,          # Adjust based on model's max_length
            num_beams=5,
            early_stopping=True
        )
        logger.info("Generated summary using the language model.")

        # Decode the generated text
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Decoded the generated summary.")
        
        return summary
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        raise

if __name__ == "__main__":
    patient_context = "A 65-year-old patient with a history of heart failure and diabetes, predicted high risk of readmission."
    conditions = ["heart failure", "diabetes mellitus"]  # Add more conditions as needed
    summary = generate_summary(patient_context, conditions)
    print("Generated Summary:\n", summary)