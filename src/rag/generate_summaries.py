
# src/rag/generate_summaries.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
import faiss
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################################
# 1. Cache Model Loading
############################################################
lru_cache(1)
def load_model(model_name="google/flan-t5-large"):
    """
    Load LLM model and tokenizer, placing them on CPU for consistent usage.
    Using Streamlit's cache_resource to avoid reloading each time.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        model.eval()
        logger.info(f"Loaded {model_name} on {device} with eval mode.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        return None, None

def run_model(prompt, max_length=1024, num_beams=5):
    """
    Use the cached LLM model to generate text from the prompt.
    """
    try:
        tokenizer, model = load_model()
        if tokenizer is None or model is None:
            logger.error("LLM model or tokenizer was not loaded correctly.")
            return "Could not load LLM model."
        # if tokenizer.eos_token is not None:
        #     tokenizer.pad_token = tokenizer.eos_token
        # else:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     model.resize_token_embeddings(len(tokenizer))
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
            # Generation
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=False
                )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Successfully generated summary via run_model.")
            return summary
        except Exception as e:
            logger.error(f"Runtime error generated during text generation: {e}")
            raise
    except Exception as e:
        logger.error(f"Error in run_model: {e}")
        return "An error occured while generating the summary"

############################################################
# 2. FAISS Index and Embedding
############################################################

def load_faiss_index(index_file="faiss_index.bin", mapping_file="kb_mapping.csv"):
    """
    Load the FAISS index and the corresponding mapping CSV.
    """
    try:
        from src.config.base_config import ROOT_DIR  
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
    Embed the query text using a sentence transformer model on CPU.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        model.eval()
        logger.info(f"Using device: {device} for embedding '{model_name}'.")

        model.eval()
        
        encoded = tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logger.info("encodings prepared")

        # Forward Pass
        with torch.inference_mode(): 
            outputs = model(**encoded)
        # Mean pooling over the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        logger.info("Generated query embedding for retrieval.")
        return embedding
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise

def retrieve_knowledge(query, top_k=3):
    """
    Retrieve the top_k relevant knowledge base entries. If none found, returns an empty DataFrame.
    """
    try:
        index, df_map = load_faiss_index()
        embedding = embed_query(query)
        D, I = index.search(embedding, top_k)
        if len(I[0]) == 0:
            logger.warning("FAISS search returned no indices.")
            return pd.DataFrame()
        retrieved_docs = df_map.iloc[I[0]]
        logger.info(f"Retrieved top {top_k} documents for query '{query}'.")
        return retrieved_docs
    except Exception as e:
        logger.error(f"Error during knowledge retrieval: {e}")
        return pd.DataFrame()

############################################################
# 3. Summarization Orchestrator
############################################################

def fallback_no_retrieval_summary(patient_context, conditions):
    """
    Fallback summary if no relevant knowledge base text was found.
    """
    cond_str = ", ".join(conditions)
    fallback_text = (
        f"No relevant knowledge base entries found for the conditions: {cond_str}. "
        f"Based on {patient_context}, clinicians should consider close monitoring for potential readmission."
    )
    return fallback_text

def generate_summary(patient_context, conditions):
    """
    Generate a summary from the retrieved knowledge base entries
    or use a fallback if no text is found.
    """
    try:
        query = " ".join(conditions).strip()
        if not query:
            logger.warning("No conditions given for summary; returning short fallback.")
            return "No conditions specified to generate a meaningful summary."

        retrieved = retrieve_knowledge(query)
        if retrieved.empty:
            logger.warning("No relevant KB docs retrieved; using fallback summary.")
            return fallback_no_retrieval_summary(patient_context, conditions)

        retrieved_texts = "\n".join(
            [str(txt) for txt in retrieved['text'].dropna().values]
        ).strip()
        if not retrieved_texts:
            logger.warning("Retrieved docs are empty or contain no text. Using fallback.")
            return fallback_no_retrieval_summary(patient_context, conditions)

        prompt = f"""
            The patient context is: {patient_context}
            Conditions of interest: {', '.join(conditions)}
            Knowledge base excerpts:
            {retrieved_texts}

            Please provide a structured summary with:
            1) Common complications.
            2) Recommended follow-up or interventions.
            3) Potential interactions between these conditions.
            4) Relevance to 30-day readmission.
            """

        summary = run_model(prompt)
        return summary or "No summary was returned."
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        return "Could not generate a summary due to an internal error."

if __name__ == "__main__":
    # Simple test
    patient_context = "A 65-year-old patient with 2 diagnoses"
    conditions = ['COPD', 'asthma']
    summary_result = generate_summary(patient_context, conditions)
    print("Generated Summary:\n", summary_result)