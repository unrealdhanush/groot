# src/nlp/embed_clinical_texts.py

"""
This script:
1. Loads the MIMIC NOTEEVENTS data.
2. Filters for discharge summaries linked to the admissions in our cohort.
3. Uses ClinicalBERT to embed these discharge summaries.
4. Produces a CSV (or parquet) file mapping hadm_id to embedding vectors.

We will:
- Load a pre-trained ClinicalBERT model from Hugging Face (e.g., "emilyalsentzer/Bio_ClinicalBERT").
- Tokenize each discharge summary.
- Obtain embeddings (CLS token or average pooling of token embeddings).
- Save a DataFrame with hadm_id and the embedding vector.
"""

import os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from src.config.base_config import PROCESSED_DATA_DIR
from src.data_processing.data_loader import load_notes

def embed_texts(model_name="fine_tuned_clinicalbert_mlm", 
                output_filename="discharge_embeddings.parquet",
                max_length=512,
                batch_size=16):
    """
    Embed discharge summaries using ClinicalBERT.
    
    Steps:
    - Load NOTEEVENTS and filter to discharge summaries.
    - For each hadm_id, possibly combine multiple discharge summaries if they exist (usually one per admission).
    - Tokenize and run through model in batches.
    - Extract embeddings (use CLS token representation or average of last hidden state).
    - Save results.
    """
    # Load notes
    discharge_summaries = load_notes()
    
    # Filter to discharge summaries (only in MIMIC-III)
    # discharge_summaries = notes_df[notes_df['CATEGORY'] == 'Discharge summary'].copy()
    
    # We assume each hadm_id has one main discharge summary. If multiple rows exist, concatenate text.
    discharge_summaries = discharge_summaries.groupby('hadm_id')['text'].apply(lambda x: " ".join(x)).reset_index()

    # Load the ClinicalBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Decide on device (CPU or GPU)
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    hadm_ids = discharge_summaries['hadm_id'].values
    texts = discharge_summaries['text'].values

    # Tokenize in batches
    # Note: If texts are very long, consider truncation or summarization.
    embeddings = []
    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx+batch_size]
        batch_texts = batch_texts.tolist()
        encoded = tokenizer.batch_encode_plus(batch_texts, 
                                               padding=True, 
                                               truncation=True,
                                               max_length=max_length,
                                               return_tensors='pt')
        encoded = {k: v.to(device) for k,v in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded)
            # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
            # We can take the CLS token representation (index 0)
            cls_embeddings = outputs.last_hidden_state[:,0,:]  # shape: [batch_size, hidden_size]
            cls_embeddings = cls_embeddings.to("mps")

        cls_embeddings = cls_embeddings.to("cpu").numpy()
        embeddings.append(cls_embeddings)

    # Concatenate all batches
    embeddings = pd.DataFrame(
        data = torch.tensor([item for batch in embeddings for item in batch]).numpy(),
        index = hadm_ids
    )
    embeddings.reset_index(inplace=True)
    embeddings.rename(columns={'index': 'hadm_id'}, inplace=True)

    # Save embeddings
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    embeddings.to_parquet(output_path, index=False)
    print(f"Discharge embeddings saved to {output_path}")

if __name__ == "__main__":
    embed_texts()
