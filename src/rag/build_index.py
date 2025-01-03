# src/rag/build_index.py

"""
1. Load the medical_knowledge.csv (KB).
2. Embed each row using a sentence transformer or ClinicalBERT.
3. Build a FAISS index for similarity search.
4. Save the index and the mapping (id -> text).
"""

import os
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from src.config.base_config import ROOT_DIR

def build_faiss_index(kb_file="data/knowledge_base/medical_knowledge.csv", model_name="sentence-transformers/all-MiniLM-L6-v2", index_file="faiss_index.bin", mapping_file="kb_mapping.csv"):
    # Load KB
    kb_path = os.path.join(ROOT_DIR, kb_file)
    df = pd.read_csv(kb_path)

    # Load embedding model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    def embed_texts(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded = {k: v.to(device) for k,v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    embeddings = embed_texts(df['text'].tolist())

    # Build FAISS index
    d = embeddings.shape[1]  # dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Save index
    index_path = os.path.join(ROOT_DIR, "data", "knowledge_base", index_file)
    faiss.write_index(index, index_path)

    # Save mapping (id->text)
    mapping_path = os.path.join(ROOT_DIR, "data", "knowledge_base", mapping_file)
    df.to_csv(mapping_path, index=False)
    print(f"FAISS index and mapping saved to {index_path} and {mapping_path}.")

if __name__ == "__main__":
    build_faiss_index()
