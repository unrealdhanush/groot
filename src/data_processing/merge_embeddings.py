# src/data_processing/merge_embeddings.py

"""
Merge the previously engineered structured features with the discharge summary embeddings.

We assume:
- final_features.csv has hadm_id column.
- discharge_embeddings.parquet has columns [hadm_id, embedding_dim_1, ... embedding_dim_n].

We will:
- Load both files,
- Merge on hadm_id,
- Save a combined dataset for modeling.
"""

import os
import pandas as pd
from src.config.base_config import PROCESSED_DATA_DIR

def merge_embeddings(structured_file="final_features.csv", embeddings_file="discharge_embeddings.parquet", output_file="features_with_embeddings.csv"):
    structured_path = os.path.join(PROCESSED_DATA_DIR, structured_file)
    embeddings_path = os.path.join(PROCESSED_DATA_DIR, embeddings_file)
    
    if not os.path.exists(structured_path):
        raise FileNotFoundError(f"{structured_file} not found.")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"{embeddings_file} not found.")

    df_struct = pd.read_csv(structured_path)
    df_embed = pd.read_parquet(embeddings_path)

    # Merge on hadm_id
    df_merged = df_struct.merge(df_embed, on='hadm_id', how='left')
    
    # Some admissions might not have a discharge summary (or embedding)
    # We can either drop them or impute embeddings as zeros.
    # For simplicity, fill missing embeddings with 0:
    embedding_cols = [c for c in df_merged.columns if c not in df_struct.columns and c != 'hadm_id']
    for col in embedding_cols:
        df_merged[col].fillna(0, inplace=True)

    # Save merged dataset
    output_path = os.path.join(PROCESSED_DATA_DIR, output_file)
    df_merged.to_csv(output_path, index=False)
    print(f"Features with embeddings saved to {output_path}")

if __name__ == "__main__":
    merge_embeddings()
