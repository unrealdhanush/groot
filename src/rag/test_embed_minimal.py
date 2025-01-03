# test_embed_minimal.py

import torch
from transformers import AutoModel, AutoTokenizer

def test_embedding_forward(query, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")

    encoded = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)  # This is where forward pass occurs
    embedding = outputs.last_hidden_state.mean(dim=1)
    print("Embedding shape:", embedding.shape)

if __name__ == "__main__":
    test_embedding_forward("Testing forward pass for segmentation fault.")
