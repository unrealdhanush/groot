# test_model_loading.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "google/flan-t5-small"  # or "gpt-neo"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Loaded {model_name} successfully on device.")

if __name__ == "__main__":
    main()
