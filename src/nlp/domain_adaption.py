# src/nlp/domain_adaptation_mlm.py (Option 2)

"""
Option 2: Standard MLM fine-tuning for domain adaptation with BertForMaskedLM,
without using PEFT. 
You can optionally freeze certain layers to reduce compute.
"""

import os
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset


def domain_adapt_model_mlm(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    output_dir="fine_tuned_clinicalbert_mlm",
    max_length=512,
    num_train_epochs=1,
    batch_size=8
):
    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # (Optional) Partial Freeze Example:
    for name, param in model.named_parameters():
        if "bert.encoder.layer.0" in name or "bert.encoder.layer.1" in name:
            param.requires_grad = False

    # 2. Load domain texts (e.g., discharge notes). Customize as needed: e.g. from MIMIC, let's just say we have a function load_notes() that returns a list of text. Below is a placeholder.
    from src.data_processing.data_loader import load_notes
    domain_texts = load_notes()  # list of strings or a DataFrame with a 'text' column

    domain_texts = domain_texts['text'].tolist()

    domain_dataset = Dataset.from_dict({'text': domain_texts})

    # 3. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=max_length, 
            return_special_tokens_mask=True
        )

    tokenized_dataset = domain_dataset.map(tokenize_function, batched=True, num_proc=1)

    # 4. Data Collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True,
        mlm_probability=0.15
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        do_train=True
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 7. Train
    trainer.train()

    # 8. Save Model
    trainer.save_model(output_dir)
    print(f"Domain-adapted model (MLM) saved to {output_dir}")


if __name__ == "__main__":
    domain_adapt_model_mlm()
