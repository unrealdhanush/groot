# src/modeling/trainer.py

"""
Trainer for Fusion Model:
1. Loads data with structured features and text embeddings.
2. Trains the FusionModel on binary classification task.
3. Saves the trained model and evaluation metrics.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from src.config.base_config import PROCESSED_DATA_DIR, RANDOM_SEED, MODELS_DIR
from src.modeling.fusion_model import FusionModel

def load_data(file_name="features_with_embeddings.csv", batch_size=32, test_size=0.2):
    """
    Loads data and creates DataLoaders for training and testing.
    Dynamically determines structured_dim and ensures text embeddings are valid.
    """
    file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_name} not found in {PROCESSED_DATA_DIR}")
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Identify embedding columns (columns named as numbers from 0 to 767)
    embedding_cols = [str(i) for i in range(768)]  # Convert numbers to strings to match column names
    if not set(embedding_cols).issubset(df.columns):
        raise ValueError("Text embedding columns (0-767) are missing from the dataset. Check your preprocessing step.")
    
    # Identify structured feature columns (exclude embeddings and other non-structured columns)
    structured_features = df.drop(columns=["subject_id", "hadm_id", "readmitted_within_30d"] + embedding_cols, errors="ignore")
    structured_features = structured_features.apply(pd.to_numeric, errors="coerce").fillna(0)
    structured_dim = structured_features.shape[1]  # Number of structured features

    # Extract embeddings and validate
    text_embeddings = df[embedding_cols].values.astype("float32")  # Ensure embeddings are float32
    targets = df["readmitted_within_30d"].values

    # Train/test split
    X_structured_train, X_structured_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        structured_features.values, text_embeddings, targets, test_size=test_size, random_state=RANDOM_SEED, stratify=targets
    )
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_structured_train, dtype=torch.float32),
        torch.tensor(X_text_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_structured_test, dtype=torch.float32),
        torch.tensor(X_text_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    print(f"Structured dimension: {structured_dim}, Embedding dimension: {text_embeddings.shape[1]}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, structured_dim

def train_model(train_loader, test_loader, structured_dim, embedding_dim, hidden_dims, epochs=20, lr=0.001, model_save_name="fusion_model.pt"):
    """
    Trains the FusionModel.
    """
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model = FusionModel(structured_dim, embedding_dim, hidden_dims).to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for structured_inputs, text_embeddings, targets in train_loader:
            structured_inputs, text_embeddings, targets = structured_inputs.to(device), text_embeddings.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(structured_inputs, text_embeddings).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation phase
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for structured_inputs, text_embeddings, targets in test_loader:
                structured_inputs, text_embeddings, targets = structured_inputs.to(device), text_embeddings.to(device), targets.to(device)
                outputs = model(structured_inputs, text_embeddings).squeeze()
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
        
        # Metrics
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, (torch.tensor(y_pred) > 0.5).int().numpy())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, model_save_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_loader, test_loader, structured_dim = load_data()
    train_model(
        train_loader, test_loader,
        structured_dim=structured_dim,
        embedding_dim=768,
        hidden_dims=[64, 32],
        epochs=20,
        lr=0.001
    )
