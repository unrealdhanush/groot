# src/modeling/baseline_model.py

"""
This script trains a baseline model (e.g., XGBoost) on the structured features.
Steps:
1. Load final_features.csv
2. Split into train/test.
3. Train an XGBoost model on these features.
4. Evaluate and print performance metrics.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb

from src.config.base_config import PROCESSED_DATA_DIR, RANDOM_SEED, TEST_SIZE

def train_baseline_model(input_filename="final_features.csv"):
    input_path = os.path.join(PROCESSED_DATA_DIR, input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_filename} not found in {PROCESSED_DATA_DIR}")

    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=['readmitted_within_30d'])
    cols_to_drop = ["deathtime", "admit_provider_id", "discharge_location", "edregtime", "edouttime"]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    X = pd.get_dummies(X, columns=["admission_type", "admission_location", "insurance", "language", "marital_status", "race"],drop_first=True)
    y = df['readmitted_within_30d']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Basic hyperparameters for a quick baseline
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': RANDOM_SEED
    }

    # Train model
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=10, evals=evals)

    # Predictions
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Evaluate
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Baseline Model AUC: {auc:.4f}")
    print(f"Baseline Model Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    # Save the model
    model.save_model(os.path.join(PROCESSED_DATA_DIR, "baseline_xgb_model.json"))
    print("Model saved.")

if __name__ == "__main__":
    train_baseline_model()
