# src/explainability/explain.py

"""
Use SHAP to explain model predictions.
We will:
1. Load the trained model and the final dataset with embeddings.
2. Split into test data as previously done.
3. Compute SHAP values.
4. Save a summary plot or a few example explanations.

Note: SHAP can be computationally expensive, so consider using a sample of test data.
"""

import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from src.config.base_config import PROCESSED_DATA_DIR, TEST_SIZE, RANDOM_SEED

def generate_shap_values(input_filename="features_with_embeddings.csv", model_filename="baseline_xgb_model.json"):
    # Load data
    input_path = os.path.join(PROCESSED_DATA_DIR, input_filename)
    model_path = os.path.join(PROCESSED_DATA_DIR, model_filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_filename} not found in {PROCESSED_DATA_DIR}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_filename} not found in {PROCESSED_DATA_DIR}")

    df = pd.read_csv(input_path)
    X = df.drop(columns=['readmitted_within_30d'])
    cols_to_drop = ["deathtime", "admit_provider_id", "discharge_location", "edregtime", "edouttime"]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    X = pd.get_dummies(X, columns=["admission_type", "admission_location", "insurance", "language", "marital_status", "race"],drop_first=True)
    y = df['readmitted_within_30d']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    
    # Load model
    model = xgb.Booster()
    model.load_model(model_path)
    feature_names = model.feature_names
    X_test = X_test.reindex(columns=feature_names)
    # SHAP values for XGBoost
    # Convert data to DMatrix
    dtest = xgb.DMatrix(X_test)

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Save shap values if needed, or directly create a summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    # To save the plot, you can do:
    shap_plot_path = os.path.join(PROCESSED_DATA_DIR, "shap_summary.png")
    plt.savefig(shap_plot_path)
    print(f"SHAP summary plot saved at {shap_plot_path}")

if __name__ == "__main__":
    generate_shap_values()
