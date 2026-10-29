"""
SunPredict: Solar Efficiency Intelligence
Model Evaluation & Conformal Verification Module
Provides standalone validation and test set auditing,
computing MAE, RMSE, R², and empirical conformal coverage rates.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Add local directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_data


def run_evaluation_report(models_dir=None, dataset_path=None):
    """
    Loads saved model artifacts, runs evaluation on untouched test data,
    and prints a thorough benchmark and uncertainty verification report.
    """
    if models_dir is None:
        if os.path.exists("SunPredict/models"):
            models_dir = "SunPredict/models"
        else:
            models_dir = "models"

    pipeline_path = os.path.join(models_dir, "sunpredict_pipeline.joblib")
    calib_path = os.path.join(models_dir, "conformal_calibration.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Missing pipeline artifact at {pipeline_path}. Run train.py first.")

    pipeline = joblib.load(pipeline_path)
    calib_data = joblib.load(calib_path) if os.path.exists(calib_path) else None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    print("=" * 75)
    print(" SUNPREDICT: MODEL EVALUATION & CONFORMAL UNCERTAINTY VERIFICATION")
    print("=" * 75)
    print(f" Model Architecture:      {meta.get('best_model_name')}")
    print(f" Training Samples:        {meta.get('training_samples'):,}")
    print(f" Calibration Samples:     {meta.get('calibration_samples'):,}")
    print(f" Test Samples:            {meta.get('test_samples'):,}")
    print("-" * 75)

    print("\n[SunPredict] Loading untouched test partition from dataset...")
    data = prepare_data(filepath=dataset_path, test_size=0.15, calib_size=0.15, random_state=42)
    X_test, y_test = data["X_test"], data["y_test"]

    print("[SunPredict] Generating predictions on untouched test set...")
    test_preds = pipeline.predict(X_test)

    # Standard metrics
    mae = mean_absolute_error(y_test, test_preds)
    rmse = root_mean_squared_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)

    print("\n--- Point Prediction Accuracy Metrics ---")
    print(f"  * Mean Absolute Error (MAE):       {mae:.3f} % PCE")
    print(f"  * Root Mean Squared Error (RMSE):  {rmse:.3f} % PCE")
    print(f"  * Coefficient of Determination (R²): {r2:.3f}")

    # Conformal Uncertainty Coverage
    if calib_data and "quantiles" in calib_data:
        print("\n--- Split Conformal Prediction Verification ---")
        quantiles = calib_data["quantiles"]
        print(f"{'Coverage Target':<18} | {'Empirical Test Coverage':<25} | {'Margin (±q)':<15} | {'Avg Width':<12}")
        print("-" * 75)
        for cov_k, q_info in quantiles.items():
            q = q_info["margin_q"]
            lower = np.clip(test_preds - q, 0.0, None)
            upper = np.clip(test_preds + q, None, 45.0)
            emp_cov = np.mean((y_test >= lower) & (y_test <= upper)) * 100
            avg_w = np.mean(upper - lower)
            print(f"{cov_k + '%':<18} | {emp_cov:.2f}%{'':<18} | ±{q:.2f}%{'':<9} | {avg_w:.2f}%")

    print("\n" + "=" * 75)
    print(" EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    run_evaluation_report()

