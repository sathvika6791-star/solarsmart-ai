"""
SunPredict: Solar Efficiency Intelligence
Model Training & Uncertainty Calibration Script
Trains candidate models, selects the best using validation metrics,
calibrates split conformal prediction intervals, evaluates on untouched test data,
saves evaluation figures, and exports joblib/JSON artifacts.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    prepare_data,
    build_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)

# Styling for scientific figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 0.8


def calculate_conformal_quantiles(nonconformity_scores, coverage_levels=(0.80, 0.90, 0.95)):
    """
    Computes split conformal prediction quantiles with finite-sample correction:
    q_level = ceil((n + 1) * (1 - alpha)) / n
    """
    n = len(nonconformity_scores)
    quantiles = {}
    for cov in coverage_levels:
        q_level = min(1.0, np.ceil((n + 1) * cov) / n)
        q_val = float(np.quantile(nonconformity_scores, q_level, method="higher"))
        quantiles[str(int(cov * 100))] = {
            "desired_coverage": cov,
            "quantile_level": float(q_level),
            "margin_q": q_val
        }
    return quantiles


def evaluate_conformal_coverage(y_true, y_pred, quantiles):
    """
    Evaluates empirical coverage and interval width on test samples.
    """
    coverage_results = {}
    for cov_key, q_info in quantiles.items():
        q = q_info["margin_q"]
        lower = np.clip(y_pred - q, 0.0, None)
        upper = np.clip(y_pred + q, None, 45.0)

        inside = (y_true >= lower) & (y_true <= upper)
        emp_coverage = float(np.mean(inside))
        avg_width = float(np.mean(upper - lower))

        coverage_results[cov_key] = {
            "desired_coverage_pct": int(cov_key),
            "empirical_coverage_pct": round(emp_coverage * 100, 2),
            "average_interval_width": round(avg_width, 2),
            "margin_q": round(q, 3)
        }
    return coverage_results


def plot_eda_distribution(y_train, output_dir):
    """Plots and saves the solar cell efficiency target distribution."""
    plt.figure(figsize=(8, 5))
    sns.histplot(y_train, kde=True, color="#2e7d32", bins=40, alpha=0.6)
    plt.axvline(np.mean(y_train), color="#d32f2f", linestyle="--", label=f"Mean: {np.mean(y_train):.2f}%")
    plt.axvline(np.median(y_train), color="#1976d2", linestyle="-.", label=f"Median: {np.median(y_train):.2f}%")
    plt.title("Solar Cell Power Conversion Efficiency (PCE) Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Power Conversion Efficiency - PCE (%)", fontsize=11)
    plt.ylabel("Frequency (Cell Counts)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    path = os.path.join(output_dir, "eda_target_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_model_comparison(comparison_df, output_dir):
    """Generates comparison bar chart across candidate models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = ["#9e9e9e", "#42a5f5", "#26a69a", "#43a047", "#8e24aa"]

    # MAE
    axes[0].bar(comparison_df["Model"], comparison_df["Val MAE"], color=colors[:len(comparison_df)], edgecolor="#333", alpha=0.85)
    axes[0].set_title("Validation MAE (Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("MAE (%)")
    axes[0].tick_params(axis="x", rotation=30)

    # RMSE
    axes[1].bar(comparison_df["Model"], comparison_df["Val RMSE"], color=colors[:len(comparison_df)], edgecolor="#333", alpha=0.85)
    axes[1].set_title("Validation RMSE (Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("RMSE (%)")
    axes[1].tick_params(axis="x", rotation=30)

    # R2
    axes[2].bar(comparison_df["Model"], comparison_df["Val R2"], color=colors[:len(comparison_df)], edgecolor="#333", alpha=0.85)
    axes[2].set_title("Validation R² (Higher is Better)", fontweight="bold")
    axes[2].set_ylabel("R² Score")
    axes[2].tick_params(axis="x", rotation=30)

    plt.suptitle("SunPredict: Candidate Model Benchmark Comparison", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_actual_vs_predicted(y_true, y_pred, output_dir):
    """Plots actual vs predicted efficiency on the untouched test set."""
    plt.figure(figsize=(7, 6.5))
    plt.scatter(y_true, y_pred, alpha=0.25, s=16, color="#1b5e20", edgecolors="none")
    lims = [0, max(y_true.max(), y_pred.max()) + 2]
    plt.plot(lims, lims, color="#d32f2f", linestyle="--", linewidth=1.8, label="Ideal 1:1 Parity")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("Actual vs. Predicted Solar Cell Efficiency (Untouched Test Set)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Actual Power Conversion Efficiency (%)", fontsize=11)
    plt.ylabel("Predicted Power Conversion Efficiency (%)", fontsize=11)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_residuals(y_true, y_pred, output_dir):
    """Plots residual distribution and residuals vs predicted values."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Residuals scatter
    axes[0].scatter(y_pred, residuals, alpha=0.25, s=15, color="#00897b")
    axes[0].axhline(0, color="#d32f2f", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residuals vs. Predicted Efficiency", fontweight="bold")
    axes[0].set_xlabel("Predicted Efficiency (%)")
    axes[0].set_ylabel("Residual (Actual - Predicted, %)")

    # Residuals histogram
    sns.histplot(residuals, kde=True, ax=axes[1], color="#0288d1", bins=40, alpha=0.6)
    axes[1].axvline(0, color="#d32f2f", linestyle="--", label="Zero Error")
    axes[1].set_title("Residual Distribution (Error Symmetry)", fontweight="bold")
    axes[1].set_xlabel("Residual (%)")
    axes[1].legend(frameon=True)

    plt.suptitle("SunPredict: Residual Diagnostics on Untouched Test Set", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "residuals_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_importance(model, preprocessor, output_dir):
    """Extracts top feature importances from tree-based model."""
    if not hasattr(model, "feature_importances_"):
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(20)

    # Clean up prefixes for display
    fi_df["Feature"] = (
        fi_df["Feature"]
        .str.replace("cat__", "")
        .str.replace("num__", "")
        .str.replace("Perovskite_", "Pvk_")
    )

    plt.figure(figsize=(9, 6.5))
    sns.barplot(data=fi_df, y="Feature", x="Importance", color="#2e7d32", edgecolor="#333", alpha=0.85)
    plt.title("Top 20 Drivers of Solar Cell Efficiency (Feature Importance)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Relative Importance (Gini Impurity Reduction)", fontsize=11)
    plt.ylabel("Material / Process Feature", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_uncertainty_intervals(y_true, y_pred, q_90, output_dir, n_samples=60):
    """Plots predictions with split conformal prediction intervals."""
    indices = np.random.RandomState(42).choice(len(y_true), size=min(n_samples, len(y_true)), replace=False)
    order = np.argsort(y_pred[indices])
    sorted_idx = indices[order]

    sub_true = y_true[sorted_idx]
    sub_pred = y_pred[sorted_idx]
    lower = np.clip(sub_pred - q_90, 0, None)
    upper = np.clip(sub_pred + q_90, None, 45)

    x = np.arange(len(sorted_idx))

    plt.figure(figsize=(11, 5.5))
    plt.fill_between(x, lower, upper, color="#81c784", alpha=0.35, label=f"90% Conformal Interval (±{q_90:.1f}%)")
    plt.plot(x, sub_pred, color="#2e7d32", marker="o", markersize=4, linestyle="-", linewidth=1.5, label="Predicted Efficiency")
    plt.scatter(x, sub_true, color="#c62828", s=25, zorder=5, label="Actual Efficiency")

    # Highlight coverage
    covered = (sub_true >= lower) & (sub_true <= upper)
    plt.title(
        f"Split Conformal Prediction Intervals (90% Coverage Level)\nSample Coverage: {np.mean(covered)*100:.1f}% ({sum(covered)}/{len(sub_true)} observed inside)",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.xlabel("Test Cells (Sorted by Predicted Efficiency)", fontsize=11)
    plt.ylabel("Solar Cell Efficiency PCE (%)", fontsize=11)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    path = os.path.join(output_dir, "uncertainty_intervals.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def run_training_pipeline(dataset_path=None, base_dir=None):
    """
    Main training execution function.
    """
    start_time = time.time()
    print("=" * 70)
    print(" SUNPREDICT: SOLAR EFFICIENCY INTELLIGENCE - TRAINING PIPELINE")
    print("=" * 70)

    # Resolve directories
    if base_dir is None:
        # Always write next to this project (parent of src/), whatever the working directory is
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Prepare data
    data = prepare_data(filepath=dataset_path, test_size=0.15, calib_size=0.15, random_state=42)
    X_train, y_train = data["X_train"], data["y_train"]
    X_calib, y_calib = data["X_calib"], data["y_calib"]
    X_test, y_test = data["X_test"], data["y_test"]

    # 2. Build preprocessor
    print("\n[SunPredict] Building Scikit-Learn Preprocessing Pipeline...")
    preprocessor = build_preprocessor(data["categorical_cols"], data["numerical_cols"])
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_calib_trans = preprocessor.transform(X_calib)
    X_test_trans = preprocessor.transform(X_test)
    print(f"[SunPredict] Feature transformation complete. Input dimensionality: {X_train_trans.shape[1]}")

    # 3. Model candidates definition
    candidates = {
        "Dummy (Baseline)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=20, random_state=42),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, max_depth=22, n_jobs=-1, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=22, n_jobs=-1, random_state=42)
    }

    # 4. Train candidates and benchmark on validation (calibration) set
    print("\n[SunPredict] Training & benchmarking candidate models on Validation Set...")
    benchmarks = []
    fitted_candidates = {}

    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train_trans, y_train)
        train_time = time.time() - t0

        val_preds = model.predict(X_calib_trans)
        mae = mean_absolute_error(y_calib, val_preds)
        rmse = root_mean_squared_error(y_calib, val_preds)
        r2 = r2_score(y_calib, val_preds)

        benchmarks.append({
            "Model": name,
            "Val MAE": round(mae, 3),
            "Val RMSE": round(rmse, 3),
            "Val R2": round(r2, 3),
            "Train Time (s)": round(train_time, 2)
        })
        fitted_candidates[name] = model
        print(f"  -> {name:<25}: Val MAE = {mae:.3f} | RMSE = {rmse:.3f} | R² = {r2:.3f} ({train_time:.1f}s)")

    benchmark_df = pd.DataFrame(benchmarks)

    # 5. Model Selection (Lowest MAE on validation set)
    best_row = benchmark_df.sort_values(by="Val MAE").iloc[0]
    best_name = best_row["Model"]
    best_model = fitted_candidates[best_name]
    print(f"\n[SunPredict] Selected Best Model: '{best_name}' (Val MAE: {best_row['Val MAE']:.3f}, Val R²: {best_row['Val R2']:.3f})")

    # 6. Assemble Full End-to-End Pipeline
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", best_model)
    ])

    # 7. Split Conformal Prediction Uncertainty Calibration
    print("\n[SunPredict] Calibrating Split Conformal Prediction Uncertainty on Calibration Set...")
    calib_preds = best_model.predict(X_calib_trans)
    nonconformity_scores = np.abs(y_calib - calib_preds)

    quantiles = calculate_conformal_quantiles(nonconformity_scores, coverage_levels=(0.80, 0.90, 0.95))
    print(f"  -> Calibration sample size: {len(nonconformity_scores):,}")
    for cov_k, q_i in quantiles.items():
        print(f"  -> {cov_k}% Coverage conformal margin q = ±{q_i['margin_q']:.2f}% PCE")

    # 8. Final Untouched Test Set Evaluation
    print("\n[SunPredict] Evaluating Champion Model on Untouched Test Set...")
    test_preds = best_model.predict(X_test_trans)
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_rmse = float(root_mean_squared_error(y_test, test_preds))
    test_r2 = float(r2_score(y_test, test_preds))

    conformal_coverage = evaluate_conformal_coverage(y_test, test_preds, quantiles)

    print(f"  === FINAL UNTOUCHED TEST RESULTS ===")
    print(f"  * Test MAE:  {test_mae:.3f} %")
    print(f"  * Test RMSE: {test_rmse:.3f} %")
    print(f"  * Test R²:   {test_r2:.3f}")
    for cov_k, res in conformal_coverage.items():
        print(f"  * {cov_k}% Target Coverage -> Empirical Test Coverage: {res['empirical_coverage_pct']:.2f}% (Avg Interval Width: {res['average_interval_width']:.2f}%)")

    # 9. Generate & Save Visualizations
    print("\n[SunPredict] Generating evaluation plots in 'outputs/'...")
    plot_eda_distribution(y_train, outputs_dir)
    plot_model_comparison(benchmark_df, outputs_dir)
    plot_actual_vs_predicted(y_test, test_preds, outputs_dir)
    plot_residuals(y_test, test_preds, outputs_dir)
    plot_feature_importance(best_model, preprocessor, outputs_dir)
    plot_uncertainty_intervals(y_test, test_preds, quantiles["90"]["margin_q"], outputs_dir)

    # 10. Save Artifacts with Joblib and JSON
    print("\n[SunPredict] Saving model artifacts in 'models/'...")
    pipeline_path = os.path.join(models_dir, "sunpredict_pipeline.joblib")
    calibration_path = os.path.join(models_dir, "conformal_calibration.joblib")
    joblib.dump(final_pipeline, pipeline_path, compress=3)

    calib_payload = {
        "nonconformity_scores": nonconformity_scores,
        "quantiles": quantiles,
        "test_coverage_results": conformal_coverage,
        "n_calibration_samples": int(len(nonconformity_scores)),
        "test_predictions_sample": test_preds[:100].tolist(),
        "test_actuals_sample": y_test[:100].tolist()
    }
    joblib.dump(calib_payload, calibration_path, compress=3)

    # Metadata JSON
    metadata = {
        "project": "SunPredict: Solar Efficiency Intelligence",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_name": best_name,
        "training_samples": int(len(X_train)),
        "calibration_samples": int(len(X_calib)),
        "test_samples": int(len(X_test)),
        "target_variable": "JV_default_PCE",
        "target_unit": "%",
        "test_metrics": {
            "mae": round(test_mae, 4),
            "rmse": round(test_rmse, 4),
            "r2": round(test_r2, 4)
        },
        "conformal_test_coverage": conformal_coverage,
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES
        },
        "elapsed_seconds": round(time.time() - start_time, 2)
    }

    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(models_dir, "candidate_metrics.json"), "w") as f:
        json.dump(benchmarks, f, indent=2)

    with open(os.path.join(models_dir, "dataset_summary.json"), "w") as f:
        json.dump(data["raw_summary"], f, indent=2)

    with open(os.path.join(models_dir, "ui_options.json"), "w") as f:
        json.dump(data["ui_options"], f, indent=2)

    print("=" * 70)
    print(f" TRAINING SUCCESSFULLY COMPLETED in {time.time() - start_time:.1f}s")
    print(f" Fitted Pipeline:    {pipeline_path}")
    print(f" Conformal Artifact: {calibration_path}")
    print("=" * 70)

    return {
        "pipeline": final_pipeline,
        "metadata": metadata,
        "benchmarks": benchmark_df,
        "test_metrics": metadata["test_metrics"],
        "conformal": conformal_coverage
    }


if __name__ == "__main__":
    run_training_pipeline()

