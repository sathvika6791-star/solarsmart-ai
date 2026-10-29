"""
SunPredict: Solar Efficiency Intelligence
Prediction Module
Provides a clean, reusable inference interface with Split Conformal Prediction
uncertainty estimation, input validation, and plain-English explanations.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Add local directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    engineer_features,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)


class SunPredictor:
    """
    Inference and uncertainty quantification engine for SunPredict.
    """

    def __init__(self, models_dir=None):
        self.models_dir = self._resolve_models_dir(models_dir)
        self.pipeline_path = os.path.join(self.models_dir, "sunpredict_pipeline.joblib")
        self.calib_path = os.path.join(self.models_dir, "conformal_calibration.joblib")
        self.metadata_path = os.path.join(self.models_dir, "model_metadata.json")

        self.pipeline = None
        self.calib_data = None
        self.metadata = None

        self._load_artifacts()

    def _resolve_models_dir(self, custom_path=None):
        if custom_path and os.path.exists(custom_path):
            return custom_path

        candidates = [
            "models",
            "SunPredict/models",
            "../models",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.exists(os.path.join(c, "sunpredict_pipeline.joblib")):
                return os.path.abspath(c)

        # Fallback to default
        return os.path.abspath("models")

    def _load_artifacts(self):
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(
                f"Trained model artifact not found at '{self.pipeline_path}'. "
                f"Please execute 'python src/train.py' first to train the model."
            )

        self.pipeline = joblib.load(self.pipeline_path)

        if os.path.exists(self.calib_path):
            self.calib_data = joblib.load(self.calib_path)
        else:
            self.calib_data = None

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    def predict(self, raw_input, coverage_level=0.90):
        """
        Generates point efficiency prediction and statistically sound conformal interval.

        Args:
            raw_input (dict or pd.DataFrame): Solar cell device and materials properties.
            coverage_level (float): Desired confidence level (e.g. 0.80, 0.90, 0.95).

        Returns:
            dict: Structured prediction with point estimate, lower/upper bounds, interval width,
                  and beginner-friendly scientific explanation.
        """
        if isinstance(raw_input, dict):
            df_in = pd.DataFrame([raw_input])
        elif isinstance(raw_input, pd.DataFrame):
            df_in = raw_input.copy()
        else:
            raise TypeError("Input must be a Python dictionary or pandas DataFrame.")

        # Ensure all raw column keys are present (defaulting to sensible unknown/standard values)
        df_engineered = engineer_features(df_in)

        # Feature matrix
        required_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        for col in required_cols:
            if col not in df_engineered.columns:
                df_engineered[col] = "Unknown" if col in CATEGORICAL_FEATURES else np.nan

        X_input = df_engineered[required_cols]

        # 1. Point Prediction
        point_pred = float(self.pipeline.predict(X_input)[0])
        point_pred = max(0.0, point_pred)

        # 2. Conformal Prediction Interval
        margin_q = None
        lower_bound = None
        upper_bound = None
        interval_width = None
        cov_pct = int(round(coverage_level * 100))

        if self.calib_data and "quantiles" in self.calib_data:
            quantiles = self.calib_data["quantiles"]
            cov_key = str(cov_pct)

            if cov_key in quantiles:
                margin_q = quantiles[cov_key]["margin_q"]
            else:
                # Dynamic interpolation from nonconformity scores
                scores = self.calib_data["nonconformity_scores"]
                n = len(scores)
                q_level = min(1.0, np.ceil((n + 1) * coverage_level) / n)
                margin_q = float(np.quantile(scores, q_level, method="higher"))

            lower_bound = max(0.0, round(point_pred - margin_q, 2))
            upper_bound = min(45.0, round(point_pred + margin_q, 2))
            interval_width = round(upper_bound - lower_bound, 2)

        # 3. Plain English Explanation
        explanation = (
            f"Under split conformal prediction with exchangeable experimental conditions, "
            f"there is a {cov_pct}% statistical coverage guarantee that the true efficiency of a newly fabricated cell "
            f"with these material parameters falls between {lower_bound:.2f}% and {upper_bound:.2f}%. "
            f"Note: This is a prediction interval reflecting experimental fabrication variability, not an unconditional promise."
        )

        return {
            "predicted_pce": round(point_pred, 2),
            "unit": "%",
            "coverage_level": coverage_level,
            "coverage_level_pct": cov_pct,
            "margin_q": round(margin_q, 2) if margin_q is not None else None,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interval_width": interval_width,
            "explanation": explanation
        }


    def get_feature_importance(self, top_n=12):
        """
        Impurity-based importance of the fitted tree model, with one-hot columns
        summed back into the original material/process feature they came from.
        Returns None when the selected algorithm has no feature_importances_.
        """
        model = self.pipeline.named_steps["regressor"]
        if not hasattr(model, "feature_importances_"):
            return None
        names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        # Longest prefix first so e.g. "HTL_stack_sequence_x" never matches a shorter name
        parents = sorted(CATEGORICAL_FEATURES + NUMERICAL_FEATURES, key=len, reverse=True)
        grouped = {}
        for name, imp in zip(names, model.feature_importances_):
            parent = next((p for p in parents if name == p or name.startswith(p + "_")), name)
            grouped[parent] = grouped.get(parent, 0.0) + float(imp)
        ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [{"feature": k, "importance": v} for k, v in ranked]


# ---------------------------------------------------------------------------
# Helpers that connect the web form to the model (kept out of the UI code so
# they can be tested and reused without Streamlit).
# ---------------------------------------------------------------------------

UI_CATEGORICAL_KEYS = [
    "Cell_architecture", "Substrate_stack_sequence", "ETL_stack_sequence",
    "ETL_deposition_procedure", "Perovskite_composition_short_form",
    "Perovskite_additives_compounds", "Perovskite_deposition_procedure",
    "Perovskite_deposition_solvents", "HTL_stack_sequence", "HTL_deposition_procedure",
    "Backcontact_stack_sequence", "Backcontact_deposition_procedure",
]
UI_NUMERIC_KEYS = ["perovskite_thickness", "annealing_temp", "annealing_time", "backcontact_thickness"]
NO_ADDITIVE = "none"


def load_ui_options(models_dir):
    """Dropdown choices / numeric ranges written by train.py (None if absent)."""
    path = os.path.join(models_dir, "ui_options.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def default_selection(options):
    """Most common value of every dropdown and the median of every numeric field."""
    cat = options["categorical"]
    sel = {}
    for key in UI_CATEGORICAL_KEYS:
        values = [o["value"] for o in cat.get(key, [])]
        sel[key] = NO_ADDITIVE if key == "Perovskite_additives_compounds" else (values[0] if values else "Unknown")
    for key in UI_NUMERIC_KEYS:
        sel[key] = round(options["numeric"][key]["median"], 1)
    sel["quenching"] = True
    sel["encapsulation"] = False
    return sel


def validate_selection(sel, options):
    """
    Check a form selection. Returns (errors, warnings):
      errors   -> the prediction must not run
      warnings -> the prediction runs but may be unreliable
    """
    errors, warnings = [], []
    cat = options["categorical"]

    for key in UI_CATEGORICAL_KEYS:
        value = sel.get(key)
        if value in (None, ""):
            errors.append(f"'{key}' is missing.")
            continue
        known = {o["value"] for o in cat.get(key, [])}
        if key == "Perovskite_additives_compounds":
            known.add(NO_ADDITIVE)
        if value not in known:
            warnings.append(
                f"'{value}' is not a well-represented option for {key.replace('_', ' ')} "
                f"(fewer than 25 training examples). The model will treat it as an 'infrequent' category."
            )

    labels = {"perovskite_thickness": "Perovskite thickness (nm)", "annealing_temp": "Annealing temperature (deg C)",
              "annealing_time": "Annealing time (min)", "backcontact_thickness": "Back-contact thickness (nm)"}
    for key in UI_NUMERIC_KEYS:
        value = sel.get(key)
        if value is None or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            errors.append(f"{labels[key]} must be a positive number.")
            continue
        rng = options["numeric"][key]
        if value < rng["min"] or value > rng["max"]:
            warnings.append(
                f"{labels[key]} = {value:g} is outside the full range seen in training data "
                f"({rng['min']:g} to {rng['max']:g}). The prediction is an extrapolation."
            )
        elif value < rng["p01"] or value > rng["p99"]:
            warnings.append(
                f"{labels[key]} = {value:g} is in the extreme tail of the training data "
                f"(typical range {rng['p01']:g} to {rng['p99']:g}); expect lower reliability."
            )
    return errors, warnings


def selection_to_raw_input(sel, options):
    """Turn a validated form selection into the raw-column dict the model pipeline expects."""
    ions = options["composition_map"].get(
        sel["Perovskite_composition_short_form"], {"a": "Unknown", "b": "Unknown", "c": "Unknown"}
    )
    additive = sel["Perovskite_additives_compounds"]
    raw = {
        "Perovskite_composition_a_ions": ions["a"],
        "Perovskite_composition_b_ions": ions["b"],
        "Perovskite_composition_c_ions": ions["c"],
        "Perovskite_additives_compounds": "" if additive == NO_ADDITIVE else additive,
        "Perovskite_thickness": str(sel["perovskite_thickness"]),
        "Perovskite_deposition_thermal_annealing_temperature": str(sel["annealing_temp"]),
        "Perovskite_deposition_thermal_annealing_time": str(sel["annealing_time"]),
        "Backcontact_thickness_list": str(sel["backcontact_thickness"]),
        "Perovskite_deposition_number_of_deposition_steps": options.get("modal_deposition_steps", 1),
        "Perovskite_deposition_quenching_induced_crystallisation": bool(sel["quenching"]),
        "Encapsulation": bool(sel["encapsulation"]),
    }
    for key in UI_CATEGORICAL_KEYS:
        if key not in ("Perovskite_additives_compounds", "Perovskite_composition_short_form"):
            raw[key] = sel[key]
    raw["Perovskite_composition_short_form"] = sel["Perovskite_composition_short_form"]
    return raw


def predict_baseline_pce(predictor, sel, options, coverage_level=0.90):
    """
    Validate a form selection and run the saved ML pipeline.
    Raises ValueError (with every problem listed) when the input is invalid.
    Returns the prediction dict plus a 'warnings' list.
    """
    errors, warnings = validate_selection(sel, options)
    if errors:
        raise ValueError(" ".join(errors))
    result = predictor.predict(selection_to_raw_input(sel, options), coverage_level=coverage_level)
    result["warnings"] = warnings
    return result


def get_default_sample_cell():
    """Provides a realistic champion perovskite solar cell recipe for quick demo testing."""
    return {
        "Cell_architecture": "nip",
        "Substrate_stack_sequence": "SLG | FTO",
        "ETL_stack_sequence": "TiO2-c | TiO2-mp",
        "ETL_deposition_procedure": "Spray-pyrolys | Spin-coating",
        "Perovskite_composition_a_ions": "Cs; FA; MA",
        "Perovskite_composition_b_ions": "Pb",
        "Perovskite_composition_c_ions": "Br; I",
        "Perovskite_composition_short_form": "CsFAMAPbBrI",
        "Perovskite_additives_compounds": "SnF2",
        "Perovskite_thickness": "450",
        "Perovskite_deposition_number_of_deposition_steps": "1",
        "Perovskite_deposition_procedure": "Spin-coating",
        "Perovskite_deposition_solvents": "DMF; DMSO",
        "Perovskite_deposition_quenching_induced_crystallisation": True,
        "Perovskite_deposition_thermal_annealing_temperature": "100",
        "Perovskite_deposition_thermal_annealing_time": "30",
        "HTL_stack_sequence": "Spiro-MeOTAD",
        "HTL_deposition_procedure": "Spin-coating",
        "Backcontact_stack_sequence": "Au",
        "Backcontact_thickness_list": "80",
        "Backcontact_deposition_procedure": "Evaporation",
        "Encapsulation": False
    }


if __name__ == "__main__":
    predictor = SunPredictor()
    sample = get_default_sample_cell()
    print("=" * 60)
    print(" SUNPREDICT: INFERENCE & UNCERTAINTY TEST")
    print("=" * 60)
    for cov in [0.80, 0.90, 0.95]:
        res = predictor.predict(sample, coverage_level=cov)
        print(f"\nCoverage: {res['coverage_level_pct']}%")
        print(f"  Predicted PCE: {res['predicted_pce']}{res['unit']}")
        print(f"  Prediction Interval: [{res['lower_bound']}{res['unit']}, {res['upper_bound']}{res['unit']}] (Width: {res['interval_width']}{res['unit']})")
        print(f"  Margin (±q): ±{res['margin_q']}{res['unit']}")
    print("=" * 60)

