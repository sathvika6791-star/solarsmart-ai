"""
SunPredict: Solar Efficiency Intelligence
Preprocessing Module
Handles dataset loading, data cleaning, target leakage prevention,
feature engineering, and scikit-learn preprocessing pipelines.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Target variable name
TARGET_COL = "JV_default_PCE"

# Columns to strictly exclude to prevent target leakage and data snooping
LEAKAGE_COLS = [
    "JV_measured",       # Post-fabrication test indicator
    "JV_default_Voc",     # Open-circuit voltage (downstream electrical JV curve measurement)
    "JV_default_Jsc",     # Short-circuit current density (downstream electrical JV curve measurement)
    "JV_default_FF",      # Fill Factor (downstream electrical JV curve measurement)
    "Cell_stack_sequence" # Redundant full composite string (individual layers already separated)
]

# Primary categorical features
CATEGORICAL_FEATURES = [
    "Cell_architecture",
    "Substrate_stack_sequence",
    "ETL_stack_sequence",
    "ETL_deposition_procedure",
    "Perovskite_composition_a_ions",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_c_ions",
    "Perovskite_composition_short_form",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_solvents",
    "HTL_stack_sequence",
    "HTL_deposition_procedure",
    "Backcontact_stack_sequence",
    "Backcontact_deposition_procedure"
]

# Engineered and parsed numerical features
NUMERICAL_FEATURES = [
    "annealing_temp",
    "annealing_time",
    "perovskite_thickness",
    "backcontact_thickness",
    "deposition_steps",
    "quenching",
    "encapsulation",
    "has_additive",
    "is_lead_free",
    "is_mixed_cation",
    "is_mixed_halide"
]


def resolve_dataset_path(custom_path=None):
    """
    Automatically discovers the solar cell dataset file across common locations.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path

    search_candidates = [
        "dataset/Solar Cell data set2 (1).csv",
        "Solar Cell data set2 (1).csv",
        "../Solar Cell data set2 (1).csv",
        "../dataset/Solar Cell data set2 (1).csv",
        os.path.join(_PROJECT_ROOT, "dataset", "Solar Cell data set2 (1).csv"),
    ]

    for p in search_candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    # Search current directory and subdirectories for any CSV with solar in name
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".csv") and "solar" in f.lower():
                return os.path.abspath(os.path.join(root, f))

    raise FileNotFoundError(
        "Could not locate the solar cell dataset CSV file. "
        "Please place 'Solar Cell data set2 (1).csv' in the dataset/ directory."
    )


def parse_numeric_value(val):
    """
    Safely parses numeric values from text fields (e.g. '100', '70 >> 100.0', '1.0; 8.0').
    Extracts the peak/maximum numeric quantity.
    """
    if pd.isna(val):
        return np.nan
    nums = [float(x) for x in re.findall(r"(\d+\.?\d*)", str(val))]
    return max(nums) if nums else np.nan


def load_raw_dataset(filepath=None):
    """
    Loads raw CSV dataset with memory-efficient typing.
    """
    path = resolve_dataset_path(filepath)
    print(f"[SunPredict] Loading dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[SunPredict] Raw dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def inspect_dataset(df):
    """
    Generates structured dataset summary dictionary for logging and reporting.
    """
    summary = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "target_column": TARGET_COL,
        "target_null_count": int(df[TARGET_COL].isna().sum()) if TARGET_COL in df.columns else None,
        "exact_duplicates": int(df.duplicated().sum()),
        "missing_per_column": {col: int(df[col].isna().sum()) for col in df.columns}
    }
    if TARGET_COL in df.columns:
        pce = df[TARGET_COL].dropna()
        summary["pce_stats"] = {
            "valid_count": int(len(pce)),
            "mean": float(pce.mean()),
            "std": float(pce.std()),
            "min": float(pce.min()),
            "q25": float(pce.quantile(0.25)),
            "median": float(pce.median()),
            "q75": float(pce.quantile(0.75)),
            "max": float(pce.max())
        }
    return summary


def engineer_features(df):
    """
    Performs domain-specific feature engineering from materials & process parameters.
    """
    df_out = df.copy()

    # 1. Thermal annealing temperature (°C)
    if "Perovskite_deposition_thermal_annealing_temperature" in df_out.columns:
        df_out["annealing_temp"] = (
            df_out["Perovskite_deposition_thermal_annealing_temperature"]
            .apply(parse_numeric_value)
            .clip(0, 500)
        )
    else:
        df_out["annealing_temp"] = np.nan

    # 2. Thermal annealing time (min)
    if "Perovskite_deposition_thermal_annealing_time" in df_out.columns:
        df_out["annealing_time"] = (
            df_out["Perovskite_deposition_thermal_annealing_time"]
            .apply(parse_numeric_value)
            .clip(0, 300)
        )
    else:
        df_out["annealing_time"] = np.nan

    # 3. Perovskite absorber thickness (nm)
    if "Perovskite_thickness" in df_out.columns:
        df_out["perovskite_thickness"] = (
            df_out["Perovskite_thickness"]
            .apply(parse_numeric_value)
            .clip(10, 2000)
        )
    else:
        df_out["perovskite_thickness"] = np.nan

    # 4. Back-contact thickness (nm)
    if "Backcontact_thickness_list" in df_out.columns:
        df_out["backcontact_thickness"] = (
            df_out["Backcontact_thickness_list"]
            .apply(parse_numeric_value)
            .clip(10, 1000)
        )
    else:
        df_out["backcontact_thickness"] = np.nan

    # 5. Deposition steps
    if "Perovskite_deposition_number_of_deposition_steps" in df_out.columns:
        df_out["deposition_steps"] = (
            pd.to_numeric(df_out["Perovskite_deposition_number_of_deposition_steps"], errors="coerce")
            .clip(0, 10)
        )
    else:
        df_out["deposition_steps"] = 1.0

    # 6. Quenching crystallization (anti-solvent dripping / gas quenching)
    if "Perovskite_deposition_quenching_induced_crystallisation" in df_out.columns:
        df_out["quenching"] = df_out["Perovskite_deposition_quenching_induced_crystallisation"].astype(int)
    else:
        df_out["quenching"] = 0

    # 7. Device encapsulation
    if "Encapsulation" in df_out.columns:
        df_out["encapsulation"] = df_out["Encapsulation"].astype(int)
    else:
        df_out["encapsulation"] = 0

    # 8. Additives presence
    if "Perovskite_additives_compounds" in df_out.columns:
        non_additives = ["Undoped", "Unknown", "none", "nan", ""]
        has_add = (
            ~df_out["Perovskite_additives_compounds"].isna() &
            ~df_out["Perovskite_additives_compounds"].astype(str).str.strip().isin(non_additives)
        )
        df_out["has_additive"] = has_add.astype(int)
    else:
        df_out["has_additive"] = 0

    # 9. Lead-free formulation indicator
    if "Perovskite_composition_b_ions" in df_out.columns:
        df_out["is_lead_free"] = (
            ~df_out["Perovskite_composition_b_ions"].astype(str).str.contains("Pb")
        ).astype(int)
    else:
        df_out["is_lead_free"] = 0

    # 10. Multi-cation formulation indicator (e.g. Cs/FA/MA)
    if "Perovskite_composition_a_ions" in df_out.columns:
        df_out["is_mixed_cation"] = (
            df_out["Perovskite_composition_a_ions"].astype(str).str.contains(";")
        ).astype(int)
    else:
        df_out["is_mixed_cation"] = 0

    # 11. Multi-halide formulation indicator (e.g. Br/I)
    if "Perovskite_composition_c_ions" in df_out.columns:
        df_out["is_mixed_halide"] = (
            df_out["Perovskite_composition_c_ions"].astype(str).str.contains(";")
        ).astype(int)
    else:
        df_out["is_mixed_halide"] = 0

    return df_out


def prepare_data(filepath=None, test_size=0.15, calib_size=0.15, random_state=42):
    """
    End-to-end dataset preparation:
    - Loads raw data
    - Filters invalid/missing target PCE
    - Prevents target leakage
    - Performs feature engineering
    - Splits into Train (70%), Calibration (15%), and Test (15%)
    """
    df_raw = load_raw_dataset(filepath)

    if TARGET_COL not in df_raw.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset. "
            f"Available columns: {list(df_raw.columns)}"
        )

    # Filter out missing or physically impossible PCE values (0% to 40%)
    valid_mask = (
        df_raw[TARGET_COL].notna() &
        (df_raw[TARGET_COL] >= 0.0) &
        (df_raw[TARGET_COL] <= 40.0)
    )
    df_clean = df_raw[valid_mask].copy()
    print(f"[SunPredict] Filtered valid target rows: {len(df_clean):,} of {len(df_raw):,}")

    # Feature engineering
    df_engineered = engineer_features(df_clean)

    # Feature matrix X and target y
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df_engineered[feature_cols].copy()
    y = df_engineered[TARGET_COL].values

    # Step 1: Split into Train (70%) and Temp (30%)
    temp_size = test_size + calib_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state
    )

    # Step 2: Split Temp evenly into Calibration (15%) and Test (15%)
    calib_fraction = calib_size / temp_size
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, train_size=calib_fraction, random_state=random_state
    )

    # Remove exact duplicate rows only from the training set
    train_dedup_mask = ~X_train.duplicated()
    removed_dups = int((~train_dedup_mask).sum())
    X_train = X_train[train_dedup_mask].copy()
    y_train = y_train[train_dedup_mask]

    print(
        f"[SunPredict] Data split summary:\n"
        f"  - Training samples (deduplicated): {len(X_train):,} ({removed_dups} training duplicates removed)\n"
        f"  - Calibration samples (for uncertainty): {len(X_calib):,}\n"
        f"  - Untouched Test samples: {len(X_test):,}\n"
        f"  - Total features: {len(feature_cols)} ({len(CATEGORICAL_FEATURES)} categorical, {len(NUMERICAL_FEATURES)} numerical)"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_calib": X_calib,
        "y_calib": y_calib,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "categorical_cols": CATEGORICAL_FEATURES,
        "numerical_cols": NUMERICAL_FEATURES,
        "raw_summary": inspect_dataset(df_raw),
        "ui_options": build_ui_options(df_raw),
        "n_valid_target_rows": int(len(df_clean)),
    }


# Categories seen fewer times than this are grouped as "infrequent" by the encoder,
# so they are not offered as dropdown choices.
MIN_CATEGORY_COUNT = 25
# Values that carry no information for a dropdown.
_HIDDEN_CATEGORY_VALUES = {"Unknown", "nan", ""}
_NO_ADDITIVE_VALUES = {"Undoped", "Unknown", "none", "nan", ""}
NUMERIC_UI_COLUMNS = ["annealing_temp", "annealing_time", "perovskite_thickness", "backcontact_thickness"]


def build_ui_options(df_raw, min_count=MIN_CATEGORY_COUNT):
    """
    Builds the dropdown choices and numeric ranges used by the web app, derived
    from the rows that are actually used for training (valid PCE only).
    """
    valid = df_raw[
        df_raw[TARGET_COL].notna() & (df_raw[TARGET_COL] >= 0.0) & (df_raw[TARGET_COL] <= 40.0)
    ]
    eng = engineer_features(valid)

    categorical = {}
    for col in CATEGORICAL_FEATURES:
        counts = eng[col].dropna().astype(str).value_counts()
        categorical[col] = [
            {"value": v, "count": int(n)}
            for v, n in counts.items()
            if n >= min_count and v not in _HIDDEN_CATEGORY_VALUES
        ]

    # Additives: the model only uses "has an additive or not", but the user picks a named one.
    add_counts = eng["Perovskite_additives_compounds"].dropna().astype(str).str.strip().value_counts()
    categorical["Perovskite_additives_compounds"] = [
        {"value": v, "count": int(n)}
        for v, n in add_counts.items()
        if n >= min_count and v not in _NO_ADDITIVE_VALUES
    ]

    # A single composition dropdown fills the A/B/C ion columns consistently.
    composition_map = {}
    for short in [o["value"] for o in categorical["Perovskite_composition_short_form"]]:
        sub = eng[eng["Perovskite_composition_short_form"] == short]
        ions = {}
        for key, col in (("a", "Perovskite_composition_a_ions"),
                         ("b", "Perovskite_composition_b_ions"),
                         ("c", "Perovskite_composition_c_ions")):
            mode = sub[col].dropna().astype(str).mode()
            ions[key] = mode.iloc[0] if len(mode) else "Unknown"
        composition_map[short] = ions

    numeric = {}
    for col in NUMERIC_UI_COLUMNS:
        s = eng[col].dropna()
        numeric[col] = {
            "min": float(s.min()), "p01": float(s.quantile(0.01)),
            "median": float(s.median()), "p99": float(s.quantile(0.99)),
            "max": float(s.max()), "n_valid": int(len(s)),
            "missing_fraction": float(1 - len(s) / len(eng)),
        }

    return {
        "n_training_rows_considered": int(len(eng)),
        "categorical": categorical,
        "composition_map": composition_map,
        "numeric": numeric,
        "modal_deposition_steps": int(eng["deposition_steps"].mode().iloc[0]),
    }


def build_preprocessor(categorical_cols=CATEGORICAL_FEATURES, numerical_cols=NUMERICAL_FEATURES):
    """
    Constructs a robust scikit-learn ColumnTransformer:
    - Numerical: Median imputation + RobustScaler (resistant to manufacturing outliers)
    - Categorical: Constant imputation ('Unknown') + OneHotEncoder with infrequent grouping
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=25,
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        verbose_feature_names_out=False
    )

    return preprocessor

