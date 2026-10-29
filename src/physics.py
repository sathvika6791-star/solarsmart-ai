"""
SolarSmart AI
Physics-based adjustments (NOT machine learning).

This module holds the idealised incidence-angle and shading estimates and the
combined scenario calculation. Nothing here is learned from the dataset: the
dataset contains no measured angle or shading variables, so these are clearly
labelled approximations driven by config/solar_assumptions.json.

Combined model (each factor is applied exactly once):

    effective_PCE = baseline_PCE x angle_factor x shading_factor
    power_W       = irradiance x area x effective_PCE / 100

`irradiance` is the beam irradiance measured perpendicular to the sun's rays
(W/m^2). The cosine law, which converts it to the irradiance that actually
reaches the tilted panel surface, lives only in `angle_factor`, so it is not
double counted in the power calculation.
"""

import copy
import json
import math
import os

import numpy as np

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "solar_assumptions.json",
)

# Used only if the JSON file is missing, so the app can still start.
_FALLBACK_ASSUMPTIONS = {
    "angle": {"model": "cosine", "diffuse_fraction": 0.0},
    "shading": {
        "model": "bypass_diode",
        "available_models": ["linear", "bypass_diode", "power_law"],
        "linear": {},
        "bypass_diode": {"n_substrings": 3, "activation_shaded_fraction": 0.5},
        "power_law": {"exponent": 1.5},
    },
    "defaults": {"angle_deg": 0, "shading_pct": 0, "irradiance_w_m2": 1000, "panel_area_m2": 1.6},
    "limits": {
        "angle_deg": [0, 90],
        "shading_pct": [0, 100],
        "irradiance_w_m2": [0, 1500],
        "panel_area_m2": [0.01, 100],
    },
}

SHADING_MODEL_LABELS = {
    "linear": "Linear (loss = shaded area)",
    "bypass_diode": "Bypass-diode substrings",
    "power_law": "Series-limited power law",
}


def load_assumptions(path=None):
    """Load the adjustable assumptions (a fresh deep copy each call)."""
    path = path or _CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(_FALLBACK_ASSUMPTIONS)


def _check_range(name, value, low, high):
    if value is None or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}.")


def angle_factor(angle_deg, assumptions=None):
    """
    Fraction of the direct-beam power that reaches the panel for a given
    incidence angle (angle between the sunlight and the panel normal).
    0 deg = perpendicular, 90 deg = grazing (zero in the pure cosine model).
    """
    a = assumptions or load_assumptions()
    _check_range("Incidence angle (deg)", angle_deg, 0, 90)
    d = float(a["angle"].get("diffuse_fraction", 0.0))
    if not 0.0 <= d <= 1.0:
        raise ValueError("diffuse_fraction must be between 0 and 1.")
    cos_val = math.cos(math.radians(angle_deg))
    if angle_deg >= 90:
        cos_val = 0.0  # avoid the 6e-17 floating-point residue at exactly 90 deg
    return max(0.0, (1.0 - d) * cos_val + d)


def shading_factor(shading_pct, assumptions=None, model=None):
    """
    Fraction of unshaded output that remains for a given shaded area (%).
    This is a configurable APPROXIMATION, not a validated prediction: real
    losses depend on wiring, bypass diodes, cell layout and shadow position.
    """
    a = assumptions or load_assumptions()
    _check_range("Shading (%)", shading_pct, 0, 100)
    s = shading_pct / 100.0
    model = model or a["shading"]["model"]

    if model == "linear":
        factor = 1.0 - s
    elif model == "bypass_diode":
        p = a["shading"]["bypass_diode"]
        n = int(p.get("n_substrings", 3))
        thr = float(p.get("activation_shaded_fraction", 0.5))
        if n < 1:
            raise ValueError("n_substrings must be at least 1.")
        if not 0.0 < thr <= 1.0:
            raise ValueError("activation_shaded_fraction must be in (0, 1].")
        total = 0.0
        for k in range(n):
            f_k = min(1.0, max(0.0, s * n - k))  # shaded fraction of substring k
            total += 0.0 if f_k >= thr else (1.0 - f_k)
        factor = total / n
    elif model == "power_law":
        exponent = float(a["shading"]["power_law"].get("exponent", 1.5))
        if exponent <= 0:
            raise ValueError("exponent must be positive.")
        factor = (1.0 - s) ** exponent
    else:
        raise ValueError(f"Unknown shading model '{model}'.")

    return min(1.0, max(0.0, factor))


def bypass_substring_states(shading_pct, assumptions=None):
    """
    For the bypass-diode model: shaded fraction and state of each substring
    ('active', 'partial' or 'bypassed'). Used for the panel illustration.
    """
    a = assumptions or load_assumptions()
    _check_range("Shading (%)", shading_pct, 0, 100)
    p = a["shading"]["bypass_diode"]
    n = int(p.get("n_substrings", 3))
    thr = float(p.get("activation_shaded_fraction", 0.5))
    s = shading_pct / 100.0
    states = []
    for k in range(n):
        f_k = min(1.0, max(0.0, s * n - k))
        state = "bypassed" if f_k >= thr else ("partial" if f_k > 0 else "active")
        states.append({"shaded_fraction": f_k, "state": state})
    return states


def compute_scenario(baseline_pce, angle_deg, shading_pct,
                     irradiance_w_m2=None, panel_area_m2=None, assumptions=None):
    """
    Combine the ML baseline PCE (%) with the physics-based angle and shading
    estimates. Returns a plain dict; raises ValueError on invalid input.
    """
    a = assumptions or load_assumptions()
    irradiance_w_m2 = a["defaults"]["irradiance_w_m2"] if irradiance_w_m2 is None else irradiance_w_m2
    panel_area_m2 = a["defaults"]["panel_area_m2"] if panel_area_m2 is None else panel_area_m2

    _check_range("Baseline PCE (%)", baseline_pce, 0, 100)
    lo, hi = a["limits"]["irradiance_w_m2"]
    _check_range("Irradiance (W/m^2)", irradiance_w_m2, lo, hi)
    lo, hi = a["limits"]["panel_area_m2"]
    _check_range("Panel area (m^2)", panel_area_m2, lo, hi)

    f_angle = angle_factor(angle_deg, a)
    f_shade = shading_factor(shading_pct, a)

    angle_adjusted = baseline_pce * f_angle
    shading_adjusted = baseline_pce * f_shade
    effective = baseline_pce * f_angle * f_shade

    power_w = irradiance_w_m2 * panel_area_m2 * effective / 100.0
    reference_power_w = irradiance_w_m2 * panel_area_m2 * baseline_pce / 100.0

    return {
        "baseline_pce": baseline_pce,
        "angle_deg": angle_deg,
        "shading_pct": shading_pct,
        "shading_model": a["shading"]["model"],
        "irradiance_w_m2": irradiance_w_m2,
        "panel_area_m2": panel_area_m2,
        "angle_factor": f_angle,
        "shading_factor": f_shade,
        "angle_adjusted_pce": angle_adjusted,
        "shading_adjusted_pce": shading_adjusted,
        "effective_pce": effective,
        "abs_change_pp": effective - baseline_pce,
        "rel_change_pct": ((effective - baseline_pce) / baseline_pce * 100.0) if baseline_pce > 0 else 0.0,
        "power_w": power_w,
        "reference_power_w": reference_power_w,
        "power_loss_w": reference_power_w - power_w,
    }


def angle_curve(baseline_pce, shading_pct, assumptions=None, step=1):
    """Effective PCE for every incidence angle 0..90 at a fixed shading level."""
    a = assumptions or load_assumptions()
    angles = np.arange(0, 90 + step, step)
    f_shade = shading_factor(shading_pct, a)
    pce = [baseline_pce * angle_factor(float(x), a) * f_shade for x in angles]
    return angles, np.array(pce)


def shading_curve(baseline_pce, angle_deg, assumptions=None, model=None, step=1):
    """Effective PCE for every shading level 0..100 % at a fixed angle."""
    a = assumptions or load_assumptions()
    levels = np.arange(0, 100 + step, step)
    f_angle = angle_factor(angle_deg, a)
    pce = [baseline_pce * f_angle * shading_factor(float(x), a, model=model) for x in levels]
    return levels, np.array(pce)
