"""
SolarSmart AI: Solar Efficiency Prediction and Optimization
Streamlit web application.

Layout (top to bottom): header -> overview cards -> material configuration (ML)
and solar conditions (physics) -> results -> visual analytics -> scenario
comparison -> model details and methodology.

Responsibilities are split across modules in src/:
    predict.py   saved ML pipeline, input validation, feature importance
    physics.py   idealised angle / shading estimates and the combined scenario
    charts.py    Plotly figures        ui.py   CSS, cards, SVG panel drawings
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import charts  # noqa: E402
import physics  # noqa: E402
import scene  # noqa: E402
import ui  # noqa: E402
from predict import (  # noqa: E402
    NO_ADDITIVE, SunPredictor, UI_CATEGORICAL_KEYS, UI_NUMERIC_KEYS, default_selection,
    load_ui_options, predict_baseline_pce, validate_selection,
)

st.set_page_config(page_title="SolarSmart AI", page_icon="☀️", layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)

MAX_SCENARIOS = 6


# ----------------------------------------------------------------------------
# Loading (cached: the model is read from disk once, never retrained)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    try:
        predictor = SunPredictor(models_dir=MODELS_DIR)
        options = load_ui_options(MODELS_DIR)
        if options is None:
            raise FileNotFoundError("models/ui_options.json is missing. Run: python src/train.py")
        return predictor, options, None
    except Exception as exc:  # shown to the user in the header and the empty state
        return None, None, str(exc)


@st.cache_data
def load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


predictor, options, load_error = load_model()
metadata = load_json("model_metadata.json")
benchmarks = load_json("candidate_metrics.json")
dataset_summary = load_json("dataset_summary.json")
base_assumptions = physics.load_assumptions()
DEF = base_assumptions["defaults"]

model_ok = predictor is not None and metadata is not None
st.markdown(ui.header_html(model_ok, metadata["best_model_name"] if metadata else "", load_error), unsafe_allow_html=True)

if not model_ok:
    st.error("**The trained model could not be loaded.**")
    st.markdown(f"Details: `{load_error or 'model_metadata.json missing'}`")
    st.markdown("Train it once from the project folder, then refresh this page:")
    st.code("python src\\train.py", language="bash")
    st.stop()


# ----------------------------------------------------------------------------
# Session state and callbacks
# ----------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    for k, v in default_selection(options).items():
        ss.setdefault(f"m_{k}", v)
    ss.setdefault("preset", "Custom configuration")
    for name, value in {
        "angle": float(DEF["angle_deg"]), "shade": float(DEF["shading_pct"]),
        "irr": float(DEF["irradiance_w_m2"]), "area": float(DEF["panel_area_m2"]),
        "shade_model": base_assumptions["shading"]["model"],
        "bd_n": int(base_assumptions["shading"]["bypass_diode"]["n_substrings"]),
        "bd_thr": float(base_assumptions["shading"]["bypass_diode"]["activation_shaded_fraction"]),
        "pl_exp": float(base_assumptions["shading"]["power_law"]["exponent"]),
        "diffuse": float(base_assumptions["angle"]["diffuse_fraction"]),
    }.items():
        ss.setdefault(name, value)
        # Widgets that are not drawn on a given run lose their state, so re-seed from the canonical value
        ss.setdefault(f"{name}_w1", ss[name])  # slider / first widget
        ss.setdefault(f"{name}_w2", ss[name])  # numeric box
    ss.setdefault("ml", None)
    ss.setdefault("scenarios", [])
    ss.setdefault("scenario_counter", 0)


init_state()


def sync(name, source):
    """Keep a slider and its numeric box showing the same value."""
    ss = st.session_state
    value = ss[f"{name}_{source}"]
    ss[name] = value
    ss[f"{name}_w1"] = value
    ss[f"{name}_w2"] = value


def reset_conditions():
    ss = st.session_state
    for name, value in {"angle": DEF["angle_deg"], "shade": DEF["shading_pct"],
                        "irr": DEF["irradiance_w_m2"], "area": DEF["panel_area_m2"]}.items():
        for suffix in ("", "_w1", "_w2"):
            ss[f"{name}{suffix}"] = float(value)
    for name, value in {"shade_model": base_assumptions["shading"]["model"],
                        "bd_n": int(base_assumptions["shading"]["bypass_diode"]["n_substrings"]),
                        "bd_thr": float(base_assumptions["shading"]["bypass_diode"]["activation_shaded_fraction"]),
                        "pl_exp": float(base_assumptions["shading"]["power_law"]["exponent"]),
                        "diffuse": float(base_assumptions["angle"]["diffuse_fraction"])}.items():
        for suffix in ("", "_w1", "_w2"):
            ss[f"{name}{suffix}"] = value


def reset_materials():
    for k, v in default_selection(options).items():
        st.session_state[f"m_{k}"] = v
    st.session_state["preset"] = "Custom configuration"
    st.session_state["ml"] = None


def reset_all():
    reset_conditions()
    reset_materials()


PRESETS = {
    "Multi-cation perovskite (n-i-p)": {
        "Cell_architecture": "nip", "Substrate_stack_sequence": "SLG | FTO",
        "ETL_stack_sequence": "TiO2-c | TiO2-mp", "Perovskite_composition_short_form": "CsFAMAPbBrI",
        "Perovskite_additives_compounds": "none", "Perovskite_deposition_solvents": "DMF; DMSO",
        "HTL_stack_sequence": "Spiro-MeOTAD", "Backcontact_stack_sequence": "Au",
        "perovskite_thickness": 450.0, "annealing_temp": 100.0, "annealing_time": 30.0, "quenching": True,
    },
    "Classic MAPbI (n-i-p)": {
        "Cell_architecture": "nip", "Substrate_stack_sequence": "SLG | FTO",
        "ETL_stack_sequence": "TiO2-c | TiO2-mp", "Perovskite_composition_short_form": "MAPbI",
        "Perovskite_additives_compounds": "none", "Perovskite_deposition_solvents": "DMF",
        "HTL_stack_sequence": "Spiro-MeOTAD", "Backcontact_stack_sequence": "Au",
        "perovskite_thickness": 350.0, "annealing_temp": 100.0, "annealing_time": 10.0, "quenching": False,
    },
    "Inverted p-i-n (PEDOT:PSS / PCBM)": {
        "Cell_architecture": "pin", "Substrate_stack_sequence": "SLG | ITO",
        "ETL_stack_sequence": "PCBM-60 | BCP", "Perovskite_composition_short_form": "MAPbI",
        "Perovskite_additives_compounds": "none", "Perovskite_deposition_solvents": "DMF; DMSO",
        "HTL_stack_sequence": "PEDOT:PSS", "Backcontact_stack_sequence": "Ag",
        "perovskite_thickness": 400.0, "annealing_temp": 100.0, "annealing_time": 15.0, "quenching": True,
    },
    "HTL-free, carbon back contact": {
        "Cell_architecture": "nip", "Substrate_stack_sequence": "SLG | FTO",
        "ETL_stack_sequence": "TiO2-c | TiO2-mp", "Perovskite_composition_short_form": "MAPbI",
        "Perovskite_additives_compounds": "5-AVAI", "Perovskite_deposition_solvents": "DMF",
        "HTL_stack_sequence": "none", "Backcontact_stack_sequence": "Carbon",
        "perovskite_thickness": 500.0, "annealing_temp": 100.0, "annealing_time": 60.0, "quenching": False,
    },
}


def apply_preset():
    """Fill the form from a preset; only values that exist in the dataset options are applied."""
    preset = PRESETS.get(st.session_state["preset"])
    if not preset:
        return
    for key, value in preset.items():
        if key in UI_CATEGORICAL_KEYS:
            allowed = {o["value"] for o in options["categorical"][key]} | {NO_ADDITIVE}
            if value not in allowed:
                continue
        st.session_state[f"m_{key}"] = value


def current_selection():
    ss = st.session_state
    return {k: ss[f"m_{k}"] for k in UI_CATEGORICAL_KEYS + UI_NUMERIC_KEYS + ["quenching", "encapsulation"]}


def current_assumptions():
    """Config defaults plus whatever the user changed in the assumptions panel."""
    ss = st.session_state
    a = physics.load_assumptions()
    a["shading"]["model"] = ss["shade_model"]
    a["shading"]["bypass_diode"]["n_substrings"] = int(ss["bd_n"])
    a["shading"]["bypass_diode"]["activation_shaded_fraction"] = float(ss["bd_thr"])
    a["shading"]["power_law"]["exponent"] = float(ss["pl_exp"])
    a["angle"]["diffuse_fraction"] = float(ss["diffuse"])
    return a


def run_prediction():
    ss = st.session_state
    sel = current_selection()
    try:
        result = predict_baseline_pce(predictor, sel, options)
        ss["ml"] = {"sel": sel, "result": result, "error": None}
        st.toast(f"Baseline PCE predicted: {result['predicted_pce']:.2f}%", icon="⚡")
    except Exception as exc:
        ss["ml"] = {"sel": sel, "result": None, "error": str(exc)}


def save_scenario(name, sc):
    ss = st.session_state
    if len(ss["scenarios"]) >= MAX_SCENARIOS:
        return False
    ss["scenario_counter"] += 1
    ss["scenarios"].append({"id": ss["scenario_counter"], "name": name, **sc})
    return True


def add_examples():
    """Three illustrative conditions evaluated with the CURRENT ML baseline and physics."""
    ss = st.session_state
    base = ss["ml"]["result"]["predicted_pce"]
    a = current_assumptions()
    for name, ang, shd in [("Perpendicular, no shade", 0, 0), ("Angled, moderate shade", 30, 50),
                           ("Oblique, heavy shade", 60, 75)]:
        sc = physics.compute_scenario(base, ang, shd, ss["irr"], ss["area"], a)
        if not save_scenario(name, sc):
            break


def remove_scenario(sid):
    st.session_state["scenarios"] = [s for s in st.session_state["scenarios"] if s["id"] != sid]


ml = st.session_state["ml"]
ml_result = ml["result"] if ml and ml["result"] else None
assumptions = current_assumptions()
ss = st.session_state

# The overview cards sit at the top of the page but are filled in at the end of the
# conditions section, after any drag in the 3D scenes has updated the controls.
cards_box = st.container()

# ----------------------------------------------------------------------------
# C. Material configuration (machine learning)
# ----------------------------------------------------------------------------
left = st.container()

with left:
    st.markdown('<span class="ss-tag ml">MACHINE LEARNING</span><span class="ss-panel-title">Material configuration</span>',
                unsafe_allow_html=True)
    st.markdown('<div class="ss-hint">Dropdown choices come from the training dataset (number of cells shown in brackets).</div>',
                unsafe_allow_html=True)
    st.selectbox("Load an example recipe", ["Custom configuration"] + list(PRESETS), key="preset", on_change=apply_preset)

    cat = options["categorical"]

    def choice(label, key, help_text=None):
        values = [o["value"] for o in cat[key]]
        counts = {o["value"]: o["count"] for o in cat[key]}
        if key == "Perovskite_additives_compounds":
            values = [NO_ADDITIVE] + values
        st.selectbox(label, values, key=f"m_{key}", help=help_text,
                     format_func=lambda v: f"{v} ({counts[v]:,})" if v in counts else v)

    def number(label, key, step, help_text):
        st.number_input(label, min_value=0.0, max_value=5000.0, step=step, key=f"m_{key}", help=help_text)

    a1, a2, a3 = st.columns(3)
    with a1:
        choice("Cell architecture", "Cell_architecture", "n-i-p: electrons collected at the substrate side; p-i-n is the inverted stack.")
        choice("Substrate", "Substrate_stack_sequence")
        choice("Electron transport layer (ETL)", "ETL_stack_sequence", "Layer that extracts electrons from the perovskite.")
        choice("ETL deposition", "ETL_deposition_procedure")
    with a2:
        choice("Perovskite composition", "Perovskite_composition_short_form")
        choice("Additive", "Perovskite_additives_compounds", "Only 'additive present or not' is used by the model.")
        choice("Deposition solvents", "Perovskite_deposition_solvents")
    with a3:
        choice("Hole transport layer (HTL)", "HTL_stack_sequence", "Layer that extracts holes. 'none' means HTL-free.")
        choice("HTL deposition", "HTL_deposition_procedure")
        choice("Back contact", "Backcontact_stack_sequence")

    comp = options["composition_map"].get(ss["m_Perovskite_composition_short_form"])
    if comp:
        st.caption(f"Composition fills A-site: {comp['a']} · B-site: {comp['b']} · halide: {comp['c']} "
                   "(most common in the dataset for this formula)")

    num = options["numeric"]
    n1, n2, n3, n4 = st.columns([1, 1, 1, 1])
    with n1:
        number("Perovskite thickness (nm)", "perovskite_thickness", 25.0,
               f"Typical range {num['perovskite_thickness']['p01']:g}-{num['perovskite_thickness']['p99']:g} nm. "
               f"Note: {num['perovskite_thickness']['missing_fraction'] * 100:.0f}% of training rows lack this value.")
    with n2:
        number("Annealing temperature (°C)", "annealing_temp", 5.0,
               f"Typical range {num['annealing_temp']['p01']:g}-{num['annealing_temp']['p99']:g} °C.")
    with n3:
        number("Annealing time (min)", "annealing_time", 5.0,
               f"Typical range {num['annealing_time']['p01']:g}-{num['annealing_time']['p99']:g} min.")
    with n4:
        st.checkbox("Anti-solvent quenching", key="m_quenching", help="Rapid anti-solvent drip that triggers crystallisation.")
        st.checkbox("Encapsulated", key="m_encapsulation", help="Sealed against moisture and oxygen.")
    with st.expander("More fabrication details"):
        choice("Perovskite deposition method", "Perovskite_deposition_procedure")
        choice("Back-contact deposition", "Backcontact_deposition_procedure")
        number("Back-contact thickness (nm)", "backcontact_thickness", 10.0,
               f"Typical range {num['backcontact_thickness']['p01']:g}-{num['backcontact_thickness']['p99']:g} nm.")

    errors, live_warnings = validate_selection(current_selection(), options)
    for msg in errors:
        st.error(msg)
    # Once a prediction for exactly these inputs exists, its warnings appear in the results panel instead
    if not (ml_result and ml["sel"] == current_selection()):
        for msg in live_warnings:
            st.warning(msg)

    b1, b2 = st.columns([2, 1])
    b1.button("Predict Composition Efficiency", type="primary", width="stretch",
              on_click=run_prediction, disabled=bool(errors))
    b2.button("Reset all", width="stretch", on_click=reset_all,
              help="Restore every material input and solar condition to its default and clear the prediction.")

st.write("")
st.markdown('<span class="ss-tag phys">PHYSICS ESTIMATE</span><span class="ss-panel-title">Interactive solar conditions</span>',
            unsafe_allow_html=True)
st.markdown('<div class="ss-hint">Drag the 3D sun and cloud (or use the sliders). Results update instantly; the ML model is not re-run.</div>',
            unsafe_allow_html=True)


def apply_scene(name, value, field):
    """A drag in a 3D scene moves the matching slider and number box (runs before they are drawn)."""
    if not value or value["nonce"] == ss.get(f"{name}_scene_nonce"):
        return
    ss[f"{name}_scene_nonce"] = value["nonce"]
    for suffix in ("", "_w1", "_w2"):
        ss[f"{name}{suffix}"] = float(value[field])


col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown("**Incidence angle**")
    apply_scene("angle", scene.angle_scene(ss["angle"], physics.angle_factor(ss["angle"], assumptions)), "angle")
    sa, sb = st.columns([3, 1])
    sa.slider("Incidence angle (°)", 0.0, 90.0, step=0.5, key="angle_w1", on_change=sync, args=("angle", "w1"),
              help="Angle between sunlight and the panel's perpendicular (normal). 0° = straight on, 90° = grazing.")
    sb.number_input("Angle (°)", 0.0, 90.0, step=0.5, key="angle_w2", on_change=sync, args=("angle", "w2"),
                    label_visibility="collapsed")
    st.caption("0° means sunlight hits the panel perpendicularly; 90° is grazing. In idealised direct-beam geometry the "
               "contribution is ≈ cos(angle). Real output also depends on diffuse light, reflection, the atmosphere and panel characteristics.")

with col_b:
    st.markdown("**Panel shading**")
    diode_states = (physics.bypass_substring_states(ss["shade"], assumptions)
                    if assumptions["shading"]["model"] == "bypass_diode" else [])
    apply_scene("shade", scene.shade_scene(ss["shade"], physics.shading_factor(ss["shade"], assumptions), diode_states), "shade")
    sc_a, sc_b = st.columns([3, 1])
    sc_a.slider("Shaded panel area (%)", 0.0, 100.0, step=1.0, key="shade_w1", on_change=sync, args=("shade", "w1"),
                help="Share of the panel area that is in shadow.")
    sc_b.number_input("Shading (%)", 0.0, 100.0, step=1.0, key="shade_w2", on_change=sync, args=("shade", "w2"),
                      label_visibility="collapsed")
    st.caption("Shading cuts power because shaded cells limit the current of the cells wired in series with them. "
               "50% shading does not necessarily mean a 50% loss: wiring, bypass diodes, cell layout and shadow position matter.")

e1, e2, e3 = st.columns([1, 1, 1])
e1.number_input("Beam irradiance (W/m²)", 0.0, 1500.0, step=50.0, key="irr_w1", on_change=sync, args=("irr", "w1"),
                help="Sunlight power per m² measured perpendicular to the rays (1000 W/m² is the standard test value).")
e2.number_input("Panel area (m²)", 0.01, 100.0, step=0.1, key="area_w1", on_change=sync, args=("area", "w1"))
e3.write("")
e3.button("Reset conditions", on_click=reset_conditions, width="stretch",
          help="Restore angle, shading, irradiance, area and assumptions to the defaults.")

with st.expander("Physics assumptions (change the shading approximation)"):
    st.selectbox("Shading model", assumptions["shading"]["available_models"], key="shade_model_w1",
                 format_func=physics.SHADING_MODEL_LABELS.get, on_change=sync, args=("shade_model", "w1"))
    if ss["shade_model"] == "bypass_diode":
        st.number_input("Bypass-diode substrings", 1, 12, step=1, key="bd_n_w1", on_change=sync, args=("bd_n", "w1"))
        st.slider("Substring is bypassed when this fraction is shaded", 0.05, 1.0, step=0.05, key="bd_thr_w1",
                  on_change=sync, args=("bd_thr", "w1"))
    elif ss["shade_model"] == "power_law":
        st.slider("Loss exponent", 0.5, 3.0, step=0.1, key="pl_exp_w1", on_change=sync, args=("pl_exp", "w1"))
    st.slider("Diffuse-light fraction (0 = pure cosine law)", 0.0, 0.5, step=0.05, key="diffuse_w1",
              on_change=sync, args=("diffuse", "w1"),
              help="Share of light assumed to arrive from all directions, independent of the angle.")

# One validated physics calculation per rerun (cheap: no model inference here)
scenario = None
physics_error = None
if ml_result:
    try:
        scenario = physics.compute_scenario(ml_result["predicted_pce"], ss["angle"], ss["shade"],
                                            ss["irr"], ss["area"], assumptions)
    except ValueError as exc:
        physics_error = str(exc)

# ----------------------------------------------------------------------------
# B. Overview cards (drawn into the container reserved at the top of the page)
# ----------------------------------------------------------------------------
with cards_box:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(ui.card_html(
        "Baseline predicted PCE", f"{ml_result['predicted_pce']:.2f}%" if ml_result else None,
        "ML model" if ml_result else "Run a prediction to see this", "cyan",
        "Power Conversion Efficiency: the share of sunlight energy a cell turns into electricity."), unsafe_allow_html=True)
    c2.markdown(ui.card_html(
        "Combined estimated PCE", f"{scenario['effective_pce']:.2f}%" if scenario else None,
        (f"{scenario['rel_change_pct'] + 0.0:+.1f}% vs baseline" if scenario else "Needs a baseline prediction"), "good",
        "Baseline PCE x angle factor x shading factor (physics approximation)."), unsafe_allow_html=True)
    c3.markdown(ui.card_html(
        "Solar incidence angle", f"{ss['angle']:.1f}°", f"cos factor {physics.angle_factor(ss['angle'], assumptions):.3f}", "sun",
        "Angle between incoming sunlight and the panel's perpendicular (normal)."), unsafe_allow_html=True)
    c4.markdown(ui.card_html(
        "Panel shading", f"{ss['shade']:.0f}%",
        f"factor {physics.shading_factor(ss['shade'], assumptions):.3f} ({physics.SHADING_MODEL_LABELS[assumptions['shading']['model']].split(' (')[0]})",
        "sun", "Share of the panel area in shadow."), unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------------------------
# E. Prediction results
# ----------------------------------------------------------------------------
st.markdown("## Prediction results")
if ml and ml["error"]:
    st.error(f"Prediction failed: {ml['error']}")
elif not ml_result:
    st.info("No prediction yet. Choose a material configuration and click **Predict Composition Efficiency**. "
            "Until then no efficiency values are shown.")
else:
    if ml["sel"] != current_selection():
        st.warning("The material inputs changed after this prediction. Click **Predict Composition Efficiency** to refresh it.")
    for msg in ml_result["warnings"]:
        st.warning(msg)

    r1, r2 = st.columns([1, 1.25], gap="large")
    with r1:
        st.markdown('<span class="ss-tag ml">MACHINE LEARNING</span>**Baseline PCE from material composition**',
                    unsafe_allow_html=True)
        st.metric("Predicted baseline PCE", f"{ml_result['predicted_pce']:.2f}%",
                  help="Estimated efficiency under standard test conditions, from the fitted model.")
        st.caption(f"Model: **{metadata['best_model_name']}** · {ml_result['coverage_level_pct']}% prediction interval: "
                   f"{ml_result['lower_bound']:.1f}% to {ml_result['upper_bound']:.1f}%")
        st.write(
            "This is the efficiency that devices with similar recipes reached in the published records the model learned "
            "from. It shows a statistical association, **not** proof that a composition causes higher efficiency.")
        m = metadata["test_metrics"]
        k1, k2, k3 = st.columns(3)
        k1.metric("MAE", f"{m['mae']:.2f} pp", help="Mean Absolute Error: the average size of the prediction error, in percentage points of PCE, on held-out test cells.")
        k2.metric("RMSE", f"{m['rmse']:.2f} pp", help="Root Mean Squared Error: like MAE but penalises large misses more.")
        k3.metric("R²", f"{m['r2']:.3f}", help="Share of the variation in PCE the model explains (1 = perfect, 0 = no better than predicting the average).")
        st.caption(f"Measured on {metadata['test_samples']:,} held-out cells never used for training or model selection. "
                   "The model explains roughly half of the variation, so treat individual predictions as rough guides.")

    with r2:
        if physics_error:
            st.error(f"Invalid solar conditions: {physics_error}")
        elif scenario:
            st.markdown('<span class="ss-tag phys">PHYSICS ESTIMATE</span>**Adjusted for sunlight angle and shading**',
                        unsafe_allow_html=True)
            rows = [
                ("Baseline PCE (ML)", f"{scenario['baseline_pce']:.2f}%", "-"),
                ("Angle-adjusted PCE", f"{scenario['angle_adjusted_pce']:.2f}%", f"× {scenario['angle_factor']:.3f}"),
                ("Shading-adjusted PCE", f"{scenario['shading_adjusted_pce']:.2f}%", f"× {scenario['shading_factor']:.3f}"),
                ("Combined effective PCE", f"{scenario['effective_pce']:.2f}%",
                 f"× {scenario['angle_factor'] * scenario['shading_factor']:.3f}"),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Quantity", "Value", "Factor vs. baseline"]),
                         hide_index=True, width="stretch")
            p1, p2, p3 = st.columns(3)
            p1.metric("Change vs. baseline", f"{scenario['abs_change_pp']:+.2f} pp", f"{scenario['rel_change_pct']:+.1f}%",
                      delta_color="normal")
            p2.metric("Estimated power", f"{scenario['power_w']:.1f} W",
                      help="Beam irradiance × panel area × combined effective PCE.")
            p3.metric("Power at baseline", f"{scenario['reference_power_w']:.1f} W",
                      f"{-scenario['power_loss_w'] + 0.0:+.1f} W", delta_color="normal",
                      help="Same irradiance and area with 0° angle and 0% shading.")
            if abs(scenario["abs_change_pp"]) < 1e-9:
                st.success("At 0° and 0% shading the adjusted result equals the ML baseline, as the ideal model requires.")
            st.markdown(
                f'<div class="ss-box assump"><b>Assumptions used</b><br>'
                f'• Angle factor = {"(1 - d)·cos(θ) + d with d = " + format(assumptions["angle"]["diffuse_fraction"], "g") if assumptions["angle"]["diffuse_fraction"] else "cos(θ)"} (ideal direct beam).<br>'
                f'• Shading factor: <b>{physics.SHADING_MODEL_LABELS[assumptions["shading"]["model"]]}</b> - a configurable approximation, not a validated prediction.<br>'
                f'• Combined PCE = baseline × angle factor × shading factor (the two effects are assumed independent).<br>'
                f'• Power = {scenario["irradiance_w_m2"]:g} W/m² × {scenario["panel_area_m2"]:g} m² × PCE; the irradiance is measured '
                f'perpendicular to the rays, so the cosine law is applied exactly once.<br>'
                f'• Temperature, spectrum, reflection and degradation are not modelled.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# F. Visual analytics
# ----------------------------------------------------------------------------
st.markdown("## Visual analytics")
CHART_CFG = {"displayModeBar": False}
if not scenario:
    st.info("Charts appear after a successful baseline prediction.")
else:
    st.caption("Curves for angle and shading are physics-based **estimates**, not measured data. "
               "The dot marks your current selection.")
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("**Baseline vs. adjusted efficiency**")
        st.plotly_chart(charts.fig_baseline_vs_adjusted(scenario), width="stretch", config=CHART_CFG)
    with g2:
        st.markdown("**Feature importance of the fitted model**")
        importances = predictor.get_feature_importance(12)
        if importances:
            st.plotly_chart(charts.fig_feature_importance(importances), width="stretch", config=CHART_CFG)
            st.caption("Importance shows what the model relied on, not what causes efficiency.")
        else:
            st.info("The selected algorithm does not provide feature importances.")
    g3, g4 = st.columns(2, gap="large")
    with g3:
        st.markdown("**Estimated performance across incidence angles** (at current shading)")
        angles, pce_a = physics.angle_curve(scenario["baseline_pce"], ss["shade"], assumptions)
        st.plotly_chart(charts.fig_angle_curve(angles, pce_a, scenario), width="stretch", config=CHART_CFG)
    with g4:
        st.markdown("**Estimated performance across shading levels** (at current angle)")
        curves = {}
        for model in assumptions["shading"]["available_models"]:
            levels, curves[model] = physics.shading_curve(scenario["baseline_pce"], ss["angle"], assumptions, model=model)
        st.plotly_chart(charts.fig_shading_curve(levels, curves, assumptions["shading"]["model"],
                                                 physics.SHADING_MODEL_LABELS, scenario),
                        width="stretch", config=CHART_CFG)
        st.caption("The three approximations disagree on purpose: the real loss depends on wiring and shadow position.")

# ----------------------------------------------------------------------------
# G. Scenario comparison
# ----------------------------------------------------------------------------
st.markdown("## Scenario comparison")
if not scenario:
    st.info("Make a baseline prediction first, then save scenarios to compare them side by side.")
else:
    s1, s2, s3, s4 = st.columns([2, 1.2, 1.4, 1])
    default_name = f"{ss['angle']:g}° / {ss['shade']:g}% shade"
    scen_name = s1.text_input("Scenario name", value=default_name, max_chars=40, key=f"scen_name_{default_name}")
    s2.write("")
    s2.button("Save current scenario", on_click=lambda: save_scenario(scen_name.strip() or default_name, scenario),
              disabled=len(ss["scenarios"]) >= MAX_SCENARIOS, width="stretch")
    s3.write("")
    s3.button("Add 3 example scenarios", on_click=add_examples, width="stretch",
              disabled=len(ss["scenarios"]) > MAX_SCENARIOS - 3,
              help="Perpendicular/no shade, 30°/50% shade and 60°/75% shade, evaluated with your current baseline.")
    s4.write("")
    s4.button("Clear all", on_click=lambda: ss.update(scenarios=[]), width="stretch", disabled=not ss["scenarios"])
    if len(ss["scenarios"]) >= MAX_SCENARIOS:
        st.caption(f"Maximum of {MAX_SCENARIOS} scenarios reached. Remove one to add another.")

    if not ss["scenarios"]:
        st.info("No saved scenarios yet. Save the current conditions or add the three examples.")
    else:
        st.plotly_chart(charts.fig_scenarios(ss["scenarios"]), width="stretch", config=CHART_CFG)
        table = pd.DataFrame([{
            "#": i, "Scenario": s["name"], "Baseline PCE (%)": round(s["baseline_pce"], 2), "Angle (°)": s["angle_deg"],
            "Shading (%)": s["shading_pct"], "Shading model": physics.SHADING_MODEL_LABELS[s["shading_model"]].split(" (")[0],
            "Effective PCE (%)": round(s["effective_pce"], 2), "Change (%)": round(s["rel_change_pct"], 1),
            "Power (W)": round(s["power_w"], 1),
        } for i, s in enumerate(ss["scenarios"], start=1)])
        st.dataframe(table, hide_index=True, width="stretch")
        cols = st.columns(MAX_SCENARIOS)
        for col, (i, s) in zip(cols, enumerate(ss["scenarios"], start=1)):
            col.button(f"Remove #{i}", key=f"rm_{s['id']}", on_click=remove_scenario, args=(s["id"],),
                       help=f"Remove “{s['name']}”", width="stretch")
        st.caption("Scenario values are calculated from the baseline prediction and the physics assumptions in force when each was saved.")

# ----------------------------------------------------------------------------
# Model details and methodology
# ----------------------------------------------------------------------------
st.divider()
st.markdown("## Model details and methodology")
t_model, t_diag, t_phys, t_guide = st.tabs(["Model & metrics", "Diagnostics", "Physics assumptions", "Guide & limitations"])

with t_model:
    n_total = dataset_summary["total_rows"] if dataset_summary else "?"
    n_valid = dataset_summary["pce_stats"]["valid_count"] if dataset_summary else "?"
    st.write(f"**Dataset:** {n_total:,} records, {dataset_summary['total_columns']} columns; {n_valid:,} have a valid PCE and are "
             f"the only rows used for supervised training. Split: {metadata['training_samples']:,} train (after removing exact duplicate "
             f"feature rows) / {metadata['calibration_samples']:,} validation / {metadata['test_samples']:,} test, fixed random seed 42.")
    st.write(f"**Final model:** {metadata['best_model_name']}, chosen by lowest validation MAE. "
             "Inputs are material and fabrication features only; the measured `JV_default_Voc`, `JV_default_Jsc`, `JV_default_FF` and "
             "`JV_measured` columns are excluded because they come from the same test as PCE (target leakage).")
    if benchmarks:
        bdf = pd.DataFrame(benchmarks)
        st.dataframe(bdf, hide_index=True, width="stretch")
        st.caption("Validation metrics for each candidate (lower MAE/RMSE and higher R² are better). "
                   "Held-out test metrics for the chosen model are shown in the results panel.")
    cov = metadata.get("conformal_test_coverage")
    if cov:
        st.dataframe(pd.DataFrame([{
            "Target coverage": f"{v['desired_coverage_pct']}%", "Empirical test coverage": f"{v['empirical_coverage_pct']}%",
            "Interval half-width": f"±{v['margin_q']} pp", "Average width": f"{v['average_interval_width']} pp",
        } for v in cov.values()]), hide_index=True, width="stretch")
        st.caption("Prediction intervals use split conformal prediction on the validation set.")
    img = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    if os.path.exists(img):
        st.image(img, caption="Candidate model comparison (validation set)")

with t_diag:
    d1, d2 = st.columns(2)
    for col, fname, cap in [(d1, "actual_vs_predicted.png", "Actual vs. predicted PCE (test set)"),
                            (d2, "residuals_distribution.png", "Residual diagnostics (test set)"),
                            (d1, "feature_importance.png", "Feature importance (per one-hot column)"),
                            (d2, "eda_target_distribution.png", "Distribution of PCE in the training data")]:
        path = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(path):
            col.image(path, caption=cap)

with t_phys:
    st.markdown(r"""
**Incidence angle** - angle θ between sunlight and the panel normal. Idealised direct-beam geometry:

`angle factor = cos(θ)` for 0° ≤ θ ≤ 90° (optionally `(1 − d)·cos(θ) + d` with a diffuse fraction *d*).

**Shading** - three selectable approximations of how much output remains for a shaded fraction *s*:
- *Linear*: `1 − s`.
- *Bypass-diode substrings*: the panel is split into *n* substrings; the shadow sweeps across them in order and a substring
  is bypassed once its shaded fraction reaches a threshold, otherwise it contributes its unshaded fraction.
- *Series-limited power law*: `(1 − s)^k` with an adjustable exponent *k*.

**Combination** - `effective PCE = baseline PCE × angle factor × shading factor`, and
`power (W) = beam irradiance (W/m²) × area (m²) × effective PCE / 100`.

Defaults live in `config/solar_assumptions.json`. **None of these effects were learned from the dataset**: it has no measured
angle or shading variables, so they are physics-based estimates and are labelled that way throughout the app.
""")

with t_guide:
    st.markdown("""
**Glossary**
- **PCE** - Power Conversion Efficiency, the percentage of sunlight energy converted to electricity.
- **Incidence angle** - angle between the sunlight and the panel's perpendicular.
- **Shading** - fraction of the panel area in shadow.
- **MAE / RMSE** - typical size of the model's error in PCE percentage points (RMSE punishes big misses more).
- **R²** - share of PCE variation the model explains.

**Limitations**
- The model reports associations in published lab devices, not cause and effect, and only covers perovskite cells.
- Records come from many papers and a random split may place similar devices in both train and test sets, so real-world
  error on genuinely new recipes can be higher than the reported test error.
- Rare or unseen categories are grouped as "infrequent"; inputs outside the training range are extrapolations (a warning appears).
- Lab-cell efficiency does not directly equal large-module efficiency.
- Angle and shading results are idealised estimates, not measurements.
""")
