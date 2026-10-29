"""
SolarSmart AI
Interactive Plotly charts. Every function receives real numbers (from the
fitted model or from physics.py) and returns a figure; nothing is hard-coded.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

CYAN = "#38d9ff"
TEXT = "#dbe6ff"
SUN = "#ffb800"
GREEN = "#34d399"
SOFT_GREEN = "#86efac"
GREY = "#7d8db3"
LINE_GREY = "rgba(120,160,255,0.16)"

_FEATURE_LABELS = {
    "quenching": "Anti-solvent quenching",
    "is_lead_free": "Lead-free composition",
    "HTL_stack_sequence": "Hole transport layer",
    "ETL_stack_sequence": "Electron transport layer",
    "ETL_deposition_procedure": "ETL deposition method",
    "Perovskite_composition_a_ions": "A-site cations",
    "Perovskite_composition_b_ions": "B-site metal",
    "Perovskite_composition_c_ions": "Halide anions",
    "Perovskite_composition_short_form": "Perovskite composition",
    "Perovskite_deposition_solvents": "Deposition solvents",
    "Perovskite_deposition_procedure": "Perovskite deposition method",
    "annealing_temp": "Annealing temperature",
    "annealing_time": "Annealing time",
    "perovskite_thickness": "Perovskite thickness",
    "backcontact_thickness": "Back-contact thickness",
    "Backcontact_stack_sequence": "Back-contact material",
    "Backcontact_deposition_procedure": "Back-contact method",
    "HTL_deposition_procedure": "HTL deposition method",
    "Cell_architecture": "Cell architecture",
    "Substrate_stack_sequence": "Substrate",
    "has_additive": "Additive present",
    "is_mixed_cation": "Mixed cations",
    "is_mixed_halide": "Mixed halides",
    "encapsulation": "Encapsulation",
    "deposition_steps": "Deposition steps",
}


def _base_layout(fig, height=340, ytitle=None, xtitle=None):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#141d3d", font_size=13, font_color="#e8eefc"),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE_GREY, title=xtitle)
    fig.update_yaxes(gridcolor=LINE_GREY, zeroline=False, title=ytitle)
    return fig


def fig_baseline_vs_adjusted(sc):
    """Baseline PCE next to each adjusted estimate."""
    labels = ["ML baseline", "Angle-adjusted", "Shading-adjusted", "Combined"]
    values = [sc["baseline_pce"], sc["angle_adjusted_pce"], sc["shading_adjusted_pce"], sc["effective_pce"]]
    colors = [CYAN, SUN, SUN, GREEN]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.2f}%" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    _base_layout(fig, ytitle="PCE (%)")
    fig.update_yaxes(range=[0, max(values) * 1.2 if max(values) > 0 else 1])
    return fig


def fig_angle_curve(angles, pce, sc):
    """Estimated effective PCE vs incidence angle (idealised, physics-based)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=pce, mode="lines", name="Estimated (cosine model)",
        line=dict(color=SUN, width=3, dash="dash"),
        hovertemplate="%{x}°: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[sc["angle_deg"]], y=[sc["effective_pce"]], mode="markers", name="Selected",
        marker=dict(color="#ffffff", size=13, line=dict(color=SUN, width=3)),
        hovertemplate="Selected %{x}°: %{y:.2f}%<extra></extra>",
    ))
    _base_layout(fig, ytitle="Effective PCE (%)", xtitle="Incidence angle (°)")
    fig.update_xaxes(range=[0, 90], dtick=15)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_shading_curve(levels, curves, selected_model, labels, sc):
    """Estimated effective PCE vs shading, comparing the available approximations."""
    fig = go.Figure()
    for model, pce in curves.items():
        is_sel = model == selected_model
        fig.add_trace(go.Scatter(
            x=levels, y=pce, mode="lines", name=labels.get(model, model),
            line=dict(color=SUN if is_sel else GREY, width=3 if is_sel else 1.5, dash="dash" if is_sel else "dot"),
            hovertemplate="%{x}% shaded: %{y:.2f}%<extra>" + labels.get(model, model) + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[sc["shading_pct"]], y=[sc["effective_pce"]], mode="markers", name="Selected",
        marker=dict(color="#ffffff", size=13, line=dict(color=SUN, width=3)),
        hovertemplate="Selected %{x}%: %{y:.2f}%<extra></extra>",
    ))
    _base_layout(fig, ytitle="Effective PCE (%)", xtitle="Shaded area (%)")
    fig.update_xaxes(range=[0, 100], dtick=20)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_scenarios(scenarios):
    """Side-by-side effective PCE and estimated power for the saved scenarios."""
    names = [s["name"] for s in scenarios]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Effective PCE (%)", "Estimated power (W)"),
                        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(
        x=names, y=[s["effective_pce"] for s in scenarios], marker_color=GREEN, name="Effective PCE",
        text=[f"{s['effective_pce']:.2f}" for s in scenarios], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=names, y=[s["power_w"] for s in scenarios], marker_color=SUN, name="Power (W)",
        text=[f"{s['power_w']:.0f}" for s in scenarios], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:.1f} W<extra></extra>",
    ), row=1, col=2)
    _base_layout(fig, height=360)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(rangemode="tozero")
    fig.update_annotations(font=dict(size=13, color=TEXT))
    return fig


def fig_feature_importance(importances):
    """Horizontal bars of the fitted model's (grouped) feature importance."""
    ordered = list(reversed(importances))
    fig = go.Figure(go.Bar(
        x=[d["importance"] * 100 for d in ordered],
        y=[_FEATURE_LABELS.get(d["feature"], d["feature"]) for d in ordered],
        orientation="h", marker_color=GREEN,
        hovertemplate="%{y}: %{x:.1f}% of total importance<extra></extra>",
    ))
    _base_layout(fig, height=380, xtitle="Share of model importance (%)")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(showgrid=True, gridcolor=LINE_GREY)
    fig.update_yaxes(showgrid=False)
    return fig
