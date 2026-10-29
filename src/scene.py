"""
SolarSmart AI
Interactive 3D scenes (Three.js) wrapped as Streamlit components.

The frontend lives in components/solar_scene/. It talks to Streamlit with the
plain postMessage component protocol, so no npm build step is needed. Dragging
in the scene returns a value ({angle|shade, nonce}); the app then moves the
matching slider, so the 3D scene, sliders and results always agree.
"""

import os

import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "components", "solar_scene")
_scene = components.declare_component("solar_scene", path=_DIR)


def angle_scene(angle_deg, factor, key="scene_angle"):
    """Draggable sun over a 3D panel. Returns {'angle', 'nonce'} after the user drags, else None."""
    return _scene(mode="angle", angle=float(angle_deg), factor=float(factor), key=key, default=None)


def shade_scene(shading_pct, factor, bypass_states, key="scene_shade"):
    """Draggable cloud over a 3D cell-by-cell panel. Returns {'shade', 'nonce'} after a drag, else None."""
    return _scene(mode="shade", shade=float(shading_pct), factor=float(factor),
                  states=bypass_states or [], key=key, default=None)
