"""
SolarSmart AI
Presentation helpers: dark theme styling, header and summary cards.
No model or physics logic lives here (the 3D scenes are in scene.py).
"""

CSS = """
<style>
:root { --bg:#070b18; --surface:#0f1630; --surface2:#141d3d; --line:rgba(120,160,255,.20); --text:#e8eefc;
        --muted:#8fa3c9; --sun:#ffb800; --cyan:#38d9ff; --green:#34d399; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(255,184,0,.10), transparent 60%),
                     radial-gradient(900px 500px at -5% 10%, rgba(56,120,255,.14), transparent 60%), var(--bg); color:var(--text); }
.block-container { padding-top:1.4rem; max-width:1320px; }
h1,h2,h3,h4 { color:var(--text); letter-spacing:-0.01em; }
h2 { border-left:4px solid var(--sun); padding-left:.6rem; }
[data-testid="stHeader"] { background:transparent; }

/* ---------- header ---------- */
.ss-header { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:18px; padding:1.5rem 1.7rem;
  background:linear-gradient(135deg, rgba(20,29,61,.95), rgba(12,18,40,.95)); margin-bottom:1.1rem;
  box-shadow:0 0 40px rgba(56,120,255,.12); }
.ss-sunorb { position:absolute; right:-40px; top:-50px; width:220px; height:220px; border-radius:50%;
  background:radial-gradient(circle at 40% 40%, #ffe08a 0%, #ffb800 35%, rgba(255,184,0,.0) 70%);
  animation:pulse 5s ease-in-out infinite; opacity:.85; }
@keyframes pulse { 0%,100%{ transform:scale(1); opacity:.75;} 50%{ transform:scale(1.08); opacity:.95;} }
.ss-header h1 { font-size:2.1rem; margin:0; padding:0; color:#fff; position:relative; }
.ss-sub { font-size:1.05rem; color:var(--sun); font-weight:600; margin:.15rem 0 .55rem; position:relative; }
.ss-desc { font-size:.94rem; color:var(--muted); margin:0; max-width:820px; position:relative; }
.ss-desc b { color:var(--text); }
.ss-status { display:inline-block; margin-top:.8rem; padding:.22rem .75rem; border-radius:999px; font-size:.8rem; font-weight:600;
  border:1px solid var(--line); position:relative; }
.ss-status.ok { color:#6ee7b7; border-color:rgba(52,211,153,.5); background:rgba(52,211,153,.10); }
.ss-status.bad { color:#fca5a5; border-color:rgba(248,113,113,.5); background:rgba(248,113,113,.10); }

/* ---------- cards ---------- */
.ss-card { border:1px solid var(--line); border-radius:14px; padding:.95rem 1.15rem; height:100%; margin-bottom:.6rem;
  background:linear-gradient(160deg, rgba(20,29,61,.9), rgba(13,20,44,.9)); transition:transform .18s ease, box-shadow .18s ease; }
.ss-card:hover { transform:translateY(-3px); box-shadow:0 8px 26px rgba(0,0,0,.45); }
.ss-card .lab { font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); }
.ss-card .val { font-size:1.9rem; font-weight:800; color:#fff; line-height:1.25; }
.ss-card .val.empty { color:#4c5a80; }
.ss-card .sub { font-size:.8rem; color:var(--muted); }
.ss-card.good { border-top:3px solid var(--green); box-shadow:0 -6px 22px -12px rgba(52,211,153,.6); }
.ss-card.sun  { border-top:3px solid var(--sun);   box-shadow:0 -6px 22px -12px rgba(255,184,0,.6); }
.ss-card.cyan { border-top:3px solid var(--cyan);  box-shadow:0 -6px 22px -12px rgba(56,217,255,.6); }
.ss-card .tip { cursor:help; border-bottom:1px dotted var(--muted); }

/* ---------- panels and tags ---------- */
.ss-panel-title { font-size:1.2rem; font-weight:700; margin:.2rem 0 .1rem; color:#fff; }
.ss-hint { font-size:.85rem; color:var(--muted); margin-bottom:.4rem; }
.ss-box { border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; background:rgba(15,22,48,.8); font-size:.9rem; }
.ss-box.assump { border-color:rgba(255,184,0,.35); background:rgba(255,184,0,.06); }
.ss-tag { display:inline-block; font-size:.7rem; font-weight:800; padding:.12rem .55rem; border-radius:6px; margin-right:.5rem; letter-spacing:.04em; }
.ss-tag.ml { background:rgba(52,211,153,.15); color:#6ee7b7; border:1px solid rgba(52,211,153,.4); }
.ss-tag.phys { background:rgba(255,184,0,.15); color:#ffd166; border:1px solid rgba(255,184,0,.4); }

/* ---------- widgets ---------- */
button[kind="primary"], div[data-testid="stBaseButton-primary"] button {
  background:linear-gradient(135deg,#ffc933,#ffa200); color:#1a1200; border:0; font-weight:800;
  box-shadow:0 0 18px rgba(255,184,0,.35); transition:transform .15s ease, box-shadow .15s ease; }
button[kind="primary"]:hover, div[data-testid="stBaseButton-primary"] button:hover {
  transform:translateY(-2px) scale(1.01); box-shadow:0 0 28px rgba(255,184,0,.6); color:#1a1200; }
button[kind="secondary"], div[data-testid="stBaseButton-secondary"] button { border:1px solid var(--line); transition:border-color .15s, box-shadow .15s; }
button[kind="secondary"]:hover, div[data-testid="stBaseButton-secondary"] button:hover { border-color:var(--sun); box-shadow:0 0 14px rgba(255,184,0,.25); }
div[data-testid="stMetric"] { border:1px solid var(--line); border-radius:12px; padding:.55rem .85rem; background:rgba(15,22,48,.7); }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:rgba(15,22,48,.5); }
iframe { border:0; }
@media (max-width:640px){ .ss-header h1{font-size:1.5rem;} .ss-card .val{font-size:1.45rem;} .ss-sunorb{opacity:.4;} }
</style>
"""


def header_html(model_ok, model_name, load_error=None):
    if model_ok:
        status = f'<span class="ss-status ok" role="status">● ML model loaded: {model_name}</span>'
    else:
        status = '<span class="ss-status bad" role="status">● ML model not available - run python src/train.py</span>'
    return f"""
    <div class="ss-header">
      <div class="ss-sunorb"></div>
      <h1>☀️ SolarSmart AI</h1>
      <div class="ss-sub">Solar Efficiency Prediction and Optimization</div>
      <p class="ss-desc">Explore how three factors shape solar performance: the <b>material composition</b> of the cell
      (predicted by a machine-learning model trained on real devices), the <b>incidence angle</b> of sunlight, and
      <b>panel shading</b> (both estimated with clearly labelled physics approximations). Drag the 3D sun and cloud to play.</p>
      {status}
    </div>"""


def card_html(label, value, sub="", tone="cyan", tip=""):
    """Summary card. `value=None` renders a placeholder instead of a fabricated number."""
    tip_attr = f' title="{tip}"' if tip else ""
    lab = f'<span class="tip"{tip_attr}>{label}</span>' if tip else label
    if value is None:
        val = '<div class="val empty">-</div>'
    else:
        val = f'<div class="val">{value}</div>'
    return f'<div class="ss-card {tone}"><div class="lab">{lab}</div>{val}<div class="sub">{sub}</div></div>'
