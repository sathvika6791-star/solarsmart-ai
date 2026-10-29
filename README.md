# SolarSmart AI: Solar Efficiency Prediction and Optimization

A beginner-friendly machine-learning project that studies three factors affecting solar performance:

1. **Material composition** - a real regression model (trained on 41,519 perovskite solar cells) predicts the baseline Power Conversion Efficiency (PCE).
2. **Solar incidence angle** - an idealised cosine-law estimate.
3. **Panel shading** - a configurable, clearly labelled approximation.

> **Important:** the dataset has no measured angle or shading variables. The ML model only learns the effect of
> material/fabrication features. The angle and shading effects are **physics-based estimates**, kept in separate code
> (`src/physics.py`) and labelled as estimates throughout the app. They are not ML predictions.

Built with Python, scikit-learn, Streamlit and Plotly (the existing Streamlit stack was kept).

---

## Setup on Windows

Open **PowerShell** in the project folder.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Dataset:** it must be at `dataset\Solar Cell data set2 (1).csv` (already there). If you move it, put it in the `dataset` folder.

**Trained model:** already included in `models\`. To retrain it (about 20 seconds, deterministic):

```powershell
python src\train.py
```

**Run the app:**

```powershell
streamlit run app.py
```

It opens at <http://localhost:8501>. You can also double-click `run_app.bat`, `run_train.bat` or `run_tests.bat`.

**Run the tests:**

```powershell
python -m unittest discover tests -v
```

---

## How to use the app

1. **Material configuration** - pick an example recipe or choose each option (dropdowns list only values that exist in the dataset, with the number of cells in brackets). Click **Predict Composition Efficiency**.
2. **Interactive solar conditions** - two 3D scenes (Three.js): **drag the sun** to change the incidence angle, **drag the cloud** across the panel to change shading (bypass-diode lights turn green/amber/grey). The 3D scenes, sliders and number boxes all stay in sync, and results, charts and cards update instantly without re-running the model. The 3D code is in `components/solar_scene/` (Three.js is bundled, so it works offline).
3. **Prediction results** - baseline PCE, angle-adjusted, shading-adjusted and combined PCE, power in watts (irradiance and panel area are editable), plus the model's MAE, RMSE and R².
4. **Scenario comparison** - save up to 6 scenarios (or click *Add 3 example scenarios*) and compare them side by side. **Reset conditions** / **Reset all** restore defaults.
5. **Physics assumptions** (expander) - switch the shading approximation or change its parameters.

Warnings appear when an input is outside the training-data range or is a rare category; invalid values (e.g. a 0 °C annealing temperature) block the prediction with a clear message.

---

## Machine-learning model

| Item | Detail |
|---|---|
| Dataset | 42,443 records, 30 columns; 41,519 have a valid `JV_default_PCE` and are the only rows used |
| Target | `JV_default_PCE` (%) |
| Inputs | 14 categorical (architecture, substrate, ETL/HTL stacks and deposition, perovskite composition and solvents, back contact...) and 11 numeric/engineered (annealing temperature/time, thicknesses, quenching, encapsulation, additive present, lead-free, mixed cation/halide) |
| Leakage control | `JV_default_Voc`, `JV_default_Jsc`, `JV_default_FF`, `JV_measured` and `Cell_stack_sequence` are **excluded** |
| Preprocessing | Median imputation + RobustScaler for numbers; "Unknown" imputation + one-hot encoding (rare categories grouped) for text. Part of the saved pipeline |
| Outliers | Only impossible PCE values (outside 0-40 %) are removed; unusual but valid measurements are kept, and robust scaling limits their influence. Parsed numeric fields are clipped to plausible ranges |
| Split | 70 % train / 15 % validation / 15 % test, random seed 42; exact duplicate feature rows removed from the training set only |
| Candidates | Dummy mean, Ridge, HistGradientBoosting, Extra Trees, Random Forest |
| Selection | Lowest **validation** MAE -> **Random Forest Regressor** |
| Saved files | `models/sunpredict_pipeline.joblib` (preprocessing + model), `conformal_calibration.joblib`, `ui_options.json`, metrics JSON |

### Actual results (from `python src\train.py`)

Held-out **test** set (6,228 cells never used for training or model selection):

| MAE | RMSE | R² |
|---|---|---|
| 2.615 pp | 3.501 pp | 0.556 |

Validation set comparison:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (mean) | 4.349 | 5.254 | -0.003 |
| Ridge | 3.133 | 4.019 | 0.413 |
| HistGradientBoosting | 2.840 | 3.731 | 0.494 |
| Extra Trees | 2.776 | 3.757 | 0.487 |
| **Random Forest** | **2.654** | **3.553** | **0.541** |

The model explains roughly half of the variation in PCE, so individual predictions are rough guides. A 90 % conformal prediction interval (about +/- 5.7 pp) is shown with each prediction.

**Limitations:** the model learns statistical associations from published lab devices, not causes. Records come from many papers, and a random split can put similar devices in both train and test sets, so error on truly new recipes may be higher. 69 % of rows lack a perovskite thickness (it is imputed in training).

---

## Physical assumptions (`src/physics.py`, `config/solar_assumptions.json`)

```
angle factor   = cos(angle)                 0 deg = perpendicular, 90 deg = grazing (0)
shading factor = f(shaded %)                selectable approximation (below)
effective PCE  = baseline PCE x angle factor x shading factor
power (W)      = beam irradiance (W/m2) x area (m2) x effective PCE / 100
```

* The beam irradiance is measured **perpendicular to the sun's rays**, so the cosine law is applied exactly once (in the angle factor) and never double counted in the power formula.
* Optional diffuse fraction *d*: `(1 - d) cos(angle) + d` (default 0).
* Shading models: **linear** `1 - s`; **bypass-diode substrings** (default: 3 substrings, a substring is bypassed once half of it is shaded); **power law** `(1 - s)^1.5`. 50 % shading does not necessarily mean 50 % loss. These are tunable assumptions, not validated predictions.
* Not modelled: temperature, spectrum, reflection (Fresnel) losses, atmosphere, degradation, lab-cell vs module differences.
* At 0 deg and 0 % shading the result equals the ML baseline exactly.

Edit `config/solar_assumptions.json` to change the defaults.

---

## Project structure

```
app.py                         Streamlit web app (layout and interactions)
config/solar_assumptions.json  Adjustable physics assumptions
src/preprocessing.py           Data loading, cleaning, features, dropdown options
src/train.py                   Training, model selection, evaluation, saving
src/predict.py                 Saved-model inference, input validation, feature importance
src/physics.py                 Angle, shading and combined scenario calculations
src/charts.py                  Plotly charts
src/ui.py                      Dark theme styling, header, summary cards
src/scene.py                   Python wrapper for the 3D scenes
components/solar_scene/        Three.js 3D angle and shade scenes (index.html, three.min.js)
src/evaluate.py                Command-line evaluation audit
tests/                         26 unit tests (physics, data, model)
models/  outputs/  dataset/    Trained artifacts, diagnostic plots, data
.streamlit/config.toml         Light theme
```

(The folder `SunPredict\` is an older copy of the project and is not used by this version.)

## Files changed in this upgrade

**New:** `config/solar_assumptions.json`, `src/physics.py`, `src/charts.py`, `src/ui.py`, `tests/test_physics.py`, `tests/test_model.py`, `.streamlit/config.toml`, `run_tests.bat`, `run_train.bat`, `models/ui_options.json`.

**Modified:** `app.py` (rewritten UI), `src/predict.py` (validation, feature importance, form helpers), `src/preprocessing.py` (dropdown options, path handling), `src/train.py` (saves `ui_options.json`, always writes to this project), `requirements.txt` (+plotly), `run_app.bat`, `README.md`.
