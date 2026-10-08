import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import add_training_rul, load_test_data, load_train_data
from src.model_service import (
    FEATURE_NAMES,
    ModelArtifactError,
    check_manual_values,
    load_model_bundle,
    predict_rul,
    sensor_ranges,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NASA C-MAPSS | RUL Predictor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT PATHS / CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
HERO_IMAGE = PROJECT_ROOT / "assets" / "hero_engine.png"

INPUT_MODES = ["Load Sample Engine", "Manual Sensor Input"]

# Application thresholds for the health label. They are display conventions for
# this demo, not certified maintenance limits.
CRITICAL_MAX_RUL = 50
WARNING_MAX_RUL = 100
HEALTH_SCALE_MAX = 200

# Standard C-MAPSS sensor descriptions (Saxena et al., 2008).
SENSOR_INFO = {
    "sensor_2": ("T24", "Total temperature at LPC outlet", "°R"),
    "sensor_3": ("T30", "Total temperature at HPC outlet", "°R"),
    "sensor_4": ("T50", "Total temperature at LPT outlet", "°R"),
    "sensor_6": ("P15", "Total pressure in bypass duct", "psia"),
    "sensor_7": ("P30", "Total pressure at HPC outlet", "psia"),
    "sensor_8": ("Nf", "Physical fan speed", "rpm"),
    "sensor_9": ("Nc", "Physical core speed", "rpm"),
    "sensor_11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "sensor_12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "sensor_13": ("NRf", "Corrected fan speed", "rpm"),
    "sensor_14": ("NRc", "Corrected core speed", "rpm"),
    "sensor_15": ("BPR", "Bypass ratio", "-"),
    "sensor_17": ("htBleed", "Bleed enthalpy", "-"),
    "sensor_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "sensor_21": ("W32", "LPT coolant bleed", "lbm/s"),
}


# ============================================================
# MODEL AND DATA
# ============================================================

@st.cache_resource
def get_model_bundle():
    return load_model_bundle(MODELS_DIR)


@st.cache_data
def load_fd001():
    train = add_training_rul(load_train_data(DATA_DIR / "train_FD001.txt"))
    test = load_test_data(DATA_DIR / "test_FD001.txt")
    return train, test


@st.cache_data
def get_training_ranges(train_engines: tuple[int, ...]) -> pd.DataFrame:
    """Sensor ranges observed on the engines the model was actually trained on."""
    train, _ = load_fd001()
    return sensor_ranges(train[train["engine_id"].isin(train_engines)])


try:
    bundle = get_model_bundle()
except (ModelArtifactError, OSError, ValueError) as exc:
    st.error(
        f"**The prediction model could not be loaded safely.**\n\n{exc}\n\n"
        "Re-run the export cells in `notebooks/02_EDA_FD001_cleaned.ipynb` to regenerate "
        "`models/xgboost_rul_model.pkl`, `feature_names.pkl`, and `model_metadata.json`."
    )
    st.stop()

train_df, test_df = load_fd001()
ranges = get_training_ranges(tuple(bundle.train_engines))
metrics = bundle.metrics
validation_engines = bundle.validation_engines


# ============================================================
# SESSION STATE
# ============================================================

def default_manual_values() -> dict:
    return {f: float(ranges.loc[f, "median"]) for f in FEATURE_NAMES}


st.session_state.setdefault("loaded_telemetry", None)
st.session_state.setdefault("predictions", {"sample": None, "manual": None})
st.session_state.setdefault("prediction_error", None)
for _feature, _value in default_manual_values().items():
    st.session_state.setdefault(f"manual_{_feature}", _value)


def manual_values() -> dict:
    return {f: float(st.session_state[f"manual_{f}"]) for f in FEATURE_NAMES}


def engine_cycles(engine_id: int) -> list[int]:
    return train_df.loc[train_df["engine_id"] == engine_id, "cycle"].tolist()


def clamp_selected_cycle():
    """Keep the selected cycle valid when the selected engine changes."""
    cycles = engine_cycles(st.session_state.selected_engine)
    if st.session_state.get("selected_cycle") not in cycles:
        current = st.session_state.get("selected_cycle") or cycles[0]
        st.session_state.selected_cycle = min(max(current, cycles[0]), cycles[-1])


# ============================================================
# CALLBACKS
# ============================================================

def go_to(page: str):
    st.session_state.page = page


def load_telemetry():
    engine = st.session_state.selected_engine
    cycle = st.session_state.selected_cycle
    row = train_df[(train_df["engine_id"] == engine) & (train_df["cycle"] == cycle)].iloc[0]
    st.session_state.loaded_telemetry = {
        "engine": int(engine),
        "cycle": int(cycle),
        "values": {f: float(row[f]) for f in FEATURE_NAMES},
        "actual_rul": int(row["RUL"]),
        "max_cycle": int(engine_cycles(engine)[-1]),
    }
    st.session_state.predictions["sample"] = None
    st.session_state.prediction_error = None


def predict_sample():
    telemetry = st.session_state.loaded_telemetry
    selection = (st.session_state.selected_engine, st.session_state.selected_cycle)
    # The button is disabled in this case too, but widget constraints are client-side only.
    if telemetry is None or (telemetry["engine"], telemetry["cycle"]) != selection:
        return
    try:
        rul = predict_rul(bundle.model, telemetry["values"])
    except ValueError as exc:
        st.session_state.prediction_error = str(exc)
        return
    st.session_state.prediction_error = None
    st.session_state.predictions["sample"] = {
        "engine": telemetry["engine"],
        "cycle": telemetry["cycle"],
        "values": dict(telemetry["values"]),
        "rul": rul,
        "actual_rul": telemetry["actual_rul"],
    }


def predict_manual():
    values = manual_values()
    invalid, out_of_range = check_manual_values(values, ranges)
    if invalid or (out_of_range and not st.session_state.get("manual_ack", False)):
        return
    try:
        rul = predict_rul(bundle.model, values)
    except ValueError as exc:
        st.session_state.prediction_error = str(exc)
        return
    st.session_state.prediction_error = None
    st.session_state.predictions["manual"] = {
        "values": values,
        "rul": rul,
        "out_of_range": out_of_range,
    }


def set_manual_values(values: dict):
    for feature in FEATURE_NAMES:
        st.session_state[f"manual_{feature}"] = float(values[feature])
    st.session_state.manual_ack = False


def fill_manual_with_median():
    set_manual_values(default_manual_values())


def copy_loaded_to_manual():
    if st.session_state.loaded_telemetry:
        set_manual_values(st.session_state.loaded_telemetry["values"])


# ============================================================
# PRESENTATION HELPERS
# ============================================================

def render_html(html: str) -> None:
    st.html(html.strip())


def image_to_base64(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8") if path.exists() else None


def format_value(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def sensor_label(feature: str) -> str:
    symbol, _, _ = SENSOR_INFO[feature]
    return f"{feature.replace('_', ' ').title()} · {symbol}"


def health_status(rul: float):
    if rul <= CRITICAL_MAX_RUL:
        return "Critical", "critical"
    if rul <= WARNING_MAX_RUL:
        return "Warning", "warning"
    return "Healthy", "healthy"


def icon(name: str, size: int = 20, color: str = "#0066FF") -> str:
    # st.html strips inline <svg>, so icons are embedded as data-URI images.
    return f'<img class="ico" src="{svg_data_uri(name, color)}" width="{size}" height="{size}" alt="" />'


ICONS = {
    "plane": '<path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5'
             'l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83'
            'l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4'
            'a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1'
            'H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06'
            'A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0'
            ' 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0'
            ' 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "fan": '<path d="M10.83 16.38a6.08 6.08 0 0 1-8.62-7l5.41 1.45a6.08 6.08 0 0 1 7-8.62l-1.45 5.41a6.08 6.08 0 0 1'
           ' 8.62 7l-5.41-1.45a6.08 6.08 0 0 1-7 8.62z"/><path d="M12 12v.01"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>'
            '<line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
}


def card_header(number: int, title: str, right_html: str = "") -> None:
    render_html(f'<div class="card-head"><div class="step"><div class="step-num">{number}</div>'
                f'<div class="step-title">{title}</div></div>{right_html}</div>')


def sensor_grid(values: dict | None) -> None:
    tiles = "".join(
        f'<div class="sensor-tile"><div class="sensor-name">{f.replace("_", " ").title()}'
        f'<span>{SENSOR_INFO[f][0]}</span></div>'
        f'<div class="sensor-value">{format_value(values[f]) if values else "–"}</div></div>'
        for f in FEATURE_NAMES
    )
    render_html(f'<div class="sensor-grid">{tiles}</div>')


def rul_scale(rul: float | None) -> str:
    fill = 0 if rul is None else min(rul, HEALTH_SCALE_MAX) / HEALTH_SCALE_MAX * 100
    ticks = "".join(f"<span>{t}</span>" for t in ("0", "50", "100", "150", f"{HEALTH_SCALE_MAX}+"))
    return (f'<div class="scale"><div class="scale-track"><div class="scale-fill" style="width:{fill:.1f}%"></div></div>'
            f'<div class="scale-ticks">{ticks}</div></div>')


def fact(icon_name: str, label: str, value: str) -> str:
    return (f'<div class="fact">{icon(icon_name, 18, "#475467")}<div><div class="fact-label">{label}</div>'
            f'<div class="fact-value">{value}</div></div></div>')


def result_panel(prediction: dict, source: str) -> None:
    rul = prediction["rul"]
    status, status_class = health_status(rul)

    if source == "sample":
        facts = [
            fact("fan", "Engine", str(prediction["engine"])),
            fact("clock", "Cycle", str(prediction["cycle"])),
            fact("file", "Actual RUL", f"{prediction['actual_rul']} cycles"),
            fact("target", "Absolute Error", f"{abs(rul - prediction['actual_rul']):.1f} cycles"),
        ]
        note = ("Validation engine · not seen in training" if prediction["engine"] in validation_engines
                else "Training engine · in-sample prediction")
    else:
        out_of_range = len(prediction["out_of_range"])
        facts = [
            fact("fan", "Source", "Manual input"),
            fact("alert" if out_of_range else "check", "Input check",
                 f"{out_of_range} value(s) outside the training range" if out_of_range
                 else "All values within the training range"),
        ]
        note = "Manual sensor values"

    render_html(f"""
    <div class="result-panel">
        <div class="result-top">
            <div><div class="result-kicker">Predicted RUL</div>
            <div class="result-number">{rul:.1f}<span> cycles</span></div></div>
            <div class="status-pill status-{status_class}"><i></i>{status}</div>
        </div>
        {rul_scale(rul)}
        <div class="facts facts-{len(facts)}">{''.join(facts)}</div>
        <div class="model-bar">{icon('cpu', 16, '#344054')} XGBoost <b>•</b> 15 Sensors <em>{note}</em></div>
    </div>
    """)


def placeholder_panel(title: str, text: str) -> None:
    render_html(f"""
    <div class="result-panel placeholder">
        <div class="result-top">
            <div><div class="result-kicker">Predicted RUL</div>
            <div class="result-number muted">--<span> cycles</span></div></div>
        </div>
        {rul_scale(None)}
        <div class="placeholder-title">{title}</div>
        <div class="placeholder-text">{text}</div>
        <div class="model-bar">{icon('cpu', 16, '#344054')} XGBoost <b>•</b> 15 Sensors</div>
    </div>
    """)


def threshold_note() -> None:
    st.caption(
        f"Health labels are prototype thresholds, not certified limits: Critical ≤ {CRITICAL_MAX_RUL} · "
        f"Warning {CRITICAL_MAX_RUL + 1}–{WARNING_MAX_RUL} · Healthy > {WARNING_MAX_RUL} cycles."
    )


def svg_data_uri(name: str, color: str = "#0066FF") -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
           f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


HERO_B64 = image_to_base64(HERO_IMAGE)
# The hero asset is a wide banner with text baked into both edges; these crops show only the engine.
ENGINE_CROP = (f"background-image:url(data:image/png;base64,{HERO_B64});background-repeat:no-repeat;"
               "background-size:239% auto;background-position:62% 50%;" if HERO_B64 else "")


# ============================================================
# GLOBAL CSS (layout and custom cards only; colors come from .streamlit/config.toml)
# ============================================================

render_html("""
<style>
/* Spacing scale: one set of tokens, tightened per breakpoint below. */
:root {
    --page-pad-x: 1.5rem; --page-pad-y: 0.75rem;
    --shell-pad: 14px 18px; --card-pad: 12px 16px;
    --gap: 0.6rem; --gap-tight: 0.45rem;
}
.stApp { background: linear-gradient(160deg, #FFFFFF 0%, #F7FBFF 50%, #EAF5FF 100%); }
.block-container {
    max-width: 1120px; width: 100%; box-sizing: border-box;
    padding: var(--page-pad-y) var(--page-pad-x) var(--page-pad-y);
}
/* Safety net only: every element below is sized to fit; this just guards against future content. */
[data-testid="stMain"] { overflow-x: clip; }
#MainMenu, footer, [data-testid="stSidebarCollapsedControl"], section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
.ico { flex: none; max-width: 100%; height: auto; }
.home-title, .pred-title, .step-title, .what-text, .fact-value, .sensor-value, .pill { overflow-wrap: anywhere; }

/* ---------------- Home ---------------- */
/* Center the hero in the viewport instead of leaving a blank lower half. */
[data-testid="stMainBlockContainer"]:has(.st-key-home-hero) { min-height: 100dvh; display: flex; flex-direction: column; }
/* Streamlit's top-level block would otherwise grow to fill the height and pin the hero to the top. */
[data-testid="stMainBlockContainer"]:has(.st-key-home-hero) > [data-testid="stVerticalBlock"] { flex-grow: 0; margin-block: auto; }
.st-key-home-hero {
    width: 100%; max-width: 940px; margin: 0 auto; box-sizing: border-box;
    padding: 34px 40px 26px; border-radius: 26px; overflow: hidden;
    background:
        radial-gradient(130% 55% at 50% 118%, #F4F9FF 52%, rgba(244,249,255,0) 53%),
        radial-gradient(ellipse 34% 16% at 12% 74%, rgba(255,255,255,0.85), transparent 70%),
        radial-gradient(ellipse 30% 13% at 40% 80%, rgba(255,255,255,0.65), transparent 70%),
        radial-gradient(ellipse 34% 15% at 86% 78%, rgba(255,255,255,0.70), transparent 70%),
        radial-gradient(ellipse 60% 40% at 72% 30%, rgba(120,190,255,0.35), transparent 70%),
        linear-gradient(180deg, #0A2B6E 0%, #1453B5 34%, #3F8BE0 58%, #9CCBF5 78%, #DCEEFF 100%);
    box-shadow: 0 18px 40px rgba(10,43,110,0.18);
}
.st-key-home-hero [data-testid="stVerticalBlock"] { gap: 0.75rem; }
.eyebrow { color:#5CD6FF; font-size:13px; font-weight:700; letter-spacing:3px; }
.eyebrow-bar, .title-bar { width:30px; height:2px; background:#38C6F4; border-radius:2px; margin:7px 0 10px; }
.home-title { margin:0; padding:0; color:#FFFFFF; font-size:clamp(1.7rem, 1.1rem + 2.4vw, 2.6rem); font-weight:800; line-height:1.12; }
.home-title span { color:#3DD9F5; }
.title-bar { margin:12px 0; }
.home-tagline { margin:0; color:rgba(255,255,255,0.88); font-size:15px; line-height:1.55; max-width:360px; }
/* In-flow and never wider than its column; the mask blends the crop edges into the sky. */
.engine-art { width:100%; max-width:100%; aspect-ratio: 350 / 190;
              -webkit-mask-image: radial-gradient(ellipse 62% 70% at 56% 50%, #000 48%, transparent 92%);
                      mask-image: radial-gradient(ellipse 62% 70% at 56% 50%, #000 48%, transparent 92%); }
.what { display:flex; gap:14px; align-items:center; width:100%; max-width:580px; box-sizing:border-box;
        margin:2px auto 0; padding:13px 18px;
        background:rgba(255,255,255,0.88); border:1px solid rgba(255,255,255,0.9); border-radius:16px;
        box-shadow:0 10px 30px rgba(10,43,110,0.12); backdrop-filter: blur(6px); }
.what-icon { width:44px; height:44px; border-radius:50%; background:#EAF5FF; color:#0066FF;
             display:flex; align-items:center; justify-content:center; flex:none; }
.what-title { color:#172033; font-size:15px; font-weight:800; margin-bottom:2px; }
.what-text { color:#475467; font-size:13px; line-height:1.5; }
.st-key-cta_prediction button {
    min-height:52px; max-width:100%; border:none; border-radius:999px; background:linear-gradient(90deg,#0B63F6,#1BA9E6);
    box-shadow:0 10px 24px rgba(0,102,255,0.30); transition:transform .15s ease, box-shadow .15s ease;
}
.st-key-cta_prediction button p { font-size:14.5px; font-weight:800; letter-spacing:1.4px; }
.st-key-cta_prediction button:hover { transform:translateY(-1px); box-shadow:0 14px 30px rgba(0,102,255,0.34); }

/* ---------------- Prediction ---------------- */
.st-key-pred-shell {
    width: 100%; box-sizing: border-box; padding: var(--shell-pad); border-radius: 22px; border: 1px solid #E1ECF8;
    background: linear-gradient(180deg, #F3F8FF 0%, #EDF5FF 100%); box-shadow: 0 12px 34px rgba(23,32,51,0.07);
}
.st-key-pred-shell [data-testid="stVerticalBlock"] { gap: var(--gap); }
.st-key-back_home button { min-height:32px; padding:2px 14px; border-radius:999px; background:#FFFFFF;
                           border:1px solid #DCE8F7; box-shadow:0 2px 8px rgba(23,32,51,0.05); }
.st-key-back_home button p { color:#0066FF; font-size:13px; font-weight:700; }
.pred-header { position:relative; display:flex; align-items:center; gap:14px; min-height:60px; }
.pred-header > div:last-child { min-width:0; }
.pred-header .engine-mark { position:absolute; right:0; top:-46px; width:min(300px, 40%); height:130px; opacity:0.16;
    -webkit-mask-image: radial-gradient(ellipse 60% 60% at 60% 50%, #000 40%, transparent 75%);
            mask-image: radial-gradient(ellipse 60% 60% at 60% 50%, #000 40%, transparent 75%); pointer-events:none; }
.header-icon { width:56px; height:56px; border-radius:14px; background:#DCEBFF; color:#0066FF;
               display:flex; align-items:center; justify-content:center; flex:none; }
.pred-title { position:relative; color:#172033; font-size:clamp(1.35rem, 1rem + 1.4vw, 1.85rem); font-weight:800; line-height:1.15; }
.pred-title span { color:#0066FF; }
.pred-sub { position:relative; color:#475467; font-size:14px; margin-top:2px; }

[class*="st-key-card-"] {
    width: 100%; box-sizing: border-box; padding: var(--card-pad); border-radius: 16px; background: rgba(255,255,255,0.78);
    border: 1px solid #E3ECF7; box-shadow: 0 4px 14px rgba(23,32,51,0.04);
}
[class*="st-key-card-"] [data-testid="stVerticalBlock"] { gap: var(--gap-tight); }
.card-head { display:flex; align-items:center; justify-content:space-between; gap:6px 10px; flex-wrap:wrap; }
.step { display:flex; align-items:center; gap:10px; min-width:0; }
.step-num { width:24px; height:24px; border-radius:50%; background:#0066FF; color:#FFF; font-size:12.5px; font-weight:800;
            display:flex; align-items:center; justify-content:center; flex:none; }
.step-title { color:#172033; font-size:15px; font-weight:800; }
.pill { display:inline-flex; align-items:center; gap:7px; max-width:100%; box-sizing:border-box;
        padding:5px 12px; border-radius:999px; font-size:12.5px; font-weight:600; }
.pill-ok { background:#E6F7EE; color:#1F7A4A; }
.pill-muted { background:#F1F5FA; color:#667085; }

/* Engine / cycle fields with icons */
.st-key-field-engine, .st-key-field-cycle {
    box-sizing: border-box; padding: 5px 10px 7px 50px; border-radius: 12px; background-color:#FFFFFF; border:1px solid #E3ECF7;
    background-repeat:no-repeat; background-position:15px 50%; background-size:22px;
}
.st-key-field-engine [data-baseweb="select"] > div, .st-key-field-cycle [data-baseweb="select"] > div {
    background:#FFFFFF; border:1px solid #E3ECF7; min-height:36px;
}
.st-key-field-engine label p, .st-key-field-cycle label p { font-size:12.5px; color:#475467; }
.st-key-load_btn button { min-height:62px; border-radius:12px; box-shadow:0 8px 20px rgba(0,102,255,0.22); }
.st-key-load_btn button p { font-size:15px; font-weight:700; }

.sensor-grid { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:8px; }
.sensor-tile { min-width:0; box-sizing:border-box; padding:7px 12px; border-radius:12px; background:#FFFFFF; border:1px solid #E7EEF7; }
.sensor-name { display:flex; justify-content:space-between; flex-wrap:wrap; gap:0 6px; color:#667085; font-size:12.5px; }
.sensor-name span { color:#A3B1C6; font-size:11px; }
.sensor-value { color:#172033; font-size:16px; font-weight:800; margin-top:1px; }

.st-key-predict_btn button, .st-key-manual_predict_btn button {
    min-height:58px; border:none; border-radius:12px; background:linear-gradient(90deg,#0B63F6,#18B4E0);
    box-shadow:0 10px 22px rgba(0,102,255,0.25);
}
.st-key-predict_btn button p, .st-key-manual_predict_btn button p { font-size:16px; font-weight:800; }
.st-key-predict_btn button:disabled, .st-key-manual_predict_btn button:disabled {
    background:linear-gradient(90deg,#9DBDF2,#A9DCEC); box-shadow:none; opacity:0.75;
}
.st-key-predict_btn button:disabled p, .st-key-manual_predict_btn button:disabled p { color:#FFFFFF; }
.st-key-card-predict [data-testid="stCaptionContainer"] p { font-size:12px; line-height:1.5; }

.result-panel { box-sizing:border-box; padding:12px 16px 10px; border-radius:14px;
                background:linear-gradient(135deg,#FFFFFF 0%,#F2FAFF 100%); border:1px solid #DDEFFA; }
/* Wraps so the status pill drops below the number instead of pushing past the viewport. */
.result-top { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:6px 12px; }
.result-top > div:first-child { min-width:0; }
.result-kicker { color:#172033; font-size:16px; font-weight:800; }
.result-number { color:#172033; font-size:clamp(34px, 26px + 1.6vw, 44px); font-weight:800; line-height:1.05; letter-spacing:-0.5px; }
.result-number span { font-size:20px; font-weight:800; letter-spacing:0; margin-left:6px; white-space:nowrap; }
.result-number.muted { color:#C4CFDB; }
.status-pill { display:inline-flex; align-items:center; gap:8px; padding:7px 16px; border-radius:999px; font-size:15px; font-weight:700; }
.status-pill i { width:12px; height:12px; border-radius:50%; background:currentColor; }
.status-healthy { color:#16A34A; background:#E3F6EA; }
.status-warning { color:#B7791F; background:#FFF4DC; }
.status-critical { color:#D92D20; background:#FDE8E7; }
.scale { margin-top:8px; }
.scale-track { height:8px; border-radius:8px; background:#E3EAF3; overflow:hidden; }
.scale-fill { height:100%; border-radius:8px; background:linear-gradient(90deg,#0066FF,#2E9BFF); }
.scale-ticks { display:flex; justify-content:space-between; margin-top:3px; color:#98A2B3; font-size:11px; }
.facts { display:grid; gap:8px; margin-top:8px; }
.facts-4 { grid-template-columns:repeat(4, minmax(0,1fr)); }
.facts-2 { grid-template-columns:repeat(2, minmax(0,1fr)); }
.fact { display:flex; align-items:center; gap:8px; min-width:0; box-sizing:border-box; padding:6px 10px; border-radius:10px;
        background:#FFFFFF; border:1px solid #E7EEF7; }
.fact > div { min-width:0; }
.fact-label { color:#667085; font-size:11px; }
.fact-value { color:#172033; font-size:13px; font-weight:800; }
.model-bar { display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:4px 6px; margin-top:8px;
             padding:7px 12px; border-radius:10px; background:#E4EFFC; color:#344054; font-size:13px; text-align:center; }
.model-bar b { color:#98A2B3; }
.model-bar em { font-style:normal; color:#667085; font-size:11.5px; margin-left:6px; }
.placeholder-title { margin-top:8px; color:#172033; font-weight:800; font-size:14px; }
.placeholder-text { color:#667085; font-size:12.5px; }

/* ---------------- Breakpoints ---------------- */
/* Tablet landscape */
@media (max-width: 1024px) {
    :root { --page-pad-x: 1.1rem; }
    .sensor-grid { grid-template-columns:repeat(4, minmax(0, 1fr)); }
}
/* Tablet portrait */
@media (max-width: 860px) {
    /* Engine + Cycle share a row; the Load button takes its own full-width row instead of truncating. */
    .st-key-card-select [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-card-select [data-testid="stColumn"]:nth-child(3) { flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }
    .st-key-load_btn button { min-height:50px; }
    .sensor-grid { grid-template-columns:repeat(3, minmax(0, 1fr)); }
    .facts-4 { grid-template-columns:repeat(2, minmax(0,1fr)); }
    .st-key-home-hero { padding: 28px 28px 22px; }
}
/* Mobile: Streamlit stacks columns below 640px */
@media (max-width: 640px) {
    :root { --page-pad-x: 0.75rem; --page-pad-y: 0.6rem; --shell-pad: 12px; --card-pad: 11px 12px; --gap: 0.55rem; }
    [data-testid="stMainBlockContainer"]:has(.st-key-home-hero) { min-height: 0; }
    [data-testid="stMainBlockContainer"]:has(.st-key-home-hero) > [data-testid="stVerticalBlock"] { margin-block: 0; }
    .st-key-home-hero { padding: 22px 18px 20px; border-radius: 22px; }
    .st-key-home-hero [data-testid="stVerticalBlock"] { gap: 0.6rem; }
    .engine-art { aspect-ratio: 350 / 150; }
    .what { padding: 12px 14px; gap: 12px; }
    .what-icon { width:40px; height:40px; }
    .sensor-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .header-icon { width:46px; height:46px; border-radius:12px; }
    .pred-header { min-height:0; gap:12px; }
    .pred-header .engine-mark { display:none; }
    .pred-sub { font-size:13px; }
    .st-key-load_btn button { min-height:50px; }
    .st-key-predict_btn button, .st-key-manual_predict_btn button { min-height:52px; }
}
/* Very narrow phones */
@media (max-width: 360px) {
    :root { --page-pad-x: 0.5rem; --shell-pad: 10px; --card-pad: 10px; }
    .st-key-home-hero { padding: 18px 14px 16px; }
    .eyebrow { letter-spacing:2px; }
    .home-tagline { font-size:14px; }
    .st-key-cta_prediction button p { font-size:13.5px; letter-spacing:0.8px; }
    .sensor-tile { padding:6px 9px; }
    .sensor-value { font-size:15px; }
    .status-pill { padding:6px 12px; font-size:14px; }
    .facts-2, .facts-4 { grid-template-columns:1fr 1fr; gap:6px; }
    .fact { padding:6px 8px; }
}
</style>
""")
render_html(f"""
<style>
.st-key-field-engine {{ background-image:url("{svg_data_uri('fan')}"); }}
.st-key-field-cycle {{ background-image:url("{svg_data_uri('clock')}"); }}
</style>
""")


# ============================================================
# PAGES (navigation is in-page: Home → Prediction → Back to Home)
# ============================================================

st.session_state.setdefault("page", "Home")


def render_home():
    with st.container(key="home-hero"):
        text_col, art_col = st.columns([1.05, 1], vertical_alignment="center")
        text_col.html("""
            <div class="eyebrow">NASA C-MAPSS</div>
            <div class="eyebrow-bar"></div>
            <h1 class="home-title">Aircraft Engine<br><span>RUL Prediction</span></h1>
            <div class="title-bar"></div>
            <p class="home-tagline">Predict the remaining useful life of an aircraft engine using AI.</p>
        """)
        art_col.html(f'<div class="engine-art" style="{ENGINE_CROP}"></div>')
        st.html(f"""
            <div class="what">
                <div class="what-icon">{icon('gear', 24)}</div>
                <div><div class="what-title">What does it do?</div>
                <div class="what-text">An AI-powered prototype that analyzes engine sensor data and estimates
                the remaining useful life of an aircraft engine.</div></div>
            </div>
        """)
        _, middle, _ = st.columns([1, 1.5, 1])
        middle.button("GO TO PREDICTION  →", type="primary", icon=":material/flight:", on_click=go_to,
                      args=("Prediction",), key="cta_prediction", width="stretch")


def render_sample_inputs():
    engine_ids = sorted(train_df["engine_id"].unique().tolist())
    st.session_state.setdefault("selected_engine", engine_ids[0])
    st.session_state.setdefault("selected_cycle", 1)
    clamp_selected_cycle()

    with st.container(key="card-select"):
        card_header(1, "Select Engine and Cycle")
        engine_col, cycle_col, load_col = st.columns([1, 1, 1.05], vertical_alignment="bottom")
        with engine_col.container(key="field-engine"):
            st.selectbox(
                "Engine ID", engine_ids, key="selected_engine", persist_state="session",
                on_change=clamp_selected_cycle,
                format_func=lambda e: f"{e} · {'validation' if e in validation_engines else 'training'}",
                help="Validation engines were held out from training.",
            )
        cycles = engine_cycles(st.session_state.selected_engine)
        with cycle_col.container(key="field-cycle"):
            st.selectbox("Cycle", cycles, key="selected_cycle", persist_state="session",
                         help=f"This engine ran for {cycles[-1]} cycles before failure.")
        load_col.button("Load Engine Telemetry", type="primary", icon=":material/database:",
                        on_click=load_telemetry, key="load_btn", width="stretch")

    loaded = st.session_state.loaded_telemetry
    selection = (st.session_state.selected_engine, st.session_state.selected_cycle)
    selection_matches = loaded is not None and (loaded["engine"], loaded["cycle"]) == selection

    with st.container(key="card-telemetry"):
        if loaded is None:
            pill = f'<span class="pill pill-muted">{icon("clock", 15, "#667085")} No telemetry loaded yet</span>'
        else:
            pill = (f'<span class="pill pill-ok">{icon("check", 16, "#16A34A")} Telemetry loaded: '
                    f'Engine {loaded["engine"]} • Cycle {loaded["cycle"]}</span>')
        card_header(2, f"Engine Telemetry ({len(FEATURE_NAMES)} Sensors)", pill)
        if loaded is not None and not selection_matches:
            st.warning(
                f"The selectors show **Engine {selection[0]} • Cycle {selection[1]}**, but the telemetry "
                f"below is still **Engine {loaded['engine']} • Cycle {loaded['cycle']}**. "
                "Click **Load Engine Telemetry** to use the new selection.",
                icon="⚠️",
            )
        sensor_grid(loaded["values"] if loaded else None)
    return loaded, selection_matches


def render_manual_inputs():
    values = manual_values()
    invalid, out_of_range = check_manual_values(values, ranges)

    with st.container(key="card-manual"):
        card_header(1, "Enter Sensor Values")
        b1, b2 = st.columns(2)
        b1.button("Reset to training medians", on_click=fill_manual_with_median, key="manual_reset", width="stretch")
        b2.button("Copy loaded engine telemetry", on_click=copy_loaded_to_manual, key="manual_copy", width="stretch",
                  disabled=st.session_state.loaded_telemetry is None)

        # One row of columns per 5 sensors, so stacked mobile columns keep the sensor order.
        for i, feature in enumerate(FEATURE_NAMES):
            if i % 5 == 0:
                columns = st.columns(5)
            _, description, unit = SENSOR_INFO[feature]
            low, high = ranges.loc[feature, "min"], ranges.loc[feature, "max"]
            columns[i % 5].number_input(
                sensor_label(feature), key=f"manual_{feature}", format="%.4f", step=None,
                persist_state="session",
                help=f"{description} ({unit}). Training range: {format_value(low)} – {format_value(high)}",
            )

    with st.container(key="card-manual-check"):
        card_header(2, "Input Check")
        if invalid:
            st.error(
                "These values must be greater than zero (they are physical temperatures, pressures, speeds, "
                f"ratios or flows): **{', '.join(sensor_label(f) for f in invalid)}**. Prediction is blocked.",
                icon="⛔",
            )
        if out_of_range:
            rows = pd.DataFrame({
                "Sensor": [sensor_label(f) for f in out_of_range],
                "Value": [values[f] for f in out_of_range],
                "Training min": [ranges.loc[f, "min"] for f in out_of_range],
                "Training max": [ranges.loc[f, "max"] for f in out_of_range],
            })
            st.warning(
                f"{len(out_of_range)} value(s) are outside the range observed in the training data. "
                "The model has never seen such inputs, and tree-based models do not extrapolate, "
                "so the prediction may be unreliable.",
                icon="⚠️",
            )
            st.dataframe(rows, hide_index=True, width="stretch")
            if not invalid:
                st.checkbox("I understand; predict anyway", key="manual_ack")
        if not invalid and not out_of_range:
            st.success("All 15 values are within the training range.", icon="✅")

    blocked = bool(invalid) or (bool(out_of_range) and not st.session_state.get("manual_ack", False))
    return values, blocked


def render_prediction():
    with st.container(key="pred-shell"):
        back_col, mode_col = st.columns([1, 1], vertical_alignment="center")
        back_col.button("Back to Home", icon=":material/arrow_back:", on_click=go_to, args=("Home",),
                        key="back_home")
        with mode_col.container(horizontal_alignment="right"):
            mode = st.radio("Input mode", INPUT_MODES, key="input_mode", horizontal=True, persist_state="session",
                            label_visibility="collapsed", width="content")

        render_html(f"""
        <div class="pred-header">
            <div class="engine-mark" style="{ENGINE_CROP}"></div>
            <div class="header-icon">{icon('plane', 34)}</div>
            <div><div class="pred-title">Engine <span>RUL Prediction</span></div>
            <div class="pred-sub">Estimate the remaining useful life of an aircraft engine using AI.</div></div>
        </div>
        """)

        if mode == INPUT_MODES[0]:
            loaded, selection_matches = render_sample_inputs()
            prediction = st.session_state.predictions["sample"]
            # Show a result only if it was produced from the telemetry currently on screen
            # and the selectors still point at that telemetry.
            current = (
                selection_matches and prediction is not None
                and (prediction["engine"], prediction["cycle"]) == (loaded["engine"], loaded["cycle"])
                and prediction["values"] == loaded["values"]
            )
        else:
            values, blocked = render_manual_inputs()
            prediction = st.session_state.predictions["manual"]
            current = prediction is not None and prediction["values"] == values

        predict_col, result_col = st.columns([1, 2], gap="small")
        with predict_col.container(key="card-predict"):
            card_header(3, "Predict RUL")
            if mode == INPUT_MODES[0]:
                st.button(
                    "Predict RUL", type="primary", icon=":material/play_arrow:", on_click=predict_sample,
                    key="predict_btn", width="stretch", disabled=not selection_matches,
                    help=None if selection_matches else "Load the selected engine telemetry first.",
                )
            else:
                st.button("Predict RUL", type="primary", icon=":material/play_arrow:", on_click=predict_manual,
                          key="manual_predict_btn", width="stretch", disabled=blocked)
            threshold_note()

        with result_col.container(key="card-result"):
            card_header(4, "Prediction Result")
            if st.session_state.prediction_error:
                st.error(f"Prediction failed: {st.session_state.prediction_error}")
            if current:
                result_panel(prediction, "sample" if mode == INPUT_MODES[0] else "manual")
            elif mode == INPUT_MODES[0] and st.session_state.loaded_telemetry is None:
                placeholder_panel("No telemetry loaded", "Select an engine and a cycle, then click "
                                  "<b>Load Engine Telemetry</b>.")
            elif mode == INPUT_MODES[0] and not selection_matches:
                placeholder_panel("Selection not loaded", "The selectors changed. Load the new telemetry "
                                  "before predicting.")
            elif prediction is not None:
                placeholder_panel("Inputs changed", "The sensor values changed since the last prediction. "
                                  "Click <b>Predict RUL</b> again.")
            else:
                placeholder_panel("Ready to predict", "Review the sensor values, then click <b>Predict RUL</b>.")


PAGE_RENDERERS = {
    "Home": render_home,
    "Prediction": render_prediction,
}
PAGE_RENDERERS[st.session_state.page]()
