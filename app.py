"""
Diabetes Prediction App — Multi-page, soothing-blue, icon-based UI
Streamlit front-end with a locked light theme, custom SVG icons,
interactive charts, color-coded risk badges, and SHAP-based explainability.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="⬡", layout="wide")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BLUE_DARK   = "#0B2545"
BLUE_MED    = "#1E5F8C"
BLUE_ACCENT = "#3AA6D9"
BLUE_SOFT   = "#8ECAE6"
BG          = "#EAF6FB"

TEXT_MAIN   = "#0B2545"
TEXT_SOFT   = "#3B5166"

RISK_LOW    = "#1E7A46"
RISK_MED    = "#B4780A"
RISK_HIGH   = "#B3261E"
RISK_LOW_BG  = "rgba(30,122,70,0.10)"
RISK_MED_BG  = "rgba(180,120,10,0.10)"
RISK_HIGH_BG = "rgba(179,38,30,0.10)"


def risk_band(probability: float):
    if probability < 0.30:
        return "Lower risk", RISK_LOW, RISK_LOW_BG, "Indicators are mostly within typical ranges."
    elif probability < 0.70:
        return "Moderate risk", RISK_MED, RISK_MED_BG, "Some indicators fall outside typical ranges."
    else:
        return "Higher risk", RISK_HIGH, RISK_HIGH_BG, "Several indicators fall outside typical ranges."


# ---------------------------------------------------------------------------
# SVG icon set
# ---------------------------------------------------------------------------
def icon(name, size=18, color=BLUE_ACCENT):
    icons = {
        "pregnancy": '<path d="M12 2a4 4 0 0 1 4 4c0 1.5-.8 2.7-2 3.4V11h1a3 3 0 0 1 3 3v2a5 5 0 0 1-10 0v-2a3 3 0 0 1 3-3h1V9.4C10.8 8.7 10 7.5 10 6a4 4 0 0 1 2-4z"/>',
        "droplet": '<path d="M12 2s6 7.2 6 11.5A6 6 0 0 1 6 13.5C6 9.2 12 2 12 2z"/>',
        "pulse": '<path d="M3 12h4l2-7 4 14 2-7h6"/>',
        "ruler": '<path d="M3 8h18v8H3z"/><path d="M7 8v3M11 8v3M15 8v3M19 8v3"/>',
        "vial": '<path d="M9 2h6M10 2v6.5L5.5 17a2 2 0 0 0 1.8 3h9.4a2 2 0 0 0 1.8-3L14 8.5V2"/>',
        "scale": '<circle cx="12" cy="12" r="9"/><path d="M8 12h8M12 8v8"/>',
        "dna": '<path d="M6 3c0 6 12 6 12 12M6 21c0-6 12-6 12-12"/><path d="M8 7h8M8 17h8"/>',
        "calendar": '<rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 9h18M8 2v4M16 2v4"/>',
        "home": '<path d="M4 11l8-7 8 7v9a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1z"/>',
        "chart": '<path d="M4 20V10M11 20V4M18 20v-7"/>',
        "predict": '<circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/>',
        "activity": '<path d="M3 12h4l2-8 4 16 2-8h6"/>',
        "search": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-3.5-3.5"/>',
        "shield": '<path d="M12 2l8 3v6c0 5-3.5 8.5-8 11-4.5-2.5-8-6-8-11V5z"/>',
        "check": '<path d="M5 13l4 4L19 7"/>',
        "alert": '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L14.7 3.86a2 2 0 0 0-3.4 0z"/>',
        "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
        "layers": '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
        "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
        "up": '<path d="M12 19V5M5 12l7-7 7 7"/>',
        "down": '<path d="M12 5v14M19 12l-7 7-7-7"/>',
    }
    path = icons.get(name, "")
    return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
        style="vertical-align:middle;margin-right:6px;flex-shrink:0">{path}</svg>'''


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes softPulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(58,166,217,0.35); }}
    70% {{ box-shadow: 0 0 0 14px rgba(58,166,217,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(58,166,217,0); }}
}}

html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
}}
.stApp {{
    background: radial-gradient(circle at 20% 0%, #FFFFFF 0%, {BG} 60%);
}}

/* ============================================================
   FORCE LIGHT WIDGETS — fixes the dark-navy number-input boxes.
   Streamlit's native inputs otherwise follow the visitor's
   OS/browser color scheme, which fights the light-blue theme.
   Pinned with !important so it can never flip dark again.
   ============================================================ */
div[data-testid="stNumberInput"] > div,
div[data-testid="stNumberInput"] input {{
    background-color: #FFFFFF !important;
    color: {TEXT_MAIN} !important;
    border-color: rgba(30,95,140,0.25) !important;
}}
div[data-testid="stNumberInput"] input {{
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: 1.5px solid rgba(30,95,140,0.22) !important;
}}
div[data-testid="stNumberInput"] button {{
    background-color: {BG} !important;
    border-color: rgba(30,95,140,0.22) !important;
    color: {BLUE_MED} !important;
}}
div[data-testid="stNumberInput"] button:hover {{
    background-color: {BLUE_SOFT} !important;
    color: #FFFFFF !important;
}}
div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {{
    background-color: #FFFFFF !important;
    color: {TEXT_MAIN} !important;
}}
/* Kill the reserved empty space Streamlit leaves for collapsed
   labels — this was overlapping our custom labels and clipping
   the top half of the text off. */
div[data-testid="stWidgetLabel"] {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* ---------- Typography ---------- */
.hero-title {{
    font-size: 2.5rem;
    font-weight: 800;
    color: {TEXT_MAIN};
    animation: fadeInUp 0.6s ease-out;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
}}
.hero-sub {{
    color: {TEXT_SOFT};
    font-size: 1.05rem;
    font-weight: 500;
    animation: fadeInUp 0.8s ease-out;
    margin-bottom: 1.3rem;
}}
.section-label {{
    color: {BLUE_MED};
    font-weight: 700;
    font-size: 1.1rem;
    margin: 1.6rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    animation: fadeInUp 0.5s ease-out;
}}
p.body-text {{ color: {TEXT_SOFT}; font-size: 0.98rem; line-height: 1.6; }}

/* ---------- Cards ---------- */
.info-card {{
    background: #FFFFFF;
    border: 1px solid rgba(30,95,140,0.14);
    border-radius: 14px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.9rem;
    color: {TEXT_MAIN};
    font-weight: 500;
    line-height: 1.5;
    animation: fadeInUp 0.5s ease-out both;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    box-shadow: 0 2px 10px rgba(30,95,140,0.07);
}}
.info-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(58,166,217,0.55);
    box-shadow: 0 10px 24px rgba(30,95,140,0.15);
}}
.info-card b {{ color: {TEXT_MAIN}; }}
.info-card .sub {{ color: {TEXT_SOFT}; font-weight: 400; }}
.stagger-1 {{ animation-delay: 0.05s; }}
.stagger-2 {{ animation-delay: 0.15s; }}
.stagger-3 {{ animation-delay: 0.25s; }}
.stagger-4 {{ animation-delay: 0.35s; }}
.stagger-5 {{ animation-delay: 0.45s; }}

/* ---------- Metric tiles ---------- */
.metric-tile {{
    background: #FFFFFF;
    border: 1px solid rgba(30,95,140,0.14);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: left;
    animation: fadeInUp 0.6s ease-out both;
    box-shadow: 0 2px 10px rgba(30,95,140,0.06);
    transition: transform 0.18s ease;
}}
.metric-tile:hover {{ transform: translateY(-3px); }}
.metric-tile .label {{ color: {TEXT_SOFT}; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }}
.metric-tile .value {{ color: {TEXT_MAIN}; font-size: 1.9rem; font-weight: 800; line-height: 1.3; }}
.metric-tile .tag {{ color: {BLUE_ACCENT}; font-size: 0.85rem; font-weight: 700; }}

/* ---------- Step row (How it works) ---------- */
.step-card {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    border: 1px solid rgba(30,95,140,0.14);
    box-shadow: 0 2px 10px rgba(30,95,140,0.06);
    position: relative;
    animation: fadeInUp 0.6s ease-out both;
}}
.step-num {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    background: {BLUE_ACCENT}; color: #FFF; font-weight: 800; font-size: 0.85rem;
    margin-right: 8px;
}}

/* ---------- Result card ---------- */
.result-card {{
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    margin: 1rem 0;
    animation: fadeInUp 0.5s ease-out, softPulse 2s ease-out 1;
    border: 1.5px solid var(--risk-color);
    background: var(--risk-bg);
}}
.result-card h3 {{ color: var(--risk-color); margin: 0 0 0.3rem 0; display: flex; align-items: center; font-size: 1.5rem; }}
.result-card .prob {{ color: {TEXT_MAIN}; font-size: 2.1rem; font-weight: 800; margin: 0.2rem 0; }}
.result-card .desc {{ color: {TEXT_SOFT}; font-weight: 500; margin: 0; }}

.legend-row {{ display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.4rem; }}
.legend-chip {{ display: flex; align-items: center; gap: 6px; font-size: 0.82rem; font-weight: 600; color: {TEXT_SOFT}; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

.factor-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0.9rem; border-radius: 10px; margin-bottom: 0.4rem;
    background: #FFFFFF; border: 1px solid rgba(30,95,140,0.12);
    font-weight: 600; color: {TEXT_MAIN}; font-size: 0.92rem;
    animation: fadeInUp 0.4s ease-out both;
}}

/* ---------- Reference range table ---------- */
.range-table {{ width: 100%; border-collapse: collapse; }}
.range-table th {{
    text-align: left; color: {BLUE_MED}; font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.3px; padding: 0.5rem 0.7rem; border-bottom: 2px solid rgba(30,95,140,0.15);
}}
.range-table td {{
    padding: 0.55rem 0.7rem; color: {TEXT_MAIN}; font-size: 0.92rem;
    border-bottom: 1px solid rgba(30,95,140,0.08);
}}
.range-table tr:hover td {{ background: rgba(58,166,217,0.06); }}

/* ---------- Buttons ---------- */
div.stButton > button, div.stFormSubmitButton > button {{
    background: linear-gradient(90deg, {BLUE_MED} 0%, {BLUE_ACCENT} 100%);
    color: white; border: none; border-radius: 10px; padding: 0.65rem 1.5rem;
    font-weight: 700; letter-spacing: 0.2px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    width: 100%;
}}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
    transform: translateY(-2px); box-shadow: 0 10px 22px rgba(58,166,217,0.4); filter: brightness(1.05);
}}
div.stButton > button:active, div.stFormSubmitButton > button:active {{ transform: translateY(0px) scale(0.98); }}

div[data-testid="stForm"] {{
    border-radius: 18px; padding: 1.7rem; border: 1px solid rgba(30,95,140,0.14);
    background: #FFFFFF; animation: fadeInUp 0.7s ease-out; box-shadow: 0 2px 14px rgba(30,95,140,0.07);
}}

section[data-testid="stSidebar"] {{ background: #DCEEFB; border-right: 1px solid rgba(30,95,140,0.12); }}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{ color: {TEXT_MAIN} !important; font-weight: 600; }}

.field-label {{
    font-size: 0.92rem; font-weight: 600; color: {TEXT_MAIN};
    margin: 0 0 0.35rem 0; display: flex; align-items: center;
}}

.stCaption, [data-testid="stCaptionContainer"] p {{ color: {TEXT_SOFT} !important; }}

div[data-testid="stExpander"] {{
    border: 1px solid rgba(30,95,140,0.14) !important; border-radius: 12px !important;
    background: #FFFFFF !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    return model, scaler, feature_names, explainer

model, scaler, feature_names, explainer = load_artifacts()

MODEL_RESULTS = {
    "Logistic Regression": {"accuracy": 70.78, "roc_auc": 0.8267,
        "blurb": "A linear baseline that weighs each feature's contribution directly — fast, interpretable, but limited on non-linear patterns."},
    "Random Forest": {"accuracy": 87.01, "roc_auc": 0.9472,
        "blurb": "An ensemble of decision trees voting together — captures non-linear interactions and resists overfitting well."},
    "XGBoost": {"accuracy": 87.66, "roc_auc": 0.9472,
        "blurb": "Gradient-boosted trees built sequentially, each correcting the previous one's errors — typically the strongest raw accuracy."},
    "SVM": {"accuracy": 83.77, "roc_auc": 0.8974,
        "blurb": "Finds the optimal boundary between classes in a transformed feature space — solid but sensitive to scaling."},
    "Voting Ensemble": {"accuracy": 86.36, "roc_auc": 0.9307,
        "blurb": "Combines predictions from multiple models by majority/average vote — trades a little peak accuracy for robustness."},
}

FEATURE_INFO = [
    ("Pregnancies", "Number of times pregnant", "0 – 17", "count"),
    ("Glucose", "Plasma glucose concentration (2-hr oral glucose tolerance test)", "70 – 100 typical fasting; 100–125 prediabetic; 126+ diabetic range", "mg/dL"),
    ("Blood Pressure", "Diastolic blood pressure", "60 – 80 typical", "mm Hg"),
    ("Skin Thickness", "Triceps skinfold thickness", "10 – 30 typical", "mm"),
    ("Insulin", "2-hour serum insulin", "16 – 166 typical", "mu U/mL"),
    ("BMI", "Body mass index", "18.5 – 24.9 typical/normal", "kg/m²"),
    ("Diabetes Pedigree Function", "A score summarizing family history / genetic likelihood", "0.08 – 2.42 (dataset range)", "score"),
    ("Age", "Age in years", "21+ (dataset range)", "years"),
]

@st.cache_data
def load_reference_data():
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    return pd.read_csv(url, names=cols)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:1.4rem">'
        f'{icon("shield", 26)}<span style="font-size:1.15rem;font-weight:800;color:{TEXT_MAIN}">Diabetes Risk AI</span></div>',
        unsafe_allow_html=True
    )
    page = st.radio("Navigate", ["Home", "Train Model", "Make Prediction", "Analytics"], label_visibility="collapsed")

# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
if page == "Home":
    st.markdown(f'<div class="hero-title">{icon("shield", 34)}Diabetes Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">A calm, transparent way to estimate diabetes risk from routine health '
        'measurements — built with tuned machine learning models and explainable predictions.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="info-card stagger-1">{icon("activity",22)}<b>Multiple tuned models</b><br>'
            f'<span class="sub">Logistic Regression, Random Forest, XGBoost, and an ensemble, '
            f'each optimized with cross-validated hyperparameter search.</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="info-card stagger-2">{icon("search",22)}<b>Explainable results</b><br>'
            f'<span class="sub">Every prediction shows exactly which factors influenced it, '
            f'using SHAP value analysis.</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="info-card stagger-3">{icon("shield",22)}<b>Careful data handling</b><br>'
            f'<span class="sub">Hidden missing values in the dataset are detected and '
            f'properly corrected before training.</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{icon("chart",18)}Best model on test data</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(f'<div class="metric-tile stagger-1"><div class="label">Best Accuracy</div><div class="value">87.66%</div><div class="tag">XGBoost</div></div>', unsafe_allow_html=True)
    with b2:
        st.markdown(f'<div class="metric-tile stagger-2"><div class="label">Best ROC-AUC</div><div class="value">0.9472</div><div class="tag">Random Forest</div></div>', unsafe_allow_html=True)
    with b3:
        st.markdown(f'<div class="metric-tile stagger-3"><div class="label">Dataset Size</div><div class="value">768</div><div class="tag">patients · 8 features</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{icon("arrow",18)}How it works</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="step-card stagger-1"><span class="step-num">1</span><b>Enter measurements</b><p class="body-text" style="margin-top:0.5rem">Fill in eight routine clinical values — no lab report needed, just numbers you already have.</p></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="step-card stagger-2"><span class="step-num">2</span><b>Model estimates risk</b><p class="body-text" style="margin-top:0.5rem">The tuned Random Forest model scores the input and returns a probability, shown on a color-coded gauge.</p></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="step-card stagger-3"><span class="step-num">3</span><b>See the "why"</b><p class="body-text" style="margin-top:0.5rem">SHAP analysis breaks down which specific values pushed the estimate up or down.</p></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{icon("book",18)}About the dataset</div>', unsafe_allow_html=True)
    with st.expander("What data was this trained on, and what does each field mean?", expanded=False):
        st.markdown(
            '<p class="body-text">The model is trained on the <b>Pima Indians Diabetes Dataset</b> — '
            '768 records of female patients, each with 8 diagnostic measurements and a binary outcome '
            '(diabetic / non-diabetic). It\'s one of the most widely used benchmark datasets for binary '
            'classification in medical ML.</p>', unsafe_allow_html=True
        )
        rows = "".join(
            f"<tr><td><b>{n}</b></td><td>{d}</td><td>{r}</td><td>{u}</td></tr>"
            for n, d, r, u in FEATURE_INFO
        )
        st.markdown(
            f'<table class="range-table"><tr><th>Field</th><th>Meaning</th><th>Typical range</th><th>Unit</th></tr>{rows}</table>',
            unsafe_allow_html=True
        )
        st.caption("Typical ranges are general population reference points, not diagnostic cutoffs — they exist to help you sanity-check the numbers you enter.")

    st.markdown(f'<div class="section-label">{icon("layers",18)}Methodology</div>', unsafe_allow_html=True)
    with st.expander("How was the model built?", expanded=False):
        st.markdown(
            '''<p class="body-text">
            <b>1. Cleaning hidden missing values</b> — several fields (Glucose, Blood Pressure, Skin Thickness,
            Insulin, BMI) use <code>0</code> as a placeholder for "not recorded," which isn't physiologically possible
            for those measurements. These zeros are treated as missing and imputed rather than taken at face value.<br><br>
            <b>2. Feature scaling</b> — all inputs are standardized (zero mean, unit variance) before being fed
            to the model, since several algorithms here (SVM, Logistic Regression) are sensitive to feature scale.<br><br>
            <b>3. Model comparison</b> — five algorithms were trained and evaluated: Logistic Regression, Random
            Forest, XGBoost, SVM, and a Voting Ensemble. Each was hyperparameter-tuned via cross-validated grid/random
            search before comparing on held-out test data.<br><br>
            <b>4. Explainability</b> — SHAP (SHapley Additive exPlanations) values are computed per-prediction so
            each result comes with a breakdown of which inputs mattered most, rather than a black-box number.
            </p>''', unsafe_allow_html=True
        )

    st.caption("This tool provides a statistical estimate for educational purposes only and is not a medical diagnosis.")

# ---------------------------------------------------------------------------
# TRAIN MODEL PAGE
# ---------------------------------------------------------------------------
elif page == "Train Model":
    st.markdown(f'<div class="hero-title">{icon("chart", 30)}Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">How each algorithm performed after hyperparameter tuning.</div>', unsafe_allow_html=True)

    names = list(MODEL_RESULTS.keys())
    acc = [MODEL_RESULTS[n]["accuracy"] for n in names]
    auc_vals = [MODEL_RESULTS[n]["roc_auc"] * 100 for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=acc, name="Accuracy (%)", marker_color=BLUE_MED,
                          text=[f"{v:.1f}%" for v in acc], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=auc_vals, name="ROC-AUC (x100)", marker_color=BLUE_ACCENT,
                          text=[f"{v:.1f}" for v in auc_vals], textposition="outside"))
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 13}, legend={"orientation": "h", "y": 1.12},
        margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(gridcolor="rgba(11,37,69,0.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'<div class="section-label">{icon("layers",18)}The five models, explained</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, name in enumerate(names):
        with cols[i % 2]:
            info = MODEL_RESULTS[name]
            st.markdown(
                f'<div class="info-card stagger-{(i%3)+1}"><b>{name}</b> '
                f'<span style="color:{BLUE_ACCENT};font-weight:700">· {info["accuracy"]:.2f}% acc · {info["roc_auc"]:.4f} AUC</span><br>'
                f'<span class="sub">{info["blurb"]}</span></div>', unsafe_allow_html=True
            )

    st.markdown(f'<div class="section-label">{icon("search",18)}Why Random Forest was selected for deployment</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-card">Random Forest was chosen for deployment because it achieved the highest '
        f'ROC-AUC (<b>0.9472</b>) with balanced precision and recall across both classes, despite XGBoost scoring '
        f'marginally higher on raw accuracy alone. ROC-AUC is generally a better indicator for medical screening '
        f'tools since it reflects performance across all decision thresholds, not just the default 50% cutoff — '
        f'important when the cost of a missed diabetic case differs from the cost of a false alarm.</div>',
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="section-label">{icon("book",18)}Tuning approach</div>', unsafe_allow_html=True)
    with st.expander("How were hyperparameters chosen?", expanded=False):
        st.markdown(
            '''<p class="body-text">Each model was tuned with k-fold cross-validated grid/random search over its
            key hyperparameters — for example, tree depth and the number of estimators for Random Forest and
            XGBoost, the regularization strength <code>C</code> for Logistic Regression and SVM, and kernel choice
            for SVM. Cross-validation (rather than a single train/test split) was used during tuning so the chosen
            settings generalize rather than overfitting to one particular split of the data.</p>''',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------------------
# MAKE PREDICTION PAGE
# ---------------------------------------------------------------------------
elif page == "Make Prediction":
    st.markdown(f'<div class="hero-title">{icon("predict", 30)}Make a Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter patient measurements below.</div>', unsafe_allow_html=True)

    with st.expander("Not sure what a typical value looks like? Reference ranges here.", expanded=False):
        rows = "".join(f"<tr><td><b>{n}</b></td><td>{r}</td><td>{u}</td></tr>" for n, _, r, u in FEATURE_INFO)
        st.markdown(f'<table class="range-table"><tr><th>Field</th><th>Typical range</th><th>Unit</th></tr>{rows}</table>', unsafe_allow_html=True)

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="field-label">{icon("pregnancy",16)}Pregnancies</div>', unsafe_allow_html=True)
            pregnancies = st.number_input("Pregnancies", 0, 20, 1, label_visibility="collapsed")
            st.markdown(f'<div class="field-label">{icon("droplet",16)}Glucose (mg/dL)</div>', unsafe_allow_html=True)
            glucose = st.number_input("Glucose", 40, 300, 110, label_visibility="collapsed")
            st.markdown(f'<div class="field-label">{icon("pulse",16)}Blood Pressure (mm Hg)</div>', unsafe_allow_html=True)
            blood_pressure = st.number_input("BP", 30, 200, 70, label_visibility="collapsed")
            st.markdown(f'<div class="field-label">{icon("ruler",16)}Skin Thickness (mm)</div>', unsafe_allow_html=True)
            skin_thickness = st.number_input("Skin", 5, 100, 20, label_visibility="collapsed")
        with col2:
            st.markdown(f'<div class="field-label">{icon("vial",16)}Insulin (mu U/mL)</div>', unsafe_allow_html=True)
            insulin = st.number_input("Insulin", 10, 900, 80, label_visibility="collapsed")
            st.markdown(f'<div class="field-label">{icon("scale",16)}BMI</div>', unsafe_allow_html=True)
            bmi = st.number_input("BMI", 10.0, 70.0, 25.0, step=0.1, label_visibility="collapsed")
            st.markdown(f'<div class="field-label">{icon("dna",16)}Diabetes Pedigree Function</div>', unsafe_allow_html=True)
            dpf = st.number_input("DPF", 0.05, 3.0, 0.5, step=0.01, label_visibility="collapsed")
            st.markdown(f'<div class="field-label">{icon("calendar",16)}Age</div>', unsafe_allow_html=True)
            age = st.number_input("Age", 10, 100, 30, label_visibility="collapsed")

        submitted = st.form_submit_button("Predict")

    if submitted:
        errors = []
        if glucose < 50:
            errors.append("Glucose value seems too low to be realistic — please double check.")
        if bmi < 12:
            errors.append("BMI value seems too low to be realistic — please double check.")
        if blood_pressure < 40:
            errors.append("Blood pressure value seems too low — please double check.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                         insulin, bmi, dpf, age]], columns=feature_names)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            label, r_color, r_bg, r_desc = risk_band(probability)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probability * 100,
                number={"suffix": "%", "font": {"size": 40, "color": TEXT_MAIN}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": TEXT_SOFT, "tickfont": {"color": TEXT_SOFT}},
                    "bar": {"color": r_color}, "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30], "color": RISK_LOW_BG},
                        {"range": [30, 70], "color": RISK_MED_BG},
                        {"range": [70, 100], "color": RISK_HIGH_BG},
                    ],
                },
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10),
                                     paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT_MAIN})
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(
                f'''<div class="legend-row">
                    <span class="legend-chip"><span class="legend-dot" style="background:{RISK_LOW}"></span>Lower (0-30%)</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:{RISK_MED}"></span>Moderate (30-70%)</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:{RISK_HIGH}"></span>Higher (70-100%)</span>
                </div>''', unsafe_allow_html=True
            )

            icon_name = "check" if prediction == 0 else "alert"
            st.markdown(
                f'''<div class="result-card" style="--risk-color:{r_color};--risk-bg:{r_bg}">
                    <h3>{icon(icon_name, 24, r_color)}{label} of diabetes</h3>
                    <p class="prob">{probability*100:.1f}%</p>
                    <p class="desc">{r_desc}</p>
                </div>''', unsafe_allow_html=True
            )
            st.caption("This is a statistical estimate from a machine learning model, not a medical diagnosis. Please consult a healthcare professional for actual medical advice.")

            st.markdown(f'<div class="section-label">{icon("search",18)}Why this prediction?</div>', unsafe_allow_html=True)
            try:
                sv_raw = explainer.shap_values(input_scaled)
                if isinstance(sv_raw, list):
                    sv = sv_raw[1][0]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                elif sv_raw.ndim == 3:
                    sv = sv_raw[0, :, 1]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    sv = sv_raw[0]
                    base_value = explainer.expected_value

                # Dynamic plain-English top factors, computed from the real SHAP values for THIS input.
                order = np.argsort(-np.abs(sv))[:3]
                st.markdown('<div style="margin-bottom:0.8rem">', unsafe_allow_html=True)
                for idx in order:
                    fname = feature_names[idx]
                    fval = input_data.iloc[0][fname]
                    direction = "up" if sv[idx] > 0 else "down"
                    d_color = RISK_HIGH if sv[idx] > 0 else RISK_LOW
                    d_word = "increased" if sv[idx] > 0 else "decreased"
                    st.markdown(
                        f'<div class="factor-row"><span>{fname} = {fval:g}</span>'
                        f'<span style="color:{d_color}">{icon(direction, 16, d_color)}{d_word} risk</span></div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                explanation = shap.Explanation(values=sv, base_values=base_value,
                                                data=input_data.iloc[0].values, feature_names=feature_names)
                plt.rcParams.update({
                    "text.color": TEXT_MAIN, "axes.labelcolor": TEXT_MAIN,
                    "xtick.color": TEXT_MAIN, "ytick.color": TEXT_MAIN,
                    "axes.edgecolor": TEXT_SOFT, "font.size": 11,
                })
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                shap.plots.bar(explanation, show=False)
                for txt in ax.texts:
                    txt.set_color(TEXT_MAIN)
                    txt.set_fontweight("bold")
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True, transparent=True)
                st.caption("Bars show which factors pushed the prediction toward higher or lower risk.")
            except Exception as e:
                st.info(f"Explanation unavailable for this input. ({e})")

# ---------------------------------------------------------------------------
# ANALYTICS PAGE
# ---------------------------------------------------------------------------
elif page == "Analytics":
    st.markdown(f'<div class="hero-title">{icon("chart", 30)}Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Dataset patterns and what drives the model\'s predictions overall.</div>', unsafe_allow_html=True)

    df = load_reference_data()

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f'<div class="section-label">{icon("activity",18)}Class balance</div>', unsafe_allow_html=True)
        counts = df["Outcome"].value_counts().rename({0: "Non-Diabetic", 1: "Diabetic"})
        fig_pie = px.pie(values=counts.values, names=counts.index, hole=0.55,
                          color_discrete_sequence=[BLUE_MED, RISK_HIGH])
        fig_pie.update_traces(textfont_color="#FFFFFF", textfont_size=14)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT_MAIN},
                              margin=dict(t=10, b=10, l=10, r=10), legend={"font": {"color": TEXT_MAIN}})
        st.plotly_chart(fig_pie, use_container_width=True)
        pct_diabetic = counts.get("Diabetic", 0) / counts.sum() * 100
        st.markdown(f'<p class="body-text">About <b style="color:{TEXT_MAIN}">{pct_diabetic:.1f}%</b> of the reference dataset is labeled diabetic — a moderate class imbalance that the model was tuned to handle rather than simply predicting the majority class.</p>', unsafe_allow_html=True)

    with colB:
        st.markdown(f'<div class="section-label">{icon("search",18)}What drives predictions most</div>', unsafe_allow_html=True)
        try:
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
            fig_imp = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color=BLUE_ACCENT))
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font={"color": TEXT_MAIN}, margin=dict(t=10, b=10, l=10, r=10),
                                   xaxis=dict(gridcolor="rgba(11,37,69,0.08)"))
            st.plotly_chart(fig_imp, use_container_width=True)
            top_feat = importances.index[-1]
            st.markdown(f'<p class="body-text">Across the whole model — not just one prediction — <b style="color:{TEXT_MAIN}">{top_feat}</b> carries the most predictive weight, consistent with its clinical significance for diabetes risk.</p>', unsafe_allow_html=True)
        except AttributeError:
            st.info("Feature importances aren't available for this model type.")

    st.markdown(f'<div class="section-label">{icon("droplet",18)}Feature distributions by outcome</div>', unsafe_allow_html=True)
    feature_choice = st.selectbox("Feature", feature_names, label_visibility="collapsed")
    fig_hist = px.histogram(df, x=feature_choice, color=df["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}),
                             barmode="overlay", opacity=0.75, color_discrete_sequence=[BLUE_MED, RISK_HIGH])
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": TEXT_MAIN}, legend_title_text="", legend={"font": {"color": TEXT_MAIN}},
                           xaxis=dict(gridcolor="rgba(11,37,69,0.06)"), yaxis=dict(gridcolor="rgba(11,37,69,0.06)"))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f'<div class="section-label">{icon("layers",18)}How features relate to each other</div>', unsafe_allow_html=True)
    corr = df[feature_names + ["Outcome"]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale=["#FFFFFF", BLUE_MED, BLUE_DARK],
                          aspect="auto")
    fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": TEXT_MAIN}, margin=dict(t=10, b=10, l=10, r=10),
                           coloraxis_colorbar={"tickfont": {"color": TEXT_MAIN}})
    fig_corr.update_xaxes(tickfont={"color": TEXT_MAIN})
    fig_corr.update_yaxes(tickfont={"color": TEXT_MAIN})
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown('<p class="body-text">Darker cells indicate a stronger relationship between two fields. Glucose typically shows the strongest direct correlation with the diabetes outcome in this dataset.</p>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{icon("book",18)}Summary statistics</div>', unsafe_allow_html=True)
    with st.expander("View raw dataset statistics (mean, std, min/max, quartiles)", expanded=False):
        st.dataframe(df[feature_names + ["Outcome"]].describe().T.style.format("{:.2f}"), use_container_width=True)

st.markdown(f'<hr style="border-color:rgba(142,202,230,0.25)">', unsafe_allow_html=True)
st.caption("Built with scikit-learn, XGBoost, SHAP, and Plotly.")
