"""
Diabetes Prediction App — Multi-page, soothing-blue, icon-based UI
Streamlit front-end with custom theme, SVG icons, interactive charts,
color-coded risk badges, and SHAP-based explainability.
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
# Palette — TEXT_SOFT darkened from the original for real AA contrast on BG.
# Risk colors added so the result is instantly readable, not just colored text.
# ---------------------------------------------------------------------------
BLUE_DARK   = "#0B2545"
BLUE_MED    = "#1E5F8C"
BLUE_ACCENT = "#3AA6D9"
BLUE_SOFT   = "#8ECAE6"
BG          = "#EAF6FB"
CARD_BG     = "rgba(58,166,217,0.06)"

TEXT_MAIN   = "#0B2545"   # primary text / numbers — always this, never a light gray
TEXT_SOFT   = "#3B5166"   # secondary text — darkened from #4A6178 for contrast

RISK_LOW    = "#1E7A46"   # green  — low risk
RISK_MED    = "#B4780A"   # amber  — moderate risk (darkened for contrast on white)
RISK_HIGH   = "#B3261E"   # red    — high risk

RISK_LOW_BG  = "rgba(30,122,70,0.10)"
RISK_MED_BG  = "rgba(180,120,10,0.10)"
RISK_HIGH_BG = "rgba(179,38,30,0.10)"


def risk_band(probability: float):
    """Return (label, color, bg, description) for a 0-1 probability."""
    if probability < 0.30:
        return "Lower risk", RISK_LOW, RISK_LOW_BG, "Indicators are mostly within typical ranges."
    elif probability < 0.70:
        return "Moderate risk", RISK_MED, RISK_MED_BG, "Some indicators fall outside typical ranges."
    else:
        return "Higher risk", RISK_HIGH, RISK_HIGH_BG, "Several indicators fall outside typical ranges."


# ---------------------------------------------------------------------------
# SVG icon set (heroicons-style outline icons, single color, no branding)
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
@keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}

html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 20% 0%, #FFFFFF 0%, {BG} 60%);
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
    font-size: 1.05rem;
    margin: 1.4rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    animation: fadeInUp 0.5s ease-out;
}}

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

/* staggered entrance for the 3 home cards */
.stagger-1 {{ animation-delay: 0.05s; }}
.stagger-2 {{ animation-delay: 0.15s; }}
.stagger-3 {{ animation-delay: 0.25s; }}

/* ---------- Custom metric tiles (replaces st.metric so colors stay ours) ---------- */
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
.metric-tile .label {{
    color: {TEXT_SOFT};
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}}
.metric-tile .value {{
    color: {TEXT_MAIN};
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.3;
}}
.metric-tile .tag {{
    color: {BLUE_ACCENT};
    font-size: 0.85rem;
    font-weight: 700;
}}

/* ---------- Result card: color-coded by risk, not just decorative ---------- */
.result-card {{
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    margin: 1rem 0;
    animation: fadeInUp 0.5s ease-out, softPulse 2s ease-out 1;
    border: 1.5px solid var(--risk-color);
    background: var(--risk-bg);
}}
.result-card h3 {{
    color: var(--risk-color);
    margin: 0 0 0.3rem 0;
    display: flex;
    align-items: center;
    font-size: 1.5rem;
}}
.result-card .prob {{
    color: {TEXT_MAIN};
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0.2rem 0;
}}
.result-card .desc {{ color: {TEXT_SOFT}; font-weight: 500; margin: 0; }}

.risk-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    background: var(--risk-color);
    color: #FFFFFF;
}}

/* legend chips under the gauge */
.legend-row {{ display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.4rem; }}
.legend-chip {{
    display: flex; align-items: center; gap: 6px;
    font-size: 0.82rem; font-weight: 600; color: {TEXT_SOFT};
}}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

/* ---------- Buttons ---------- */
div.stButton > button, div.stFormSubmitButton > button {{
    background: linear-gradient(90deg, {BLUE_MED} 0%, {BLUE_ACCENT} 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    width: 100%;
}}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(58,166,217,0.4);
    filter: brightness(1.05);
}}
div.stButton > button:active, div.stFormSubmitButton > button:active {{
    transform: translateY(0px) scale(0.98);
}}

div[data-testid="stForm"] {{
    border-radius: 18px;
    padding: 1.7rem;
    border: 1px solid rgba(30,95,140,0.14);
    background: #FFFFFF;
    animation: fadeInUp 0.7s ease-out;
    box-shadow: 0 2px 14px rgba(30,95,140,0.07);
}}

section[data-testid="stSidebar"] {{
    background: #DCEEFB;
    border-right: 1px solid rgba(30,95,140,0.12);
}}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{
    color: {TEXT_MAIN} !important;
    font-weight: 600;
}}

.field-label {{
    font-size: 0.92rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    margin-bottom: -0.6rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
}}

/* Make Streamlit's own caption / info text readable on the light bg */
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {TEXT_SOFT} !important;
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
    "Logistic Regression": {"accuracy": 70.78, "roc_auc": 0.8267},
    "Random Forest": {"accuracy": 87.01, "roc_auc": 0.9472},
    "XGBoost": {"accuracy": 87.66, "roc_auc": 0.9472},
    "SVM": {"accuracy": 83.77, "roc_auc": 0.8974},
    "Voting Ensemble": {"accuracy": 86.36, "roc_auc": 0.9307},
}

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
    page = st.radio(
        "Navigate",
        ["Home", "Train Model", "Make Prediction", "Analytics"],
        label_visibility="collapsed"
    )

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
            f'each optimized with cross-validated hyperparameter search.</span></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="info-card stagger-2">{icon("search",22)}<b>Explainable results</b><br>'
            f'<span class="sub">Every prediction shows exactly which factors influenced it, '
            f'using SHAP value analysis.</span></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="info-card stagger-3">{icon("shield",22)}<b>Careful data handling</b><br>'
            f'<span class="sub">Hidden missing values in the dataset are detected and '
            f'properly corrected before training.</span></div>',
            unsafe_allow_html=True
        )

    st.markdown(f'<div class="section-label">{icon("chart",18)}Best model on test data</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(
            f'<div class="metric-tile stagger-1"><div class="label">Best Accuracy</div>'
            f'<div class="value">87.66%</div><div class="tag">XGBoost</div></div>',
            unsafe_allow_html=True
        )
    with b2:
        st.markdown(
            f'<div class="metric-tile stagger-2"><div class="label">Best ROC-AUC</div>'
            f'<div class="value">0.9472</div><div class="tag">Random Forest</div></div>',
            unsafe_allow_html=True
        )
    with b3:
        st.markdown(
            f'<div class="metric-tile stagger-3"><div class="label">Dataset Size</div>'
            f'<div class="value">768</div><div class="tag">patients · 8 features</div></div>',
            unsafe_allow_html=True
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
    auc = [MODEL_RESULTS[n]["roc_auc"] * 100 for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=acc, name="Accuracy (%)", marker_color=BLUE_MED,
                          text=[f"{v:.1f}%" for v in acc], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=auc, name="ROC-AUC (x100)", marker_color=BLUE_ACCENT,
                          text=[f"{v:.1f}" for v in auc], textposition="outside"))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 13},
        legend={"orientation": "h", "y": 1.12},
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(gridcolor="rgba(11,37,69,0.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'<div class="section-label">{icon("search",18)}Why Random Forest was selected</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-card">Random Forest was chosen for deployment because it achieved the highest '
        f'ROC-AUC (<b>0.9472</b>) with balanced precision and recall across both classes, despite XGBoost scoring '
        f'marginally higher on raw accuracy alone.</div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# MAKE PREDICTION PAGE
# ---------------------------------------------------------------------------
elif page == "Make Prediction":
    st.markdown(f'<div class="hero-title">{icon("predict", 30)}Make a Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter patient measurements below.</div>', unsafe_allow_html=True)

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
            input_data = pd.DataFrame([[
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age
            ]], columns=feature_names)

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            label, r_color, r_bg, r_desc = risk_band(probability)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 40, "color": TEXT_MAIN}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": TEXT_SOFT, "tickfont": {"color": TEXT_SOFT}},
                    "bar": {"color": r_color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30], "color": RISK_LOW_BG},
                        {"range": [30, 70], "color": RISK_MED_BG},
                        {"range": [70, 100], "color": RISK_HIGH_BG},
                    ],
                },
            ))
            fig_gauge.update_layout(
                height=260, margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT_MAIN},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(
                f'''<div class="legend-row">
                    <span class="legend-chip"><span class="legend-dot" style="background:{RISK_LOW}"></span>Lower (0-30%)</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:{RISK_MED}"></span>Moderate (30-70%)</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:{RISK_HIGH}"></span>Higher (70-100%)</span>
                </div>''',
                unsafe_allow_html=True
            )

            icon_name = "check" if prediction == 0 else "alert"
            st.markdown(
                f'''<div class="result-card" style="--risk-color:{r_color};--risk-bg:{r_bg}">
                    <h3>{icon(icon_name, 24, r_color)}{label} of diabetes</h3>
                    <p class="prob">{probability*100:.1f}%</p>
                    <p class="desc">{r_desc}</p>
                </div>''',
                unsafe_allow_html=True
            )
            st.caption(
                "This is a statistical estimate from a machine learning model, not a medical diagnosis. "
                "Please consult a healthcare professional for actual medical advice."
            )

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

                explanation = shap.Explanation(
                    values=sv, base_values=base_value,
                    data=input_data.iloc[0].values, feature_names=feature_names,
                )

                # Force dark, high-contrast text everywhere in the matplotlib figure —
                # this is what was making the SHAP explanation hard to read.
                plt.rcParams.update({
                    "text.color": TEXT_MAIN,
                    "axes.labelcolor": TEXT_MAIN,
                    "xtick.color": TEXT_MAIN,
                    "ytick.color": TEXT_MAIN,
                    "axes.edgecolor": TEXT_SOFT,
                    "font.size": 11,
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
        fig_pie = px.pie(
            values=counts.values, names=counts.index, hole=0.55,
            color_discrete_sequence=[BLUE_MED, RISK_HIGH]
        )
        fig_pie.update_traces(textfont_color="#FFFFFF", textfont_size=14)
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT_MAIN},
            margin=dict(t=10, b=10, l=10, r=10),
            legend={"font": {"color": TEXT_MAIN}},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with colB:
        st.markdown(f'<div class="section-label">{icon("search",18)}What drives predictions most</div>', unsafe_allow_html=True)
        try:
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
            fig_imp = go.Figure(go.Bar(
                x=importances.values, y=importances.index, orientation="h",
                marker_color=BLUE_ACCENT
            ))
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": TEXT_MAIN}, margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="rgba(11,37,69,0.08)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except AttributeError:
            st.info("Feature importances aren't available for this model type.")

    st.markdown(f'<div class="section-label">{icon("droplet",18)}Feature distributions by outcome</div>', unsafe_allow_html=True)
    feature_choice = st.selectbox("Feature", feature_names, label_visibility="collapsed")
    fig_hist = px.histogram(
        df, x=feature_choice, color=df["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}),
        barmode="overlay", opacity=0.75,
        color_discrete_sequence=[BLUE_MED, RISK_HIGH]
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN}, legend_title_text="",
        legend={"font": {"color": TEXT_MAIN}},
        xaxis=dict(gridcolor="rgba(11,37,69,0.06)"),
        yaxis=dict(gridcolor="rgba(11,37,69,0.06)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown(f'<hr style="border-color:rgba(142,202,230,0.25)">', unsafe_allow_html=True)
st.caption("Built with scikit-learn, XGBoost, SHAP, and Plotly.")
