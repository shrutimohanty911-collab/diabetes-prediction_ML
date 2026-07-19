"""
Diabetes Prediction App — Polished UI version
Streamlit front-end with custom theme, animated interactive gauge,
and SHAP-based explainability.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------------------
# Custom CSS — gradient header, animated cards, hover effects
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.main-header {
    background: linear-gradient(90deg, #7C3AED 0%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    padding-bottom: 0.2rem;
    animation: fadeInUp 0.6s ease-out;
}
.sub-header {
    text-align: center;
    color: #B8B3C7;
    font-size: 1rem;
    margin-bottom: 1.5rem;
    animation: fadeInUp 0.8s ease-out;
}
.result-card {
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin: 1rem 0;
    animation: fadeInUp 0.5s ease-out;
    border: 1px solid rgba(255,255,255,0.08);
}
.result-high {
    background: linear-gradient(135deg, rgba(236,72,153,0.15), rgba(236,72,153,0.05));
}
.result-low {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(124,58,237,0.05));
}
div.stButton > button {
    background: linear-gradient(90deg, #7C3AED 0%, #EC4899 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(124,58,237,0.4);
}
div[data-testid="stForm"] {
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    animation: fadeInUp 0.7s ease-out;
}
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

st.markdown('<div class="main-header">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered risk estimation with explainable predictions</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Input form with validation
# ---------------------------------------------------------------------------
with st.form("input_form"):
    st.markdown("#### 🧑‍⚕️ Enter Patient Details")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("🩸 Glucose (mg/dL)", min_value=40, max_value=300, value=110)
        blood_pressure = st.number_input("💉 Blood Pressure (mm Hg)", min_value=30, max_value=200, value=70)
        skin_thickness = st.number_input("📏 Skin Thickness (mm)", min_value=5, max_value=100, value=20)

    with col2:
        insulin = st.number_input("🧪 Insulin (mu U/mL)", min_value=10, max_value=900, value=80)
        bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("🎂 Age", min_value=10, max_value=100, value=30, step=1)

    submitted = st.form_submit_button("🔮 Predict Now")

if submitted:
    # ---- Validation ----
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

        # ---- Animated gauge chart ----
        gauge_color = "#EC4899" if prediction == 1 else "#7C3AED"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#F2F2F7"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#B8B3C7"},
                "bar": {"color": gauge_color},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(124,58,237,0.15)"},
                    {"range": [40, 70], "color": "rgba(236,72,153,0.15)"},
                    {"range": [70, 100], "color": "rgba(236,72,153,0.3)"},
                ],
            },
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F2F2F7"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        card_class = "result-high" if prediction == 1 else "result-low"
        label = "⚠️ Higher risk of diabetes" if prediction == 1 else "✅ Lower risk of diabetes"
        st.markdown(
            f'<div class="result-card {card_class}"><h3>{label}</h3>'
            f'<p>Estimated probability: <b>{probability*100:.1f}%</b></p></div>',
            unsafe_allow_html=True
        )

        st.caption(
            "This is a statistical estimate from a machine learning model, not a medical diagnosis. "
            "Please consult a healthcare professional for actual medical advice."
        )

        # ---- SHAP explanation ----
        st.markdown("#### 🔍 Why this prediction?")
        try:
            sv_raw = explainer.shap_values(input_scaled)

            if isinstance(sv_raw, list):
                sv = sv_raw[1][0]
                base_value = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )
            elif sv_raw.ndim == 3:
                sv = sv_raw[0, :, 1]
                base_value = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )
            else:
                sv = sv_raw[0]
                base_value = explainer.expected_value

            explanation = shap.Explanation(
                values=sv,
                base_values=base_value,
                data=input_data.iloc[0].values,
                feature_names=feature_names,
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            shap.plots.bar(explanation, show=False)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True, transparent=True)
            st.caption("Bars show which factors pushed the prediction toward higher or lower risk.")
        except Exception as e:
            st.info(f"Explanation unavailable for this input. ({e})")

st.divider()
st.caption("Built with scikit-learn, XGBoost, SHAP, Plotly, and Streamlit.")
