"""
Diabetes Prediction App
Streamlit front-end for the trained model, with input validation
and SHAP-based explanation of each prediction.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

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

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter the patient's medical details below. This tool estimates diabetes "
    "risk using a machine learning model trained on the PIMA Indians Diabetes Dataset, "
    "and explains *why* it made that prediction."
)

st.divider()

# ---------------------------------------------------------------------------
# Input form with validation
# ---------------------------------------------------------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=300, value=110)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=30, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=5, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin (mu U/mL)", min_value=10, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)

    submitted = st.form_submit_button("Predict")

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

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Higher risk of diabetes  —  estimated probability: {probability*100:.1f}%")
        else:
            st.success(f"✅ Lower risk of diabetes  —  estimated probability: {probability*100:.1f}%")

        st.caption(
            "This is a statistical estimate from a machine learning model, not a medical diagnosis. "
            "Please consult a healthcare professional for actual medical advice."
        )

        # ---- SHAP explanation ----
        st.subheader("Why this prediction?")
        try:
            shap_values = explainer.shap_values(input_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1 (diabetic)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(
                shap_values, input_data, plot_type="bar",
                show=False, plot_size=None
            )
            st.pyplot(fig, clear_figure=True)
            st.caption("Bars show which factors pushed the prediction toward higher or lower risk.")
        except Exception as e:
            st.info("Explanation unavailable for this input.")

st.divider()
st.caption("Built with scikit-learn, XGBoost, SHAP, and Streamlit.")
