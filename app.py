import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="GlassHeart | Clinical Decision Support",
    page_icon="❤️",
    layout="wide"
)

# 2. Load Model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    # Make sure this path matches where you put your .joblib file
    model = joblib.load('models/RF_ef+cr+na.joblib')
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'models/RF_ef+cr+na.joblib' exists.")
    st.stop()

# 3. Sidebar: Patient Vitals Input
st.sidebar.header("Patient Vitals")
st.sidebar.markdown("Enter the patient's clinical metrics below:")

def user_input_features():
    # Numeric Inputs
    age = st.sidebar.slider("Age", 40, 95, 60)
    ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 14, 80, 38)
    serum_creatinine = st.sidebar.number_input("Serum Creatinine (mg/dL)", 0.5, 9.4, 1.1)
    serum_sodium = st.sidebar.slider("Serum Sodium (mEq/L)", 113, 148, 137)
    platelets = st.sidebar.number_input("Platelets (kiloplatelets/mL)", 25000.0, 850000.0, 263000.0)
    creatinine_phosphokinase = st.sidebar.number_input("CPK Level (mcg/L)", 23, 7861, 582)
    
    # Categorical Inputs
    anaemia = st.sidebar.selectbox("Anaemia", ("No", "Yes"))
    diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
    high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", ("No", "Yes"))
    sex = st.sidebar.selectbox("Sex", ("Female", "Male"))
    smoking = st.sidebar.selectbox("Smoking", ("No", "Yes"))

    # Convert to DataFrame
    data = {
        'age': age,
        'anaemia': 1 if anaemia == "Yes" else 0,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': 1 if diabetes == "Yes" else 0,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': 1 if high_blood_pressure == "Yes" else 0,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': 1 if sex == "Male" else 0,
        'smoking': 1 if smoking == "Yes" else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 4. Main Dashboard Area
st.title("❤️ GlassHeart")
st.markdown("### Heart Failure Mortality Risk Predictor")
st.markdown("This Clinical Decision Support System (CDSS) estimates the probability of mortality using a calibrated Random Forest classifier.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Profile")
    st.write(input_df)

    if st.button("Analyze Risk"):
        # Make Prediction
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        # Display Result
        st.markdown("---")
        st.subheader("Analysis Result")
        
        if prediction_proba > 0.5:
            st.error(f"⚠️ High Mortality Risk: {prediction_proba:.1%}")
            st.markdown("**Recommendation:** Initiate high-risk protocol and consider ICU admission.")
        else:
            st.success(f"✅ Low Mortality Risk: {prediction_proba:.1%}")
            st.markdown("**Recommendation:** Standard monitoring and follow-up.")

with col2:
    st.info("ℹ️ **Model Info**")
    st.markdown("""
    - **Model:** Random Forest Classifier
    - **Calibration:** Isotonic Regression
    - **Validation:** DeLong's Test (p < 0.05)
    - **Key Drivers:** Ejection Fraction, Serum Creatinine, Serum Sodium
    """)