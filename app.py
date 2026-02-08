import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="GlassHeart | Clinical Decision Support",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------------------------------------
# 2. LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        clinical = joblib.load('models/clinical_model_calibrated.joblib')
        triage = joblib.load('models/triage_model_calibrated.joblib')
        return clinical, triage
    except FileNotFoundError:
        st.error("⚠️ Models not found! Please run 'python train_models.py' locally first.")
        return None, None

clinical_model, triage_model = load_models()

if not clinical_model or not triage_model:
    st.stop()

# ---------------------------------------------------------
# 3. HEADER & MODEL SELECTION
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1], vertical_alignment="bottom")

with col1:
    st.title("❤️ GlassHeart CDSS")
    st.markdown("### Heart Failure Mortality Risk Assessment")

with col2:
    model_choice = st.segmented_control(
        "Assessment Protocol",
        options=["Triage Mode", "Clinical Mode"],
        default="Triage Mode",
        help="Select the assessment depth."
    )

st.divider()

# ---------------------------------------------------------
# 4. INPUT FORM (ROW LAYOUT)
# ---------------------------------------------------------
st.subheader("Patient Vitals & History")

# Dictionary to hold inputs
input_data = {}

# Helpers for consistent UI
def binary_select(label, key):
    return 1 if st.selectbox(label, ("No", "Yes"), key=key) == "Yes" else 0

def sex_select(label, key):
    return 1 if st.selectbox(label, ("Female", "Male"), key=key) == "Male" else 0

if "Clinical" in model_choice:
    # --- CLINICAL MODE INPUTS (11 Features) ---
    
    # Row 1: Demographics & History
    c1, c2, c3, c4 = st.columns(4)
    with c1: input_data['age'] = st.number_input("Age", 40, 95, 60, step=1)
    with c2: input_data['sex'] = sex_select("Sex", "sex_c")
    with c3: input_data['smoking'] = binary_select("Smoking", "smoke_c")
    with c4: input_data['diabetes'] = binary_select("Diabetes", "dia_c")

    # Row 2: Vitals & Lab 1
    c1, c2, c3, c4 = st.columns(4)
    with c1: input_data['ejection_fraction'] = st.number_input("Ejection Fraction (%)", 10, 80, 38)
    with c2: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_c")
    with c3: input_data['anaemia'] = binary_select("Anaemia", "ana_c")
    with c4: input_data['platelets'] = st.number_input("Platelets (k/mL)", 25000.0, 850000.0, 263000.0, step=1000.0)

    # Row 3: Lab 2 (Bio-markers)
    c1, c2, c3 = st.columns(3)
    with c1: input_data['serum_creatinine'] = st.number_input("Serum Creatinine (mg/dL)", 0.5, 9.5, 1.1, step=0.1)
    with c2: input_data['serum_sodium'] = st.number_input("Serum Sodium (mEq/L)", 110, 150, 137, step=1)
    with c3: input_data['creatinine_phosphokinase'] = st.number_input("CPK Level (mcg/L)", 23, 7861, 582, step=10)

    active_model = clinical_model
    # Feature list must match training order exactly
    feature_order = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                     'ejection_fraction', 'high_blood_pressure', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking']

else:
    # --- TRIAGE MODE INPUTS (6 Features) ---
    st.info("ℹ️ Rapid Triage: Requires only patient history and basic observation.")
    
    # Row 1
    c1, c2, c3 = st.columns(3)
    with c1: input_data['age'] = st.number_input("Age", 40, 95, 60, step=1)
    with c2: input_data['sex'] = sex_select("Sex", "sex_t")
    with c3: input_data['smoking'] = binary_select("Smoking", "smoke_t")
    
    # Row 2
    c1, c2, c3 = st.columns(3)
    with c1: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_t")
    with c2: input_data['diabetes'] = binary_select("Diabetes", "dia_t")
    with c3: input_data['anaemia'] = binary_select("Anaemia", "ana_t")
    
    active_model = triage_model
    feature_order = ['age', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# ---------------------------------------------------------
# 5. PREDICTION LOGIC
# ---------------------------------------------------------
st.markdown("###")
predict_btn = st.button("Calculate Mortality Risk", type="primary", use_container_width=True)

if predict_btn:
    # Convert input dict to DataFrame with correct column order
    df_input = pd.DataFrame([input_data])[feature_order]
    
    # Get Probability
    risk_prob = active_model.predict_proba(df_input)[0][1]
    
    # Display Results
    st.markdown("---")
    
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        # Color coding the metric
        color = "normal" if risk_prob < 0.3 else "off" # off is grey, streamlit metric handles delta colors better
        st.metric(
            label="30-Day Mortality Risk", 
            value=f"{risk_prob:.1%}",
            delta="High Risk" if risk_prob > 0.5 else "Low Risk",
            delta_color="inverse"
        )
    
    with c_right:
        if risk_prob > 0.5:
            st.error("⚠️ **High Risk Alert**")
            st.write("Patient requires immediate clinical attention. Consider ICU admission or advanced heart failure therapies.")
        elif risk_prob > 0.3:
            st.warning("⚠️ **Moderate Risk**")
            st.write("Patient requires close monitoring. Review medications and consider additional lab work.")
        else:
            st.success("✅ **Low Risk**")
            st.write("Standard monitoring and routine follow-up advised.")