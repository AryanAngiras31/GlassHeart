import streamlit as st
import pandas as pd
import joblib

import shap
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="GlassHeart | Clinical Decision Support",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
    """
    <style>
      /* Card container */
      .gh-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 20px;
      }

      /* Section title */
      .gh-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f0f2f6;
      }

      /* Muted text */
      .gh-muted { color: rgba(255,255,255,0.6); font-size: 0.9rem; }
      
      /* Make Segmented Control Larger */
      div[data-baseweb="segmented-control"] button {
        font-size: 1.1rem !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
      }
      
      /* Custom Metric Style for Card */
      .gh-metric-label { font-size: 0.9rem; color: rgba(255,255,255,0.7); }
      .gh-metric-value { font-size: 2.2rem; font-weight: 800; color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# 2. LOAD RESOURCES
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

@st.cache_data
def load_evidence():
    """Loads the CSV tables and validation images for Tab 2"""
    evidence = {}
    try:
        evidence['ladder'] = pd.read_csv('reports/tables/final_evaluation_rf.csv')
        evidence['calibration'] = pd.read_csv('reports/tables/calibration_results.csv')
    except FileNotFoundError:
        evidence['ladder'] = None
    return evidence

clinical_model, triage_model = load_models()
evidence_data = load_evidence()

if not clinical_model or not triage_model:
    st.stop()

# ---------------------------------------------------------
# 3. HEADER
# ---------------------------------------------------------
col_title, col_choice = st.columns([3, 1], vertical_alignment="bottom")

with col_title:
    st.title("❤️ GlassHeart CDSS")
    st.markdown("### Heart Failure Mortality Risk Assessment")

with col_choice:
    model_choice = st.segmented_control(
        "Assessment Protocol",
        ["Triage Mode", "Clinical Mode"],
        default="Triage Mode"
    )

# ---------------------------------------------------------
# 4. TABS SYSTEM
# ---------------------------------------------------------
tab_dashboard, tab_evidence = st.tabs(["Clinical Dashboard", "Model Evidence"])

# =========================================================
# TAB 1: THE CLINICAL DASHBOARD
# =========================================================
with tab_dashboard:
    
    # Create Main Layout: Inputs (Left) vs Results (Right)
    col_inputs, col_results = st.columns([5, 2], gap="large")

    # --- INPUT FORM (Left Column) ---
    input_data = {}
    
    # Helpers
    def binary_select(label, key):
        return 1 if st.selectbox(label, ("No", "Yes"), key=key) == "Yes" else 0
    def sex_select(label, key):
        return 1 if st.selectbox(label, ("Female", "Male"), key=key) == "Male" else 0

    with col_inputs:
        st.subheader("Patient Vitals")
        
        if "Clinical" in model_choice:
            active_model = clinical_model
            feature_order = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                             'ejection_fraction', 'high_blood_pressure', 'platelets', 
                             'serum_creatinine', 'serum_sodium', 'sex', 'smoking']

            # Row 1 (3 items)
            c1, c2, c3 = st.columns(3)
            with c1: input_data['age'] = st.number_input("Age", 40, 95, 60)
            with c2: input_data['sex'] = sex_select("Sex", "sex_c")
            with c3: input_data['smoking'] = binary_select("Smoking", "smoke_c")

            # Row 2 (3 items)
            c1, c2, c3 = st.columns(3)
            with c1: input_data['diabetes'] = binary_select("Diabetes", "dia_c")
            with c2: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_c")
            with c3: input_data['anaemia'] = binary_select("Anaemia", "ana_c")

            # Row 3 (3 items)
            c1, c2, c3 = st.columns(3)
            with c1: input_data['serum_sodium'] = st.number_input("Sodium (mEq/L)", 110, 150, 137)
            with c2: input_data['platelets'] = st.number_input("Platelets", 25000.0, 850000.0, 263000.0)
            with c3: input_data['serum_creatinine'] = st.number_input("Creatinine (mg/dL)", 0.5, 9.5, 1.1)

            # Row 4 (2 items)
            c1, c2, c3 = st.columns(3)
            with c1: input_data['ejection_fraction'] = st.number_input("EF (%)", 10, 80, 38)
            with c2: input_data['creatinine_phosphokinase'] = st.number_input("CPK (mcg/L)", 23, 7861, 582)

        else:
            active_model = triage_model
            feature_order = ['age', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

            # Row 1 (2 items)
            c1, c2 = st.columns(2)
            with c1: input_data['age'] = st.number_input("Age", 40, 95, 60)
            with c2: input_data['sex'] = sex_select("Sex", "sex_t")

            # Row 2 (2 items)
            c1, c2 = st.columns(2)
            with c1: input_data['smoking'] = binary_select("Smoking", "smoke_t")
            with c2: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_t")

            # Row 3 (2 items)
            c1, c2 = st.columns(2)
            with c1: input_data['diabetes'] = binary_select("Diabetes", "dia_t")
            with c2: input_data['anaemia'] = binary_select("Anaemia", "ana_t")


    # --- PREDICTION & RESULTS (Right Column) ---
    with col_results:
        # Spacer to push button down slightly to align with inputs
        st.markdown("###") 
        
        calc_trigger = st.button("Calculate Risk Profile", type="primary", use_container_width=True)
        
        if calc_trigger:
            # 1. Prepare Data
            df_input = pd.DataFrame([input_data])[feature_order]
            
            # 2. Predict Risk
            risk_prob = active_model.predict_proba(df_input)[0][1]
            
            # 3. Display Result in Custom Card
            risk_color = "#ff4b4b" if risk_prob > 0.5 else "#09ab3b"
            risk_label = "HIGH RISK" if risk_prob > 0.5 else "LOW RISK"
            
            st.markdown(f"""
            <div class="gh-card">
                <div class="gh-metric-label">Mortality Probability</div>
                <div class="gh-metric-value" style="color: {risk_color}">{risk_prob:.1%}</div>
                <div style="margin-top: 10px; font-weight: bold; color: {risk_color}">{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if risk_prob > 0.5:
                st.error("**Protocol Advised:** ICU admission or advanced therapy review recommended.")
            else:
                st.success("**Protocol Advised:** Standard ward monitoring.")
            
            # Store data for SHAP below
            st.session_state['run_shap'] = True
            st.session_state['df_input'] = df_input
            st.session_state['feature_order'] = feature_order
            st.session_state['active_model'] = active_model

    # --- SHAP EXPLANATION (Full Width Below) ---
    if st.session_state.get('run_shap', False):
        st.markdown("---")
        st.subheader("Model Explanation (SHAP)")
        
        with st.spinner("Generating feature impact analysis..."):
            try:
                # Retrieve from session state
                df_in = st.session_state['df_input']
                feats = st.session_state['feature_order']
                model = st.session_state['active_model']

                # Pipeline access
                # Adjust based on your scikit-learn version (estimator vs base_estimator)
                calibrated_clf = model.calibrated_classifiers_[0]
                pipeline = calibrated_clf.estimator if hasattr(calibrated_clf, "estimator") else calibrated_clf.base_estimator
                
                if 'scaler' in pipeline.named_steps:
                    input_scaled = pipeline.named_steps['scaler'].transform(df_in)
                    input_scaled_df = pd.DataFrame(input_scaled, columns=feats)
                else:
                    input_scaled_df = df_in

                model_step = pipeline.named_steps['rf']
                
                explainer = shap.TreeExplainer(model_step)
                shap_values = explainer(input_scaled_df, check_additivity=False)
                
                # Fix visualization data
                sv = shap_values[0, :, 1]
                sv.data = df_in.iloc[0].values
                
                # Plot
                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(sv, max_display=10, show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Could not generate SHAP plot: {e}")

# =========================================================
# TAB 2: MODEL EVIDENCE
# =========================================================
with tab_evidence:
    st.header("Statistical Validation & Rigor")
    st.markdown("Likelihood Ratio Tests verifying incremental value of features.")
    
    if evidence_data['ladder'] is not None:
        def highlight_sig(s):
            return ['background-color: #12b207' if float(v) < 0.05 else '' for v in s]
        
        st.dataframe(
            evidence_data['ladder'].style.apply(highlight_sig, subset=['LR_p_value']),
            use_container_width=True
        )
    else:
        st.warning("Validation data not found.")

    st.divider()
    
    st.subheader("Calibration")
    try:
        st.image("reports/figures/calibration_curves_calibrated.png", use_container_width=True)
    except:
        st.write("Image not found")
            
    st.divider()

    st.subheader("Global Importance")
    try:
        st.image("reports/figures/numerical_feature_analysis.png", use_container_width=True)
    except:
        st.write("Image not found")