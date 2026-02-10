import streamlit as st
import pandas as pd
import joblib

import shap
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="GlassHeart | Clinical Decision Support",
    page_icon="❤️",
    layout="wide"
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
st.title("❤️ GlassHeart CDSS")
st.markdown("### Heart Failure Mortality Risk Assessment")

# ---------------------------------------------------------
# 4. TABS SYSTEM
# ---------------------------------------------------------
tab_dashboard, tab_evidence = st.tabs(["Clinical Dashboard", "Model Evidence"])

# =========================================================
# TAB 1: THE CLINICAL DASHBOARD
# =========================================================
with tab_dashboard:
    col_choice, col_input = st.columns([1, 3])
    
    with col_choice:
        model_choice = st.segmented_control(
            "Protocol",
            ["Triage Mode", "Clinical Mode"],
            default="Triage Mode"
        )

    st.divider()
    
    # --- INPUT FORM ---
    input_data = {}
    
    # Helpers
    def binary_select(label, key):
        return 1 if st.selectbox(label, ("No", "Yes"), key=key) == "Yes" else 0
    def sex_select(label, key):
        return 1 if st.selectbox(label, ("Female", "Male"), key=key) == "Male" else 0

    with col_input:
        if "Clinical" in model_choice:
            # Clinical Inputs (11 Features)
            active_model = clinical_model
            feature_order = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                             'ejection_fraction', 'high_blood_pressure', 'platelets', 
                             'serum_creatinine', 'serum_sodium', 'sex', 'smoking']

            c1, c2, c3, c4 = st.columns(4)
            with c1: input_data['age'] = st.number_input("Age", 40, 95, 60)
            with c2: input_data['sex'] = sex_select("Sex", "sex_c")
            with c3: input_data['smoking'] = binary_select("Smoking", "smoke_c")
            with c4: input_data['diabetes'] = binary_select("Diabetes", "dia_c")

            c1, c2, c3, c4 = st.columns(4)
            with c1: input_data['ejection_fraction'] = st.number_input("EF (%)", 10, 80, 38)
            with c2: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_c")
            with c3: input_data['anaemia'] = binary_select("Anaemia", "ana_c")
            with c4: input_data['platelets'] = st.number_input("Platelets", 25000.0, 850000.0, 263000.0)

            c1, c2, c3 = st.columns(3)
            with c1: input_data['serum_creatinine'] = st.number_input("Creatinine (mg/dL)", 0.5, 9.5, 1.1)
            with c2: input_data['serum_sodium'] = st.number_input("Sodium (mEq/L)", 110, 150, 137)
            with c3: input_data['creatinine_phosphokinase'] = st.number_input("CPK (mcg/L)", 23, 7861, 582)

        else:
            # Triage Inputs (6 Features)
            active_model = triage_model
            feature_order = ['age', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

            c1, c2, c3 = st.columns(3)
            with c1: input_data['age'] = st.number_input("Age", 40, 95, 60)
            with c2: input_data['sex'] = sex_select("Sex", "sex_t")
            with c3: input_data['smoking'] = binary_select("Smoking", "smoke_t")

            c1, c2, c3 = st.columns(3)
            with c1: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_t")
            with c2: input_data['diabetes'] = binary_select("Diabetes", "dia_t")
            with c3: input_data['anaemia'] = binary_select("Anaemia", "ana_t")

    # --- PREDICTION ---
    st.markdown("###")
    if st.button("Calculate Risk Profile", type="primary", width="stretch"):
        
        # 1. Prepare Data
        df_input = pd.DataFrame([input_data])[feature_order]
        
        # 2. Predict Risk
        risk_prob = active_model.predict_proba(df_input)[0][1]
        
        # 3. Display Main Result
        st.markdown("---")
        c_res, c_msg = st.columns([1, 2])
        
        with c_res:
            st.metric(
                label="30-Day Mortality Risk", 
                value=f"{risk_prob:.1%}",
                delta="High Risk" if risk_prob > 0.5 else "Stable",
                delta_color="inverse"
            )
        
        with c_msg:
            if risk_prob > 0.5:
                st.error("⚠️ **High Risk Protocol Advised**")
                st.write("Patient falls into the upper risk quartile. Recommended action: ICU admission or advanced therapy review.")
            else:
                st.success("✅ **Low Risk Profile**")
                st.write("Patient is stable. Standard ward monitoring recommended.")

        # 4. SHAP EXPLANATION (Drill-Down)
        # This is the 'Why' part
        with st.expander("▶ Click to understand why this prediction was made (SHAP Analysis)"):
            st.caption("Feature contributions pushing risk UP (Red) or DOWN (Blue) from the baseline.")
            
            try:
                # ACCESSING INNER MODEL FOR SHAP
                # Since we wrapped in CalibratedClassifierCV -> Pipeline
                # We need to access the base estimator's steps
                
                # 1. Get the pipeline from the first calibrated classifier
                pipeline = active_model.calibrated_classifiers_[0].estimator
                
                # 2. Transform input data using the pipeline's scaler
                # (SHAP needs the scaled numbers if the model saw scaled numbers)
                if 'scaler' in pipeline.named_steps:
                    input_scaled = pipeline.named_steps['scaler'].transform(df_input)
                    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_order)
                else:
                    input_scaled_df = df_input

                # 3. Get the tree model (Random Forest)
                model_step = pipeline.named_steps['rf'] # or 'model' depending on your training script
                
                # 4. Create Explainer
                explainer = shap.TreeExplainer(model_step)
                shap_values = explainer(input_scaled_df)
                
                sv = shap_values[0, :, 1]          # sample 0, all features, class 1
                sv.data = df_input.iloc[0].values
                
                # 6. Plot
                plt.style.use("dark_background")
                fig, ax = plt.subplots()
                shap.decision_plot(
                    base_value=sv.base_values,
                    shap_values=sv.values,
                    features=sv.data,
                    feature_names=feature_order,
                    show=False
                )
                ax = plt.gca()
                ax.tick_params(axis="y", colors="white")
                ax.tick_params(axis="x", colors="white")
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Could not generate SHAP plot for this model architecture: {e}")

# =========================================================
# TAB 2: MODEL EVIDENCE (Statistical Tests)
# =========================================================
with tab_evidence:
    st.header("Statistical Validation & Rigor")
    st.markdown("""
    This model has been validated using rigorous statistical testing to ensure 
    features add incremental value and probabilities are calibrated.
    """)

    st.subheader("1. Feature Ladder (Hypothesis Testing)")
    st.write("Likelihood Ratio Tests verifying incremental value of features.")
    
    if evidence_data['ladder'] is not None:
        # Highlight significant p-values
        def highlight_sig(s):
            return ['background-color: #12b207' if v < 0.05 else '' for v in s]
        st.dataframe(
            evidence_data['ladder'].style.apply(highlight_sig, subset=['LR_p_value']),
            width = 'stretch'
        )
        st.caption("Green rows indicate features that added statistically significant predictive power (p < 0.05).")
    else:
        st.warning("Ladder validation CSV not found in reports/tables/")

    st.divider()

    st.subheader("2. Model Calibration")
    st.write("Comparison of predicted probabilities vs. observed mortality.")
    
    try:
        st.image("reports/figures/calibration_curves_calibrated.png", caption="Isotonic Regression Calibration", width='content')
    except:
        st.warning("Calibration image not found in reports/figures/")

    st.divider()
    
    st.subheader("3. Global Feature Importance")
    st.markdown("Derived from SHAP Summary analysis on the validation cohort.")

    try:
        st.image("reports/figures/numerical_feature_analysis.png", caption="Feature Distributions", width='content')
    except:
        st.info("Feature analysis plot not available.")