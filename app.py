import streamlit as st
import pandas as pd
import joblib
import altair as alt
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
        padding: 24px;
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
      
      /* --- SEGMENTED CONTROL STYLING (FIXED) --- */
      /* Target the container and the text inside */
      [data-testid="stSegmentedControl"] {
        padding: 6px !important;
      }
      [data-testid="stSegmentedControl"] button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding-top: 12px !important;
        padding-bottom: 12px !important;
      }
      
      /* Custom Metric Style for Card */
      .gh-metric-label { font-size: 1.0rem; color: rgba(255,255,255,0.7); margin-top: -8px; margin-bottom: 8px; }
      .gh-metric-value { font-size: 2.2rem; font-weight: 800; color: #ffffff; line-height: 1.1; }
      .gh-metric-tag   { font-size: 1.4rem; font-weight: 700; margin-left: 80px; align-self: flex-end; }
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
        st.error("Models not found! Please run 'python train_models.py' locally first.")
        return None, None

@st.cache_data
def load_evidence():
    """Loads the CSV tables and validation images for Tab 2"""
    evidence = {}
    try:
        evidence['ladder'] = pd.read_csv('reports/tables/ladder_validation_rf.csv')
        evidence['final_evaluation'] = pd.read_csv('reports/tables/final_evaluation_rf.csv')
        evidence['calibration'] = pd.read_csv('reports/tables/calibration_results.csv')
        evidence['X_test'] = joblib.load('artifacts/splits/x_test.joblib')
        evidence['y_test'] = joblib.load('artifacts/splits/y_test.joblib')
    except Exception as e:
        st.error(f"Error loading evidence: {e}")
        evidence['ladder'] = None
    return evidence

clinical_model, triage_model = load_models()
evidence_data = load_evidence()

if not clinical_model or not triage_model:
    st.stop()

# ---------------------------------------------------------
# 3. HELPER: RESET STATE
# ---------------------------------------------------------
def reset_results():
    """Clears the SHAP result and prediction when inputs change"""
    st.session_state['run_shap'] = False
    st.session_state['prediction_made'] = False

# Initialize state if not present
if 'run_shap' not in st.session_state:
    st.session_state['run_shap'] = False
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False

# ---------------------------------------------------------
# 4. HEADER
# ---------------------------------------------------------
col_title, col_choice = st.columns([5, 2], vertical_alignment="bottom")

with col_title:
    st.title("❤️ GlassHeart CDSS")
    st.markdown("### Heart Failure Mortality Risk Assessment")

with col_choice:
    model_choice = st.segmented_control(
        "Assessment Protocol",
        ["Triage Mode", "Clinical Mode"],
        default="Triage Mode",
        on_change=reset_results # RESET ON MODE CHANGE
    )

# ---------------------------------------------------------
# 5. TABS SYSTEM
# ---------------------------------------------------------
tab_dashboard, tab_evidence = st.tabs(["Clinical Dashboard", "Model Evidence"])

# =========================================================
# TAB 1: THE CLINICAL DASHBOARD
# =========================================================
with tab_dashboard:
    # Create Main Layout: Inputs (Left) vs Results (Right)
    col_inputs, col_results = st.columns([6, 2], gap="large")

    # --- INPUT FORM (Left Column) ---
    input_data = {}
    
    # Helpers with on_change callback attached
    def binary_select(label, key):
        return 1 if st.selectbox(label, ("No", "Yes"), key=key, on_change=reset_results) == "Yes" else 0
    
    def sex_select(label, key):
        return 1 if st.selectbox(label, ("Female", "Male"), key=key, on_change=reset_results) == "Male" else 0

    def num_input(label, min_v, max_v, default_v, key, step=None):
        return st.number_input(label, min_v, max_v, default_v, step=step, key=key, on_change=reset_results)

    with col_inputs:
        st.subheader("Patient Vitals")
        
        if "Clinical" in model_choice:
            active_model = clinical_model
            feature_order = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                             'ejection_fraction', 'high_blood_pressure', 'platelets', 
                             'serum_creatinine', 'serum_sodium', 'sex', 'smoking']

            # Row 1 (3 items)
            c1, c2, c3 = st.columns(3)
            with c1: input_data['age'] = num_input("Age", 40, 95, 60, "age_c")
            with c2: input_data['sex'] = sex_select("Sex", "sex_c")
            with c3: input_data['smoking'] = binary_select("Smoking", "smoke_c")

            # Row 2 (3 items)
            c1, c2, c3 = st.columns(3)
            with c1: input_data['diabetes'] = binary_select("Diabetes", "dia_c")
            with c2: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_c")
            with c3: input_data['anaemia'] = binary_select("Anaemia", "ana_c")

            # Row 3 (3 items)
            c1, c2, c3 = st.columns(3)
            with c1: input_data['ejection_fraction'] = num_input("EF (%)", 10, 80, 38, "ef_c")
            with c2: input_data['platelets'] = num_input("Platelets", 25000.0, 850000.0, 263000.0, "pl_c", step=1000.0)
            with c3: input_data['serum_creatinine'] = num_input("Creatinine (mg/dL)", 0.5, 9.5, 1.1, "cr_c", step=0.1)

            # Row 4 (2 items - Full Width)
            c1, c2 = st.columns(2)
            with c1: input_data['serum_sodium'] = num_input("Sodium (mEq/L)", 110, 150, 137, "na_c")
            with c2: input_data['creatinine_phosphokinase'] = num_input("CPK (mcg/L)", 23, 7861, 582, "cpk_c", step=10)

        else:
            active_model = triage_model
            feature_order = ['age', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

            # Row 1 (3 items)
            c1, c2 = st.columns(2)
            with c1: input_data['age'] = num_input("Age", 40, 95, 60, "age_t")
            with c2: input_data['sex'] = sex_select("Sex", "sex_t")

            # Row 2 (3 items)
            c1, c2 = st.columns(2)
            with c1: input_data['smoking'] = binary_select("Smoking", "smoke_t")
            with c2: input_data['high_blood_pressure'] = binary_select("Hypertension", "hbp_t")

            # Row 3 (3 items)
            c1, c2 = st.columns(2)
            with c1: input_data['diabetes'] = binary_select("Diabetes", "dia_t")
            with c2: input_data['anaemia'] = binary_select("Anaemia", "ana_t")


    # --- PREDICTION & RESULTS (Right Column) ---
    with col_results:
        st.markdown("###") 
        
        calc_trigger = st.button("Calculate Risk Profile", type="primary", use_container_width=True)
        
        if calc_trigger:
            st.session_state['prediction_made'] = True
            
        # Only show results if Calculate was clicked AND state hasn't been reset
        if st.session_state['prediction_made']:
            # 1. Prepare Data
            df_input = pd.DataFrame([input_data])[feature_order]
            
            # 2. Predict Risk
            risk_prob = active_model.predict_proba(df_input)[0][1]
            
            # 3. Display Result in Custom Card
            # FIX: Label is now to the RIGHT of value using flexbox
            risk_color = "#ff4b4b" if risk_prob > 0.5 else "#09ab3b"
            risk_label = "HIGH RISK" if risk_prob > 0.5 else "LOW RISK"
            
            st.markdown(f"""
            <div class="gh-card">
                <div class="gh-metric-label">Mortality Probability</div>
                <div style="display: flex; flex-direction: row; align-items: baseline;">
                    <div class="gh-metric-value" style="color: {risk_color}">{risk_prob:.1%}</div>
                    <div class="gh-metric-tag" style="color: {risk_color}">{risk_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if risk_prob > 0.5:
                st.error("**Protocol Advised:** ICU admission or advanced therapy review recommended.")
            else:
                st.success("**Protocol Advised:** Standard ward monitoring.")
            
            # Store data for SHAP
            st.session_state['run_shap'] = True
            st.session_state['df_input'] = df_input
            st.session_state['feature_order'] = feature_order
            st.session_state['active_model'] = active_model

    # --- SHAP EXPLANATION (Full Width Below) ---
    # Only shows if run_shap is TRUE (which is cleared on any input change)
    if st.session_state.get('run_shap', False):
        st.markdown("---")
        
        # Use Expander for SHAP 
        with st.expander("▶ Click to understand why this prediction was made (SHAP Analysis)", expanded=False):
            with st.spinner("Generating feature impact analysis..."):
                try:
                    df_in = st.session_state['df_input']
                    feats = st.session_state['feature_order']
                    model = st.session_state['active_model']

                    # Pipeline access
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
                    
                    sv = shap_values[0, :, 1]
                    sv.data = df_in.iloc[0].values
                    
                    plt.style.use("dark_background")
                    fig, ax = plt.subplots(figsize=(2, 5))
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
                    st.pyplot(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {e}")

# =========================================================
# TAB 2: MODEL EVIDENCE
# =========================================================
with tab_evidence:
    if evidence_data['ladder'] is None:
        st.error("Evidence data not found. Please run notebooks/Model_Building_and_Evaluation.ipynb to generate artifacts.")
    else:
        # --- SECTION 1: THE VERDICT ---
        st.markdown("### 1. The Verdict: Hypothesis Testing")
        st.markdown('''
            - The primary goal of this project was not to build a **Clinical Decision Support System** but rather was to validate the following hypothesis:\n
            '**"Lower ejection fraction and higher serum creatinine are dominant predictors of mortality; serum sodium and age nonlinearity provide independent incremental value"**'
            - The project was an exploration of this hypothesis. The _Triage_ and _Clinical_ models were developed during the deployment of this dashboard.
            To ensure a rigorous validation process, a series of statistical tests and methodologies were employed. They are explained as follows: '''
        )


        # --- SECTION 2: AUC STEP-UP CHART (MATPLOTLIB) ---
        st.markdown("### 2. Feature Ladder (Incremental Value)")
        st.markdown('''
        - A mutual information test was performed to capture linear and non-linear dependencies between the features and the target variable.
        - The features 'smoking', 'high_blood_pressure', 'diabetes', 'anaemia' were found to have almost no relation to the target variable and were dropped.
        - Then the remaining features were compiled into the following feature sets in the form of a feature ladder:

        ```
        baseline_fs = ['age', 'sex', 'creatinine_phosphokinase', 'platelets']
        ef_fs = baseline_fs + ['ejection_fraction']
        creatinine_fs = baseline_fs + ['serum_creatinine']
        ef_creatinine_fs = baseline_fs + ['ejection_fraction'] + ['serum_creatinine']
        ef_creatinine_sodium_fs = ef_creatinine_fs + ['serum_sodium']
        ef_creatinine_age_squared_fs = ef_creatinine_fs + ['age_squared']
        ef_creatinine_sodium_age_squared_fs =  ef_creatinine_fs + ['serum_sodium'] + ['age_squared']
        ```
        
        - The ROC AUC score was calculated for each of the feature sets and is shown in the feature ladder chart below. 
        - A clear bump in the performance of the model is seen when adding 'ejection_fraction' to the feature set. 
        ''')
        
        ladder_df = evidence_data['ladder'].copy()
        
        # Prepare Data for Plotting
        # We want to plot the 'extended_fs' vs 'DeLong_extended_AUC'
        # Color based on 'LR_p_value' < 0.05
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot
        bars = ax.bar(ladder_df['feature_set'], ladder_df['roc_auc'], color='skyblue', edgecolor='white', linewidth=0.5)
        
        # Styling
        plt.style.use("dark_background")
        ax.set_ylim(0.45, 0.90)
        ax.set_ylabel("ROC AUC Score", color="#e5e7eb", fontsize=10)
        ax.set_xlabel("Feature Set", color="#e5e7eb", fontsize=10)
        ax.set_title("Feature Ladder Performance", color="white", fontsize=12, pad=15)
        
        # Axis colors
        ax.tick_params(axis='x', colors='#d1d5db', rotation=0)
        ax.tick_params(axis='y', colors='#d1d5db')
        
        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#4b5563')
        ax.spines['bottom'].set_color('#4b5563')
        
        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')

        # Annotations
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

        st.pyplot(fig, use_container_width=True)

        st.divider()

        # --- SECTION 3: INTERACTIVE THRESHOLD ---
        st.markdown("### 3. Optimal Threshold Selection")
        st.markdown("*\"Now that we have well-calibrated models, we must select the optimal classification threshold. We will use the **Youden's J statistic** from the test set ROC curve to find the threshold that provides the best balance between sensitivity and specificity.\"*")
        
        if 'X_test' in evidence_data and 'y_test' in evidence_data:
            X_test = evidence_data['X_test']
            y_test = evidence_data['y_test']
            
            cols_needed = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                             'ejection_fraction', 'high_blood_pressure', 'platelets', 
                             'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
            try:
                probs = clinical_model.predict_proba(X_test[cols_needed])[:, 1]
                
                col_slider, col_metrics = st.columns([1, 2])
                
                with col_slider:
                    st.write("**Adjust Decision Threshold:**")
                    threshold = st.slider("Threshold", 0.0, 1.0, 0.24, key="thresh_slider",
                                        help="Lower thresholds catch more cases (higher sensitivity) but increase false alarms.")
                    y_pred = (probs >= threshold).astype(int)
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    
                    st.write("<b>Confusion Matrix (Test Set)</b>", unsafe_allow_html=True)
                    st.write(pd.DataFrame(cm, 
                                        columns=['Pred: Survived', 'Pred: Died'], 
                                        index=['Actual: Survived', 'Actual: Died']))

                with col_metrics:
                    acc = accuracy_score(y_test, y_pred)
                    sens = recall_score(y_test, y_pred)
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    youden = sens + spec - 1
                    
                    st.markdown("#### Performance Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sensitivity", f"{sens:.1%}", help="True Positive Rate (Recall)")
                    m2.metric("Specificity", f"{spec:.1%}", help="True Negative Rate")
                    m3.metric("Accuracy", f"{acc:.1%}")
                    m4.metric("Youden's J", f"{youden:.2f}")
                    
                    if sens > 0.9:
                        st.caption("✅ High Sensitivity: Good for screening (few missed cases).")
                    elif spec > 0.9:
                        st.caption("✅ High Specificity: Good for confirming diagnosis (few false alarms).")
                        
            except Exception as e:
                st.warning(f"Could not compute interactive metrics: {e}")
        else:
            st.info("Test data not loaded.")

        st.divider()

        # --- SECTION 4: CALIBRATION ---
        col_calib, col_shap = st.columns(2)
        
        with col_calib:
            st.markdown("### 4. Calibration Analysis")
            st.markdown("*\"Predictive accuracy (discrimination) is not enough for a clinical risk model; we also need models that produce reliable probabilities (calibration). A well-calibrated model's predicted probability of 30% should correspond to an event actually occurring ~30% of the time.\"*")
            st.markdown("We used **Isotonic Regression** to align predicted probabilities. A lower Brier Score indicates better calibration.")
            
            try:
                st.image("reports/figures/calibration_curves_calibrated.png", use_container_width=True)
                if evidence_data['calibration'] is not None:
                    cal_df = evidence_data['calibration']
                    rf_cal = cal_df[cal_df['model'] == 'RandomForestClassifier'].iloc[0]
                    st.metric("Brier Score Improvement (RF)", 
                              f"{rf_cal['brier_orig']:.3f} → {rf_cal['brier_cal']:.3f}",
                              delta=f"{rf_cal['brier_orig'] - rf_cal['brier_cal']:.3f} (Lower is better)",
                              delta_color="inverse")
            except:
                st.write("Image not found")
                
        with col_shap:
            st.markdown("### 5. Global Feature Importance")
            st.markdown("Derived from SHAP Summary analysis on the validation cohort. This confirms that **Serum Creatinine** and **Ejection Fraction** are the most impactful features globally, aligning with clinical literature.")
            try:
                st.image("reports/figures/numerical_feature_analysis.png", use_container_width=True)
            except:
                st.write("Image not found")