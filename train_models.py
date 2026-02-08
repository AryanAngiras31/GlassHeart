import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from notebooks.utils import final_evaluation

def train_and_save():
    print("Loading Data...")
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    target = 'DEATH_EVENT'
    y = df[target]
    X = df
    X = X.drop(columns=[target, 'time'])

    # --- DEFINING FEATURE SETS ---
    # 1. Triage Features (As requested)
    triage_features = ['age', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    
    # 2. Clinical Features (All except time and target)
    clinical_features = [c for c in df.columns if c not in [target, 'time']]

    # Create same split for both models
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- HELPER: TUNING & CALIBRATION ---
    def build_tuned_calibrated_model(X, y, name):
        print(f"\n Tuning & Calibrating {name} Model...")
        
        # 1. Define Base Pipeline
        # We tune the scaler and the classifier
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ])

        # 2. Define Hyperparameter Grid
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [3, 5, 10, None],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__class_weight': ['balanced', None]
        }

        # 3. Grid Search (Optimize for Brier Score, which measures probability accuracy)
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='neg_brier_score', 
            n_jobs=-1
        )
        grid.fit(X, y)
        
        best_base_model = grid.best_estimator_
        print(f"Best Params: {grid.best_params_}")

        # 4. Calibration (Isotonic Regression)
        # We wrap the best tuned model in a CalibratedClassifier
        calibrated_model = CalibratedClassifierCV(
            estimator=best_base_model,
            method='isotonic',
            cv=5
        )
        calibrated_model.fit(X, y)
        
        return calibrated_model

    def test(model, X_test, y_test):

        # Get hard predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluate
        evaluation = final_evaluation(y_test, y_prob, y_pred)

        # Display results
        final_test_df = pd.DataFrame([evaluation])
        print(final_test_df.to_string(index=False))

    # --- TRAIN & TEST TRIAGE MODEL ---

    # Derive triage subsets from the same split
    X_train_triage = X_train_full[triage_features]
    X_test_triage = X_test_full[triage_features]

    # Save tuned and calibrated model
    triage_model = build_tuned_calibrated_model(X_train_triage, y_train, "Triage")
    joblib.dump(triage_model, 'models/triage_model_calibrated.joblib')
    print("Saved: models/triage_model_calibrated.joblib")

    # Test the model
    print("\n--- Testing Triage Model ---")
    test(triage_model, X_test_triage, y_test)

    # --- TRAIN & TEST CLINICAL MODEL ---

    # Derive clinical subsets from the same split
    X_train_clinical = X_train_full[clinical_features]
    X_test_clinical = X_test_full[clinical_features]

    # Save tuned and calibrated model
    clinical_model = build_tuned_calibrated_model(X_train_clinical, y_train, "Clinical")
    joblib.dump(clinical_model, 'models/clinical_model_calibrated.joblib')
    print("Saved: models/clinical_model_calibrated.joblib")

    # Test the model
    print("\n--- Testing Clinical Model ---")
    test(clinical_model, X_test_clinical, y_test)
    

if __name__ == "__main__":
    train_and_save()