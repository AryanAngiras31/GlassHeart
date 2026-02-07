# heart_failure_predict.py
# End-to-end template for heart failure risk prediction (tabular clinical data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib

# -------------------------
# User configuration
# -------------------------
CSV_PATH = "heart_failure_clinical_records.csv"   # <-- replace with your path
TARGET_COL = "DEATH_EVENT"                         # <-- replace if needed
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5                                       # for cross-validation
EXPORT_MODEL_PATH = "hf_best_model.joblib"         # set to None to skip export

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

# Basic data audit
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values per column:\n", df.isna().sum())
print("\nTarget value counts:\n", df[TARGET_COL].value_counts(dropna=False))

# Separate features/target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# Identify numeric vs. categorical/boolean
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols: List[str] = [c for c in X.columns if c not in num_cols]

print("\nNumeric features:", num_cols)
print("Categorical features:", cat_cols if cat_cols else "None detected")

# -------------------------
# Train/valid split (stratified)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -------------------------
# Preprocessing
# -------------------------
preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
    ],
    remainder="passthrough"
)

# -------------------------
# Models
# -------------------------
logreg = Pipeline(steps=[
    ("pre", preproc),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
        random_state=RANDOM_STATE
    ))
])

rf = Pipeline(steps=[
    ("pre", preproc),
    ("clf", RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])

models = {
    "LogisticRegression": logreg,
    "RandomForest": rf
}

# -------------------------
# Cross-validation (ROC-AUC & PR-AUC)
# -------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

print("\nCross-validation results:")
cv_results: Dict[str, Dict[str, float]] = {}
for name, est in models.items():
    scores = cross_validate(
    est, 
    X_train, 
    y_train, 
    scoring=scoring, 
    cv=cv, 
    n_jobs=1,   # ✅ Force single-threaded on Mac to avoid loky bug
    return_train_score=False
)

    
    cv_results[name] = {
        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "pr_auc_mean": scores["test_pr_auc"].mean(),
        "pr_auc_std": scores["test_pr_auc"].std(),
    }
    print(f"- {name}: ROC-AUC {cv_results[name]['roc_auc_mean']:.3f} ± {cv_results[name]['roc_auc_std']:.3f} | "
          f"PR-AUC {cv_results[name]['pr_auc_mean']:.3f} ± {cv_results[name]['pr_auc_std']:.3f}")

# Pick best by mean ROC-AUC
best_model_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
best_estimator = models[best_model_name]
print(f"\nSelected model: {best_model_name}")

# -------------------------
# Fit on full training set
# -------------------------
best_estimator.fit(X_train, y_train)

# -------------------------
# Evaluate on test set
# -------------------------
y_pred = best_estimator.predict(X_test)
y_proba = best_estimator.predict_proba(X_test)[:, 1]

print("\nClassification report:\n", classification_report(y_test, y_pred))
print("ROC-AUC (test):", roc_auc_score(y_test, y_proba))
print("PR-AUC (test):", average_precision_score(y_test, y_proba))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.show()

# PR Curve
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.show()

# -------------------------
# Export model
# -------------------------
if EXPORT_MODEL_PATH:
    joblib.dump(best_estimator, EXPORT_MODEL_PATH)
    print(f"\nModel exported to {EXPORT_MODEL_PATH}")
