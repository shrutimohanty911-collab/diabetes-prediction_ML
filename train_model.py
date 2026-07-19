"""
Diabetes Prediction - Model Training
Trains Logistic Regression, Random Forest, XGBoost, and a Voting Ensemble
on the PIMA Indians Diabetes Dataset, tunes hyperparameters, and saves the
best model + scaler + SHAP explainer for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import shap

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

df = pd.read_csv(DATA_URL, names=COLUMNS)
print("Loaded dataset:", df.shape)

# ---------------------------------------------------------------------------
# 2. Fix hidden missing values
# In this dataset, 0 is not a valid physiological value for these columns,
# it actually represents missing data. Most tutorials skip this step entirely.
# ---------------------------------------------------------------------------
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in ZERO_AS_MISSING:
    df[col] = df[col].replace(0, np.nan)

# Impute with median, grouped by Outcome so we don't leak class signal flatly
for col in ZERO_AS_MISSING:
    df[col] = df.groupby("Outcome")[col].transform(lambda x: x.fillna(x.median()))

print("Missing values handled. Remaining NaNs:", df.isna().sum().sum())

# ---------------------------------------------------------------------------
# 3. Train/test split + scaling
# ---------------------------------------------------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# 4. Models + hyperparameter tuning
# ---------------------------------------------------------------------------
results = {}

# --- Logistic Regression ---
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid={"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]},
    cv=cv, scoring="roc_auc"
)
lr_grid.fit(X_train_scaled, y_train)
results["Logistic Regression"] = lr_grid.best_estimator_

# --- Random Forest ---
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [1, 2, 4]
    },
    cv=cv, scoring="roc_auc"
)
rf_grid.fit(X_train_scaled, y_train)
results["Random Forest"] = rf_grid.best_estimator_

# --- XGBoost ---
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid={
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1]
    },
    cv=cv, scoring="roc_auc"
)
xgb_grid.fit(X_train_scaled, y_train)
results["XGBoost"] = xgb_grid.best_estimator_

# --- SVM (kept for comparison against the "classic" tutorial result) ---
svm_grid = GridSearchCV(
    SVC(probability=True),
    param_grid={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    cv=cv, scoring="roc_auc"
)
svm_grid.fit(X_train_scaled, y_train)
results["SVM"] = svm_grid.best_estimator_

# --- Voting Ensemble (soft voting on the 3 strongest models) ---
ensemble = VotingClassifier(
    estimators=[
        ("lr", results["Logistic Regression"]),
        ("rf", results["Random Forest"]),
        ("xgb", results["XGBoost"]),
    ],
    voting="soft"
)
ensemble.fit(X_train_scaled, y_train)
results["Voting Ensemble"] = ensemble

# ---------------------------------------------------------------------------
# 5. Evaluate all models, pick the best by test ROC-AUC
# ---------------------------------------------------------------------------
print("\nModel Comparison (Test Set)")
print("-" * 45)
scoreboard = {}
for name, model in results.items():
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    scoreboard[name] = {"accuracy": acc, "roc_auc": auc}
    print(f"{name:20s}  Accuracy: {acc*100:.2f}%   ROC-AUC: {auc:.4f}")

best_name = max(scoreboard, key=lambda k: scoreboard[k]["roc_auc"])
best_model = results[best_name]
print(f"\nBest model: {best_name} (ROC-AUC: {scoreboard[best_name]['roc_auc']:.4f})")
print("\nClassification report for best model:")
print(classification_report(y_test, best_model.predict(X_test_scaled)))

# ---------------------------------------------------------------------------
# 6. Save model, scaler, and a SHAP explainer for the best model
# ---------------------------------------------------------------------------
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

# Use TreeExplainer if the best model supports it (RF/XGB), else KernelExplainer
try:
    explainer = shap.TreeExplainer(best_model)
except Exception:
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train_scaled[:100])

joblib.dump(explainer, "shap_explainer.pkl")

with open("model_report.txt", "w") as f:
    f.write("Diabetes Prediction - Model Comparison\n")
    f.write("=" * 45 + "\n")
    for name, s in scoreboard.items():
        f.write(f"{name:20s}  Accuracy: {s['accuracy']*100:.2f}%   ROC-AUC: {s['roc_auc']:.4f}\n")
    f.write(f"\nBest model: {best_name}\n")

print("\nSaved: diabetes_model.pkl, scaler.pkl, feature_names.pkl, shap_explainer.pkl, model_report.txt")
