# =============================================================
#  Retail Analytics Intelligence Platform — Model Training
#  Run: python train_model.py
#
#  What this script does:
#   1. Loads & preprocesses the dataset
#   2. Trains an XGBoost Churn Prediction model
#   3. Evaluates and prints metrics
#   4. Saves: churn_model_xgb.pkl  &  churn_imputer.pkl
#      (these are required by app.py → Churn Predictor page)
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
import xgboost as xgb

# ── 1. Load data ──────────────────────────────────────────────
print("=" * 55)
print("  Retail Analytics — Churn Model Training")
print("=" * 55)

DATA_PATH = "supermarket_large_dataset.csv"
print(f"\n[1/5] Loading dataset from '{DATA_PATH}' …")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"      Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Feature engineering ────────────────────────────────────
print("\n[2/5] Engineering features …")

df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
df["transaction_date"]     = pd.to_datetime(df["transaction_date"],     errors="coerce")
df["trans_weekday"]        = df["transaction_datetime"].dt.dayofweek
df["has_discount"]         = (df["discount_percent"] > 0).astype(int)
df["is_weekend_flag"]      = df["trans_weekday"].isin([5, 6]).astype(int)

# Binary churn target (churn_risk > 0.5 → churned)
df["churn_label"] = (df["customer_churn_risk"] > 0.5).astype(int)
print(f"      Churn class distribution:\n{df['churn_label'].value_counts().to_string()}")

# ── 3. Select features ────────────────────────────────────────
print("\n[3/5] Selecting & encoding features …")

NUMERIC_FEATURES = [
    "customer_age",
    "customer_tenure_months",
    "average_monthly_visits",
    "average_transaction_value",
    "customer_lifetime_value",
    "customer_retention_score",
    "customer_engagement_score",
    "loyalty_points_balance",
    "loyalty_points_earned",
    "loyalty_points_redeemed",
    "membership_duration_months",
    "number_of_children",
    "household_size",
    "has_discount",
    "is_weekend_flag",
]

CATEGORICAL_FEATURES = [
    "customer_gender",
    "customer_segment",
    "customer_membership_status",
    "membership_tier",
    "customer_education",
    "customer_occupation",
]

# Label-encode categoricals
le = LabelEncoder()
for col in CATEGORICAL_FEATURES:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

ALL_FEATURES = [
    c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns
]
TARGET = "churn_label"

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

# Impute missing with median
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=ALL_FEATURES)

print(f"      Feature matrix : {X_imputed.shape}")
print(f"      Target balance : {y.value_counts().to_dict()}")

# ── 4. Train/test split & model training ──────────────────────
print("\n[4/5] Training XGBoost Classifier …")

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.20, random_state=42, stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos = neg / pos if pos > 0 else 1.0

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("      Training complete.")

# ── 5. Evaluate ───────────────────────────────────────────────
print("\n[5/5] Evaluating on test set …")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f"\n  ROC-AUC Score : {auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(f"         Predicted No  Predicted Yes")
print(f"  Actual No   {cm[0,0]:>6}        {cm[0,1]:>6}")
print(f"  Actual Yes  {cm[1,0]:>6}        {cm[1,1]:>6}")

# Top 10 feature importances
feat_imp = pd.Series(model.feature_importances_, index=ALL_FEATURES)
feat_imp = feat_imp.sort_values(ascending=False).head(10)
print("\n  Top 10 Feature Importances:")
for feat, score in feat_imp.items():
    bar = "█" * int(score * 40)
    print(f"  {feat:<40} {score:.4f}  {bar}")

# ── 6. Save artefacts ────────────────────────────────────────
MODEL_PATH   = "churn_model_xgb.pkl"
IMPUTER_PATH = "churn_imputer.pkl"

joblib.dump(model,   MODEL_PATH)
joblib.dump(imputer, IMPUTER_PATH)

print("\n" + "=" * 55)
print(f"  ✅ Model saved   → {MODEL_PATH}")
print(f"  ✅ Imputer saved → {IMPUTER_PATH}")
print("=" * 55)
print("\n  Now launch the dashboard:")
print("  $ streamlit run app.py\n")
