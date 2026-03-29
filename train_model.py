"""
ChurnGuard - Model Training Script
Trains a RandomForest pipeline on the Telco Customer Churn dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "models"
ASSETS_DIR = "assets"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_pipeline.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── 1. Load & Clean ───────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)

# TotalCharges has some blanks → convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID – not a feature
df.drop("customerID", axis=1, inplace=True)

# ── 2. Feature / Target Split ─────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify column types
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features  : {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# ── 3. Build Pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

# ── 4. Train / Test ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model …")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*50}")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print(f"{'='*50}")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 5. Save Pipeline ──────────────────────────────────────────────────
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ── 6. Save Assets for Dashboard ──────────────────────────────────────
# Feature importance
rf = pipeline.named_steps["classifier"]
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
importance_df.to_csv(os.path.join(ASSETS_DIR, "feature_importance.csv"), index=False)

# Churn distribution
churn_dist = df["Churn"].value_counts().reset_index()
churn_dist.columns = ["Churn", "Count"]
churn_dist["Churn"] = churn_dist["Churn"].map({0: "No", 1: "Yes"})
churn_dist.to_csv(os.path.join(ASSETS_DIR, "churn_distribution.csv"), index=False)

# Save column metadata for the app
meta = {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}
joblib.dump(meta, os.path.join(ASSETS_DIR, "column_meta.pkl"))

# Save model metrics
metrics = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)}
joblib.dump(metrics, os.path.join(ASSETS_DIR, "metrics.pkl"))

print(f"Assets saved to {ASSETS_DIR}/")
print("Done! You can now run: streamlit run app.py")
