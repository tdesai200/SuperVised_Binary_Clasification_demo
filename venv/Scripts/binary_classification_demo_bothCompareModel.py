# ================================================
# Gujarati Catering: XGBoost Binary Classification
# Dataset: us_catering_orders_final_2000.csv
# ================================================

# (Optional) Install if missing:
# pip install xgboost pandas scikit-learn seaborn matplotlib imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score
)
from xgboost import XGBClassifier

# Optional SMOTE if severe imbalance
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

# -----------------------------
# 1) Load dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("data/us_catering_orders.csv")
print("✅ Loaded:", df.shape, "rows")

# Normalize any potential column name variation for the days-since-last feature
if "days_since_last" in df.columns and "days_since_last_order" not in df.columns:
    df.rename(columns={"days_since_last": "days_since_last_order"}, inplace=True)

# -----------------------------
# 2) Define features (Before vs After)
# -----------------------------
base_features = [
    "order_amount_usd",
    "distance_miles",
    "delivery_time_min",
    "customer_rating"
]

full_features = base_features + [
    "num_past_orders",
    "days_since_last_order",
    "order_frequency_90d"
]

target = "repeat_customer"

# Safety check
for col in full_features + [target]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# -----------------------------
# 3) Train/Test split (same split used for both comparisons)
# -----------------------------
X_base = df[base_features]
X_full = df[full_features]
y = df[target]

Xb_train, Xb_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.35, random_state=42, stratify=y
)

Xf_train, Xf_test, _, _ = train_test_split(
    X_full, y, test_size=0.35, random_state=42, stratify=y
)

# -----------------------------
# 4) Helper: Fit & Evaluate XGBoost
# -----------------------------
def fit_and_eval_xgb(X_train, X_test, y_train, y_test, label="Model"):
    # Handle class imbalance using scale_pos_weight (neg/pos)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = max(1.0, neg / max(1, pos))  # avoid div-by-zero

    # Optionally apply SMOTE only if very imbalanced (>70/30) and library available
    use_smote = False
    ratio = abs((neg / (neg + pos)) - 0.5) * 2  # 0..1 scale
    if ratio > 0.3 and HAS_SMOTE:
        print(f"⚖️ {label}: Severe imbalance detected, applying SMOTE on training set...")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        # After SMOTE, classes are balanced → set scale_pos_weight back to 1
        spw = 1.0
        use_smote = True

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=spw
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results = {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Accuracy": round(float(acc), 3),
        "BalancedAcc": round(float(bacc), 3),
        "Recall": round(float(rec), 3),
        "smote_used": use_smote
    }

    return model, results, cm

# -----------------------------
# 5) Train & Evaluate
# -----------------------------
print("\n=== Training with BASE features (4) ===")
model_base, res_base, cm_base = fit_and_eval_xgb(Xb_train, Xb_test, y_train, y_test, label="Base")

print("Confusion Matrix (BASE):\n", cm_base)
print("Results (BASE):", res_base)

print("\n=== Training with FULL features (7) ===")
model_full, res_full, cm_full = fit_and_eval_xgb(Xf_train, Xf_test, y_train, y_test, label="Full")

print("Confusion Matrix (FULL):\n", cm_full)
print("Results (FULL):", res_full)

# -----------------------------
# 6) Compare Before vs After
# -----------------------------
print("\n=== BEFORE vs AFTER Feature Engineering ===")
print(f"Accuracy  - Base: {res_base['Accuracy']:.3f}  |  Full: {res_full['Accuracy']:.3f}")
print(f"Bal.Acc   - Base: {res_base['BalancedAcc']:.3f} |  Full: {res_full['BalancedAcc']:.3f}")
print(f"Recall(1) - Base: {res_base['Recall']:.3f}     |  Full: {res_full['Recall']:.3f}")

# -----------------------------
# 7) Plot Confusion Matrix (FULL)
# -----------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred No Repeat', 'Pred Repeat'],
            yticklabels=['Actual No Repeat', 'Actual Repeat'])
plt.title('Confusion Matrix - XGBoost (FULL Features)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Feature Importance (FULL)
# -----------------------------
plt.figure(figsize=(8,4))
importances = model_full.feature_importances_
order = np.argsort(importances)[::-1]
sns.barplot(x=importances[order], y=np.array(full_features)[order])
plt.title("Feature Importance - XGBoost (FULL Features)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -----------------------------
# 9) Five Use Case Predictions (with column names)
# -----------------------------
use_cases = pd.DataFrame([
    {
        "order_amount_usd": 45,
        "distance_miles": 3.2,
        "delivery_time_min": 35,
        "customer_rating": 4.8,
        "num_past_orders": 5,
        "days_since_last_order": 12,
        "order_frequency_90d": 3
    },
    {
        "order_amount_usd": 120,
        "distance_miles": 10.5,
        "delivery_time_min": 80,
        "customer_rating": 3.2,
        "num_past_orders": 0,
        "days_since_last_order": 85,
        "order_frequency_90d": 0
    },
    {
        "order_amount_usd": 75,
        "distance_miles": 5.5,
        "delivery_time_min": 55,
        "customer_rating": 4.5,
        "num_past_orders": 2,
        "days_since_last_order": 40,
        "order_frequency_90d": 1
    },
    {
        "order_amount_usd": 22,
        "distance_miles": 1.4,
        "delivery_time_min": 25,
        "customer_rating": 4.9,
        "num_past_orders": 7,
        "days_since_last_order": 5,
        "order_frequency_90d": 4
    },
    {
        "order_amount_usd": 150,
        "distance_miles": 8.5,
        "delivery_time_min": 70,
        "customer_rating": 4.6,
        "num_past_orders": 4,
        "days_since_last_order": 50,
        "order_frequency_90d": 1
    }
], columns=full_features)

use_cases["predicted_repeat"] = model_full.predict(use_cases)

print("\n=== Use Case Predictions (0/1) ===")
print(use_cases)
