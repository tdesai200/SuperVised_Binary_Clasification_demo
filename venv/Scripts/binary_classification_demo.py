# ====================================================
# Gujarati Catering - Full Feature XGBoost Classifier
# Dataset: us_catering_orders_final_2000.csv
# ====================================================

# Install if missing:
# pip install pandas xgboost scikit-learn seaborn matplotlib imbalanced-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score
)
from xgboost import XGBClassifier

# -----------------------------
# 1. Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("data/us_catering_orders.csv")
print(f"Loaded {df.shape[0]} rows")

# Ensure column matches script assumption
if "days_since_last" in df.columns and "days_since_last_order" not in df.columns:
    df.rename(columns={"days_since_last": "days_since_last_order"}, inplace=True)

# -----------------------------
# 2. Full Feature Set
# -----------------------------
features = [
    "order_amount_usd",
    "distance_miles",
    "delivery_time_min",
    "customer_rating",
    "num_past_orders",
    "days_since_last_order",
    "order_frequency_90d"
]
target = "repeat_customer"

X = df[features]
y = df[target]

# -----------------------------
# 3. Train/Test Split (65/35)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42, stratify=y
)

print(f"Training rows: {X_train.shape[0]}, Testing rows: {X_test.shape[0]}")

# -----------------------------
# 4. Train XGBoost Model
# -----------------------------
# Auto-adjust class imbalance weight
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos = max(1.0, neg / max(1, pos))

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos
)

model.fit(X_train, y_train)
print("\nModel trained successfully")

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy:           {accuracy:.3f}")
print(f"Balanced Accuracy:  {balanced_acc:.3f}")
print(f"Recall (Repeat=1):  {recall:.3f}")
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# -----------------------------
# 6. Confusion Matrix Plot
# -----------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred No Repeat', 'Pred Repeat'],
            yticklabels=['Actual No Repeat', 'Actual Repeat'])
plt.title("Confusion Matrix - XGBoost (Full Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Feature Importance Plot
# -----------------------------
plt.figure(figsize=(8,4))
importances = model.feature_importances_
order = np.argsort(importances)[::-1]
sns.barplot(x=importances[order], y=np.array(features)[order])
plt.title("Feature Importance - XGBoost (Full Feature Model)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -----------------------------
# 8.  Five Use Case Predictions
# -----------------------------
use_cases = pd.DataFrame([
    {
        "order_amount_usd": 45, "distance_miles": 3.2, "delivery_time_min": 35,
        "customer_rating": 4.8, "num_past_orders": 5, "days_since_last_order": 12,
        "order_frequency_90d": 3
    },
    {
        "order_amount_usd": 120, "distance_miles": 10.5, "delivery_time_min": 80,
        "customer_rating": 3.2, "num_past_orders": 0, "days_since_last_order": 85,
        "order_frequency_90d": 0
    },
    {
        "order_amount_usd": 22, "distance_miles": 1.4, "delivery_time_min": 25,
        "customer_rating": 4.9, "num_past_orders": 7, "days_since_last_order": 5,
        "order_frequency_90d": 4
    },
    {
        "order_amount_usd": 150, "distance_miles": 8.5, "delivery_time_min": 70,
        "customer_rating": 4.6, "num_past_orders": 4, "days_since_last_order": 50,
        "order_frequency_90d": 1
    }
])

use_cases["predicted_repeat"] = model.predict(use_cases)

print("\n=== Use Case Predictions (0 = No Repeat, 1 = Repeat) ===")
print(use_cases)
