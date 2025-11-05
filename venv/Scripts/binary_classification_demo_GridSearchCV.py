
#  Install dependencies (run once if needed)
# pip install pandas scikit-learn matplotlib seaborn imbalanced-learn

# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 2. Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("us_catering_orders.csv")

print("Dataset Loaded Successfully")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nSample Rows:")
print(df.head())

# -----------------------------
# 3. Feature & Target Selection
# -----------------------------
features = ["order_amount_usd", "distance_miles", "delivery_time_min", "customer_rating"]
target = "repeat_customer"

X = df[features]
y = df[target]

# -----------------------------
# 4. Check Class Balance
# -----------------------------
class_dist = y.value_counts(normalize=True) * 100
print("\n Class Distribution (%):")
print(class_dist)

# -----------------------------
# 5. Split Train/Test Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=52, stratify=y
)
print(X_train.shape, X_test.shape)
# -----------------------------
# 6. Train Model (Decision Tree with GridSearchCV)
# -----------------------------
from sklearn.model_selection import GridSearchCV

print("\n Running GridSearchCV for best Decision Tree parameters...")

params = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)

grid = GridSearchCV(
    estimator=dt,
    param_grid=params,
    scoring='recall',   # you can change to "recall" if you want to reduce FN
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n GridSearch Complete")
print("Best Parameters Found:", grid.best_params_)

# Replace model with tuned version
model = grid.best_estimator_
print("\n Best Decision Tree Model Selected & Trained")

# -----------------------------
# 7. Predict on Test Data
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 8. Evaluate Model
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("\n Model Evaluation Results")
print(f"Confusion Matrix:\n{cm}")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Balanced Accuracy: {balanced_acc:.3f}")

# -----------------------------
# 9. Plot Confusion Matrix
# -----------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: No Repeat', 'Predicted: Repeat'],
            yticklabels=['Actual: No Repeat', 'Actual: Repeat'])
plt.title('Confusion Matrix - Gujarati Catering Customer Repeat Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
#plt.show()
plt.show(block=False)


# -----------------------------
# 10. Test Use Cases
# -----------------------------
use_cases = pd.DataFrame([
    {"order_amount_usd": 45,  "distance_miles": 3.2, "delivery_time_min": 35, "customer_rating": 4.8},
    {"order_amount_usd": 120, "distance_miles": 10.5, "delivery_time_min": 80, "customer_rating": 3.2},
    {"order_amount_usd": 75,  "distance_miles": 5.5, "delivery_time_min": 55, "customer_rating": 4.5},
    {"order_amount_usd": 60,  "distance_miles": 2.8, "delivery_time_min": 40, "customer_rating": 4.9},
    {"order_amount_usd": 70,  "distance_miles": 6.8, "delivery_time_min": 41, "customer_rating": 4.9}
])

use_cases["predicted_repeat"] = model.predict(use_cases)

print("\n Use Case Predictions:")
print(use_cases)

# -----------------------------
# 11. Summary
# -----------------------------
print("\n Model Evaluation Summary:")
summary = {
    "TP": int(tp),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "Accuracy": round(float(accuracy), 3),
    "Balanced Accuracy": round(float(balanced_acc), 3),
    "Class Distribution": class_dist.to_dict()
}
print(summary)
