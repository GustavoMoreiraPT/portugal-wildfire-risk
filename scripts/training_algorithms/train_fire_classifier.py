import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Data ===
data_dir = Path("preprocessed_data")
train_df = pd.read_parquet(data_dir / "train.parquet")
test_df = pd.read_parquet(data_dir / "test.parquet")

X_train = train_df.drop(columns=["fire"])
y_train = train_df["fire"]
X_test = test_df.drop(columns=["fire"])
y_test = test_df["fire"]

# === Train Model ===
print("🚂 Training logistic regression...")
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# === Predict ===
print("🔍 Evaluating...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# === Metrics ===
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("🔥 ROC AUC:", roc_auc_score(y_test, y_prob))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Precision:", precision_score(y_test, y_pred))
print("🔁 Recall:", recall_score(y_test, y_pred))
print("📦 F1 Score:", f1_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# === Feature Importance ===
coef = model.coef_[0]
features = X_train.columns
importance = pd.Series(coef, index=features).sort_values(key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index)
plt.title("🔍 Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# === Save Model ===
joblib.dump(model, "logistic_fire_model.pkl")
print("\n💾 Model saved as 'logistic_fire_model.pkl'")
print("📈 Plots saved: confusion_matrix.png, feature_importance.png")
