import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Paths ===
data_dir = Path("preprocessed_data")
output_dir = Path("randomForestResults")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load Data ===
train_df = pd.read_parquet(data_dir / "train.parquet")
test_df = pd.read_parquet(data_dir / "test.parquet")

X_train = train_df.drop(columns=["fire"])
y_train = train_df["fire"]
X_test = test_df.drop(columns=["fire"])
y_test = test_df["fire"]

# === Train Random Forest ===
print("🌳 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# === Predict & Evaluate ===
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("🔥 ROC AUC:", roc_auc_score(y_test, y_prob))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Precision:", precision_score(y_test, y_pred))
print("🔁 Recall:", recall_score(y_test, y_pred))
print("📦 F1 Score:", f1_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix_rf.png")
plt.close()

# === Feature Importance Plot ===
importances = rf_model.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("🔍 Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_rf.png")
plt.close()

# === Save Model ===
model_path = output_dir / "random_forest_fire_model.pkl"
joblib.dump(rf_model, model_path)
print(f"\n💾 Model saved as: {model_path}")
print(f"📈 Plots saved to: {output_dir}")
