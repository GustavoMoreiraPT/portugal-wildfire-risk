import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# === Config ===
ERA5_FILE = Path("data/dataset/labeled_era5_2023.nc")
MODEL_FILE = Path("randomForestResults/random_forest_fire_model.pkl")
OUTPUT_FILE = Path("inference_outputs/fire_risk_2023.parquet")
FEATURE_COLS = ['t2m', 'd2m', 'u10', 'v10', 'tp', 'sp', 'number']

# === Load ERA5 Dataset ===
print("📦 Loading ERA5 data...")
ds = xr.open_dataset(ERA5_FILE)
df = ds.to_dataframe().reset_index()
df = df.dropna()

# === Drop label column if it exists
if "fire_label" in df.columns:
    df = df.drop(columns=["fire_label"])

print(f"✅ ERA5 shape (after cleaning): {df.shape}")

# === Extract input features (exactly what model expects)
raw_features = df[FEATURE_COLS]

# === Normalize using fresh scaler (match training procedure)
print("📐 Normalizing input...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(raw_features)

# === Load trained model
print("🤖 Loading trained model...")
model = joblib.load(MODEL_FILE)

# === Predict fire risk probabilities
print("🔮 Running predictions...")
probs = model.predict_proba(X_normalized)[:, 1]

# === Attach time/lat/lon metadata
print("📈 Attaching metadata...")
result_df = df[["time", "latitude", "longitude"]].copy()
result_df["fire_risk"] = probs

# === Save output
output_dir = OUTPUT_FILE.parent
output_dir.mkdir(parents=True, exist_ok=True)
result_df.to_parquet(OUTPUT_FILE)

print(f"\n✅ Inference complete.")
print(f"📁 Results saved to: {OUTPUT_FILE}")
