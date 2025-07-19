import xarray as xr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from pathlib import Path

INPUT_FILE = Path("data/dataset/labeled_era_2023.nc").resolve()
print(f"📁 Resolved path: {INPUT_FILE}")
print(f"📦 File exists: {INPUT_FILE.exists()}")

ds = xr.open_dataset(INPUT_FILE)

print(ds)


FIRE_TARGET = 10_000
NO_FIRE_TARGET = 10_000
NOISE_STD_FRACTION = 0.05  # 5% of std dev for data augmentation
OUTPUT_DIR = Path("preprocessed_data")

# === Load NetCDF and convert to DataFrame ===
print("📦 Loading NetCDF file...")
ds = xr.open_dataset(INPUT_FILE)
df = ds.to_dataframe().reset_index()

print(f"📊 Initial shape: {df.shape}")
df = df.dropna()
print(f"✅ After dropping NaNs: {df.shape}")

# === Separate classes ===
fire_df = df[df["fire"] == 1]
no_fire_df = df[df["fire"] == 0]

print(f"🔥 Fire samples: {len(fire_df)}")
print(f"❄️  No-fire samples: {len(no_fire_df)}")

# === Smart Oversampling of Fire Samples ===
print(f"🧪 Augmenting fire samples to {FIRE_TARGET}...")
fire_oversampled = fire_df.sample(n=FIRE_TARGET, replace=True, random_state=42)

feature_cols = fire_oversampled.drop(columns=["fire", "time", "latitude", "longitude"]).columns
feature_stds = fire_df[feature_cols].std()

# Inject Gaussian noise
for col in feature_cols:
    noise_std = feature_stds[col] * NOISE_STD_FRACTION
    noise = np.random.normal(loc=0.0, scale=noise_std, size=len(fire_oversampled))
    fire_oversampled[col] += noise

# === Undersample No-Fire Samples ===
print(f"📉 Sampling no-fire samples to {NO_FIRE_TARGET}...")
no_fire_undersampled = no_fire_df.sample(n=NO_FIRE_TARGET, replace=False, random_state=42)

# === Merge and Shuffle ===
balanced_df = pd.concat([fire_oversampled, no_fire_undersampled])
balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"📦 Final balanced dataset: {balanced_df.shape}")

# === Feature Extraction ===
X = balanced_df.drop(columns=["fire", "time", "latitude", "longitude"])
y = balanced_df["fire"]

# === Normalize Features ===
print("📐 Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
print("🔀 Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Save Outputs ===
print("💾 Saving to disk...")
OUTPUT_DIR.mkdir(exist_ok=True)

train_df = pd.DataFrame(X_train, columns=X.columns)
train_df["fire"] = y_train.values
train_df.to_parquet(OUTPUT_DIR / "train.parquet")

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["fire"] = y_test.values
test_df.to_parquet(OUTPUT_DIR / "test.parquet")

print("✅ Preprocessing complete.")
print(f"📁 Train saved to: {OUTPUT_DIR / 'train.parquet'}")
print(f"📁 Test saved to: {OUTPUT_DIR / 'test.parquet'}")
