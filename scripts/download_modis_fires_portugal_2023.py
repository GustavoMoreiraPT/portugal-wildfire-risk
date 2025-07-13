import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# 🌍 Portugal bounding box
LAT_MIN, LAT_MAX = 36.9, 42.0
LON_MIN, LON_MAX = -9.6, -6.1

# 📁 Dataset info
dataset = "rtatman/2020-2023-wildfire-data"
filename = "MODIS_C6_2020_2023.csv"
output_dir = "data/fires"
raw_path = os.path.join(output_dir, filename)
filtered_path = os.path.join(output_dir, "modis_portugal_2023.csv")

# 🧱 Ensure directory exists
os.makedirs(output_dir, exist_ok=True)

# 📦 Download from Kaggle if needed
if not os.path.exists(raw_path):
    print(f"⬇️ Downloading {filename} from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(dataset, file_name=filename, path=output_dir, force=True)

    # Unzip
    zip_path = raw_path + ".zip"
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)
        print(f"✅ Unzipped to {output_dir}")
    else:
        print("❌ ZIP file not found after download.")
else:
    print(f"✅ {filename} already exists.")

# 📂 Load & filter
print("📂 Reading MODIS data...")
df = pd.read_csv(raw_path, parse_dates=["acq_date"])

# 🎯 Filter by year and Portugal bounding box
mask = (
    (df["acq_date"].dt.year == 2023) &
    (df["latitude"].between(LAT_MIN, LAT_MAX)) &
    (df["longitude"].between(LON_MIN, LON_MAX))
)
df_portugal = df[mask]

# 💾 Save results
df_portugal.to_csv(filtered_path, index=False)
print(f"✅ Saved {len(df_portugal)} records to {filtered_path}")
