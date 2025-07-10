import pandas as pd
import os
from glob import glob

# Step 1: Find all fire CSVs in the data folder
fire_files = glob("data/fire_nrt_*.csv")

# Step 2: Load and concatenate them
dfs = []
for file in fire_files:
    print(f"📄 Loading: {file}")
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
print(f"📊 Combined fire records: {len(combined_df)}")

# Step 3: Filter for Portugal bounding box
pt_df = combined_df[
    (combined_df.latitude >= 36.95) & (combined_df.latitude <= 42.15) &
    (combined_df.longitude >= -9.56) & (combined_df.longitude <= -6.19)
]

# Step 4: Save filtered data
output_path = "data/portugal_fires.csv"
os.makedirs("data", exist_ok=True)
pt_df.to_csv(output_path, index=False)

print(f"✅ Filtered {len(pt_df)} fire records for Portugal.")
print(pt_df.head())
