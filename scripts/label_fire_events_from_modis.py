import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 📁 Config
era5_path = Path("data/era5/era5_portugal_2023.nc")
fire_csv_path = Path("data/fires/cleaned_fire_events_2023.csv")
output_path = Path("data/dataset/labeled_era5_2023.nc")

TIME_TOLERANCE_HOURS = 1               # ±1 hour match
SPATIAL_TOLERANCE_DEGREES = 0.25       # ±0.25° grid cell

print("📂 Loading ERA5 dataset...")
ds = xr.open_dataset(era5_path)

print("📂 Loading fire events...")
fire_df = pd.read_csv(fire_csv_path, parse_dates=["datetime"])

# Rename time dim for compatibility
if "valid_time" in ds.dims:
    ds = ds.rename({"valid_time": "time"})

# Create empty label array (all False initially)
fire_label = xr.DataArray(
    np.zeros(ds["t2m"].shape, dtype=bool),
    coords=ds["t2m"].coords,
    dims=ds["t2m"].dims,
    name="fire_label"
)

print("🔥 Labeling ERA5 cells near fire events...")
for _, row in tqdm(fire_df.iterrows(), total=len(fire_df)):
    fire_time = pd.to_datetime(row["datetime"]).round("H")

    # Time proximity mask
    time_mask = np.abs((ds["time"].values - np.datetime64(fire_time)) / np.timedelta64(1, "h")) <= TIME_TOLERANCE_HOURS
    if not time_mask.any():
        continue

    # Spatial proximity masks
    lat_mask = np.abs(ds["latitude"] - row["latitude"]) <= SPATIAL_TOLERANCE_DEGREES
    lon_mask = np.abs(ds["longitude"] - row["longitude"]) <= SPATIAL_TOLERANCE_DEGREES

    # Set fire label
    fire_label.loc[dict(
        time=ds["time"].values[time_mask],
        latitude=ds["latitude"].values[lat_mask],
        longitude=ds["longitude"].values[lon_mask]
    )] = True

print("🧬 Merging label into ERA5 dataset...")
ds["fire_label"] = fire_label

print("💾 Saving labeled dataset...")
output_path.parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(output_path)

print(f"✅ Done! Labeled dataset saved to: {output_path}")
