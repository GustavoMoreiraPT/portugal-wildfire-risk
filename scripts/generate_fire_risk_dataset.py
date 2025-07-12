import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# === Load all ERA5 files ===
era5_folder = "data/era5"
nc_files = sorted([f for f in os.listdir(era5_folder) if f.endswith(".nc")])

# === Load fire data ===
fires = pd.read_csv("data/portugal_fires.csv")
fires["acq_time"] = fires["acq_time"].apply(lambda x: f"{int(x):04d}")
fires["datetime"] = pd.to_datetime(
    fires["acq_date"] + fires["acq_time"], format="%Y-%m%d%H%M", errors='coerce'
)
fires["datetime_hour"] = fires["datetime"].dt.floor("H")

# Round lat/lon to ERA5 grid resolution
def round_coord(val, resolution=0.25):
    return round(val / resolution) * resolution

fires["lat_rounded"] = fires["latitude"].apply(round_coord)
fires["lon_rounded"] = fires["longitude"].apply(round_coord)

# Build a lookup set for quick fire detection
fire_set = set(zip(fires["datetime_hour"], fires["lat_rounded"], fires["lon_rounded"]))

# === Process all ERA5 files ===
samples = []

print("📦 Generating labeled fire/no-fire samples from ERA5 data...")

for nc_file in tqdm(nc_files):
    path = os.path.join(era5_folder, nc_file)
    try:
        ds = xr.open_dataset(path)

        # Determine time coordinate key
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"

        for t in ds[time_coord].values:
            time = pd.to_datetime(str(t))
            for lat in ds.latitude.values:
                for lon in ds.longitude.values:
                    rounded = (time, round_coord(lat), round_coord(lon))
                    fire_label = 1 if rounded in fire_set else 0

                    # Extract weather features
                    point = ds.sel({time_coord: time, "latitude": lat, "longitude": lon}, method="nearest")
                    temp_c = float(point["t2m"].values) - 273.15
                    wind_u = float(point["u10"].values)
                    wind_v = float(point["v10"].values)
                    wind_speed = np.sqrt(wind_u**2 + wind_v**2)

                    samples.append({
                        "datetime": time,
                        "latitude": lat,
                        "longitude": lon,
                        "temp_c": temp_c,
                        "wind_speed": wind_speed,
                        "fire_occurred": fire_label
                    })

    except Exception as e:
        print(f"⚠️ Failed to process {nc_file}: {e}")

# Save dataset
df = pd.DataFrame(samples)
df.to_csv("data/fire_risk_training_data.csv", index=False)
print("✅ Done! Saved to data/fire_risk_training_data.csv")
