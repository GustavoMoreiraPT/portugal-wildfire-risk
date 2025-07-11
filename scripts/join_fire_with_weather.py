import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Load fire data
fires = pd.read_csv("data/portugal_fires.csv")

# Convert date + time to full datetime
fires["acq_time"] = fires["acq_time"].apply(lambda x: f"{int(x):04d}")
fires["datetime"] = pd.to_datetime(fires["acq_date"] + fires["acq_time"], format="%Y-%m%d%H%M", errors='coerce')
fires["datetime_hour"] = fires["datetime"].dt.floor("H")  # round to nearest hour

# Load ERA5 data
ds = xr.open_dataset("data/era5/portugal_2023-08-01.nc")

# Get valid time values (make sure this matches your file!)
time_coord = "valid_time" if "valid_time" in ds.coords else "time"

# Function to get closest weather values
def get_weather_at_point(lat, lon, time):
    try:
        # Select nearest time, lat, lon
        weather = ds.sel(
            **{
                time_coord: time,
                "latitude": lat,
                "longitude": lon
            },
            method="nearest"
        )
        return {
            "temp_c": float(weather["t2m"].values - 273.15),
            "humidity": float(weather["r"].values) if "r" in weather else None,
            "wind_u": float(weather["u10"].values),
            "wind_v": float(weather["v10"].values)
        }
    except Exception as e:
        return {"temp_c": None, "humidity": None, "wind_u": None, "wind_v": None}

# Apply to first 50 fires (keep it fast while testing)
sample = fires.head(50).copy()
weather_features = []

print("🔗 Matching weather to fire points...")
for _, row in tqdm(sample.iterrows(), total=len(sample)):
    weather = get_weather_at_point(row.latitude, row.longitude, row.datetime_hour)
    weather_features.append(weather)

# Combine
weather_df = pd.DataFrame(weather_features)
sample_with_weather = pd.concat([sample.reset_index(drop=True), weather_df], axis=1)

# Calculate wind speed
sample_with_weather["wind_speed"] = np.sqrt(
    sample_with_weather["wind_u"]**2 + sample_with_weather["wind_v"]**2
)

# Optional: drop u/v wind components if not needed
sample_with_weather.drop(columns=["wind_u", "wind_v"], inplace=True)

# Save updated dataset
sample_with_weather.to_csv("data/fires_with_weather.csv", index=False)
print("✅ Final enriched data with wind speed saved to data/fires_with_weather.csv")
