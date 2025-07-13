import pandas as pd
import xarray as xr
from tqdm import tqdm

# --- Load ERA5 data ---
print("📂 Loading ERA5 data from: data/era5")
era5_files = [f"data/era5/era5_temperature_2023_{month:02d}.nc" for month in range(1, 13)]
ds_list = [xr.open_dataset(f) for f in era5_files]
ds = xr.concat(ds_list, dim="time")

# Flatten t2m using valid_time
t2m = ds["t2m"] - 273.15  # Convert Kelvin to Celsius
valid_times = ds["valid_time"].values

print("✅ ERA5 dataset merged.")
print(f"📊 ERA5 time range:\n{pd.to_datetime(valid_times[0])} → {pd.to_datetime(valid_times[-1])}")
print(f"📍 Grid: lat {ds.latitude.min().item()} → {ds.latitude.max().item()}, lon {ds.longitude.min().item()} → {ds.longitude.max().item()}")

# --- Load fire data ---
fires = pd.read_csv("data/portugal_fires.csv")
fires["datetime"] = pd.to_datetime(
    fires["acq_date"].astype(str) + " " + fires["acq_time"].astype(str).str.zfill(4),
    format="%Y-%m-%d %H%M",
    errors="coerce"
)
fires["datetime_hour"] = fires["datetime"].dt.floor("h")
fires["lat_rounded"] = fires["latitude"].round(2)
fires["lon_rounded"] = fires["longitude"].round(2)
fire_keys = set(zip(fires["datetime_hour"], fires["lat_rounded"], fires["lon_rounded"]))

print("🕒 Sample fire datetimes and coords:")
print(fires[["datetime_hour", "lat_rounded", "lon_rounded"]].dropna().head())

# --- Generate training samples ---
print("\n📦 Generating labeled fire/no-fire samples from ERA5 data...")

records = []
latitudes = ds["latitude"].values
longitudes = ds["longitude"].values

# Determine shape and flatten t2m to align with valid_time
flat_t2m = t2m.stack(valid_time_flat=("time", "valid_time"))

for i, timestamp in enumerate(tqdm(valid_times, desc="🕐 Processing hours")):
    # Get temperature slice for this timestamp
    try:
        temp_data = flat_t2m.sel(valid_time=timestamp)
    except KeyError:
        continue  # In case the timestamp isn't found, skip

    df_temp = temp_data.to_dataframe(name="temperature").reset_index()
    df_temp["datetime_hour"] = pd.to_datetime(timestamp)
    df_temp["lat_rounded"] = df_temp["latitude"].round(2)
    df_temp["lon_rounded"] = df_temp["longitude"].round(2)
    df_temp["key"] = list(zip(df_temp["datetime_hour"], df_temp["lat_rounded"], df_temp["lon_rounded"]))
    df_temp["fire_occurred"] = df_temp["key"].isin(fire_keys).astype(int)
    records.append(df_temp[["datetime_hour", "lat_rounded", "lon_rounded", "temperature", "fire_occurred"]])

# Combine and save
df_all = pd.concat(records, ignore_index=True)
df_all.to_csv("data/fire_risk_training_data.csv", index=False)
print("✅ Saved training data to: data/fire_risk_training_data.csv")
