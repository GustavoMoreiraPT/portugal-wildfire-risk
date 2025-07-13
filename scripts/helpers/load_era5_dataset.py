import xarray as xr
import os
import glob

def load_era5_dataset(directory="data/era5"):
    print(f"📂 Loading ERA5 data from: {directory}")

    files = sorted(glob.glob(os.path.join(directory, "era5_temperature_2023_*.nc")))
    if not files:
        raise FileNotFoundError("❌ No ERA5 NetCDF files found.")

    print(f"📄 Found {len(files)} files.")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    ds_list = [xr.open_dataset(f) for f in files]
    combined = xr.concat(ds_list, dim="time")

    # Convert from Kelvin to Celsius
    combined["t2m_celsius"] = combined["t2m"] - 273.15

    print(f"✅ ERA5 dataset merged. Time range: {combined.time.values[0]} → {combined.time.values[-1]}")
    return combined
