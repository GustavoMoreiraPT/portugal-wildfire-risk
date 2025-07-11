import xarray as xr

# Load the file
ds = xr.open_dataset("data/era5/portugal_2023-08-01.nc")

# See what variables are in it
print("🌐 Dataset summary:")
print(ds)

# Peek at a specific variable, e.g. temperature
print("\n🌡️ Temperature data:")
print(ds['t2m'])  # 't2m' is 2-meter air temperature
