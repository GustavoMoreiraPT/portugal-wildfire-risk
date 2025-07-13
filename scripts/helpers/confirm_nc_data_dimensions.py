import xarray as xr

ds = xr.open_dataset("data/era5/era5_portugal_2023.nc")
print(ds)

ds['t2m'].isel(valid_time=0).plot(cmap="coolwarm")
