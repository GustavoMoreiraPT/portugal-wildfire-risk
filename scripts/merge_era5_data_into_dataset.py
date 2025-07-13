import os
import xarray as xr

input_dir = "data/era5/multi"
output_path = "data/era5/era5_portugal_2023.nc"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

variables = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
    "surface_pressure": "sp",
}

merged_datasets = []

for var, short_name in variables.items():
    pattern = os.path.join(input_dir, f"era5_{var}_2023_*.nc")
    print(f"📦 Merging files for: {var}")
    ds = xr.open_mfdataset(pattern, combine="by_coords")
    
    # Rename to standard short variable name
    ds = ds.rename({list(ds.data_vars)[0]: short_name})
    merged_datasets.append(ds[[short_name]])

# Merge all variables into a single Dataset
full_ds = xr.merge(merged_datasets)
full_ds.to_netcdf(output_path)

print(f"✅ Multi-variable dataset saved at: {output_path}")
