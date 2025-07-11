import xarray as xr
import matplotlib.pyplot as plt

# Load the NetCDF file
ds = xr.open_dataset("data/era5/portugal_2023-08-01.nc")

print("📊 Dataset structure:")
print(ds)

# Extract the 2m temperature variable
t2m = ds['t2m']  # values are in Kelvin

# Convert to Celsius
t2m_celsius = t2m - 273.15

# Select the hour you want (e.g., 12:00 UTC)
t2m_hour = t2m_celsius.sel(valid_time='2023-08-01T12:00:00')

# Plot it
plt.figure(figsize=(10, 6))
t2m_hour.plot(
    cmap='inferno',
    cbar_kwargs={'label': 'Temperature (°C)'}
)

plt.title("ERA5 2m Temperature at 12:00 UTC — 2023-08-01")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()

# Save figure
plt.savefig("outputs/era5_temperature_map.png")
plt.show()
