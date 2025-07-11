import cdsapi
import os

# Create an output folder
os.makedirs("data/era5", exist_ok=True)

# Set up the CDS API client
c = cdsapi.Client()

# Request ERA5 hourly data (1 variable, 1 day, 1 region)
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',  # Or use 'grib' if needed
        'variable': [
            '2m_temperature', 'surface_pressure', 'relative_humidity',
            '10m_u_component_of_wind', '10m_v_component_of_wind'
        ],
        'year': '2023',
        'month': '08',
        'day': '01',
        'time': [f'{hour:02d}:00' for hour in range(24)],  # All 24 hours
        'area': [42.15, -9.56, 36.95, -6.19],  # North, West, South, East (Portugal)
    },
    'data/era5/portugal_2023-08-01.nc'
)

print("✅ ERA5 data downloaded!")
