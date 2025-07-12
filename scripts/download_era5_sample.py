import cdsapi
import os
from datetime import datetime, timedelta

# === CONFIG ===
start_date = datetime(2023, 7, 1)
end_date = datetime(2023, 8, 30)  # inclusive
output_dir = "data/era5"
os.makedirs(output_dir, exist_ok=True)

# === INIT CLIENT ===
c = cdsapi.Client()

# === DOWNLOAD LOOP ===
date = start_date
while date <= end_date:
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")

    output_file = f"{output_dir}/portugal_{year}-{month}-{day}.nc"
    if os.path.exists(output_file):
        print(f"✅ Already downloaded: {output_file}")
        date += timedelta(days=1)
        continue

    print(f"⬇️ Downloading ERA5 for {year}-{month}-{day}...")

    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind'
                ],
                'year': year,
                'month': month,
                'day': day,
                'time': [f"{hour:02d}:00" for hour in range(24)],
                'area': [42.15, -9.56, 36.95, -6.19],  # Portugal
            },
            output_file
        )
        print(f"✅ Saved to {output_file}")
    except Exception as e:
        print(f"❌ Failed to download {year}-{month}-{day}: {e}")

    date += timedelta(days=1)
