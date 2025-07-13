import os
import time
import random
from datetime import datetime
import concurrent.futures
import cdsapi

# 📁 Config
variables = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "surface_pressure",
]

months = [f"{i:02}" for i in range(1, 13)]
output_dir = "data/era5/multi"
os.makedirs(output_dir, exist_ok=True)

MAX_WORKERS = 2  # 🧠 Safe value to avoid CDS API job queue limits

def retry_with_backoff(func, max_retries=5, base_delay=60):
    """Retries a function with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "temporarily limited" in error_str or "429" in error_str:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 10)
                print(f"🔁 Backing off for {delay:.1f} seconds (attempt {attempt + 1})")
                time.sleep(delay)
            else:
                raise

def download_variable_month(var: str, month: str):
    year = "2023"
    filename = f"era5_{var}_{year}_{month}.nc"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"✅ File already exists: {filename}")
        return

    print(f"⬇️  Downloading {filename}...")

    c = cdsapi.Client()

    try:
        retry_with_backoff(lambda: c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": var,
                "year": year,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [42.0, -9.6, 36.9, -6.1],  # Portugal bounding box
            },
            filepath,
        ))
        print(f"✅ Saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

# 🚀 Parallel download with throttling
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            executor.submit(download_variable_month, var, month)
            for var in variables
            for month in months
        ]
        concurrent.futures.wait(tasks)

    print("🎉 All downloads complete!")
