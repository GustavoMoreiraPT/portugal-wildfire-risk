import cdsapi
import os
from concurrent.futures import ThreadPoolExecutor

# === Config ===
ERA5_DIR = "data/era5"
os.makedirs(ERA5_DIR, exist_ok=True)

VARIABLE = "2m_temperature"
YEAR = "2023"
MONTHS = [f"{m:02d}" for m in range(1, 13)]
THREADS = 4  # Number of parallel threads; can adjust

# Portugal bounding box: North, West, South, East
AREA = [42.0, -9.6, 36.9, -6.0]  # N, W, S, E

def download_month(month):
    filename = f"{ERA5_DIR}/era5_temperature_{YEAR}_{month}.nc"
    if os.path.exists(filename):
        print(f"✅ Already exists: {filename}")
        return

    print(f"⬇️  Downloading ERA5 data for {YEAR}-{month}...")
    c = cdsapi.Client()

    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": VARIABLE,
                "year": YEAR,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": AREA,
            },
            filename
        )
        print(f"✅ Downloaded: {filename}")
    except Exception as e:
        print(f"❌ Failed to download {month}: {e}")

# === Run in parallel
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        executor.map(download_month, MONTHS)
