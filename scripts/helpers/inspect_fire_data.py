import pandas as pd

df = pd.read_csv("data/portugal_fires.csv")

print("📦 File loaded!")

# Show raw column names
print("\n🧩 Columns:")
print(df.columns.tolist())

# Show row count
print(f"\n🔢 Total rows: {len(df)}")

# Show first few rows
print("\n🧪 Sample rows:")
print(df.head())

# Check date range
if "acq_date" in df.columns or "ACQ_DATE" in df.columns:
    col = "acq_date" if "acq_date" in df.columns else "ACQ_DATE"
    df[col] = pd.to_datetime(df[col])
    print(f"\n📅 Fire data date range: {df[col].min()} → {df[col].max()}")

# Basic lat/lon stats
if "latitude" in df.columns or "LATITUDE" in df.columns:
    lat_col = "latitude" if "latitude" in df.columns else "LATITUDE"
    lon_col = "longitude" if "longitude" in df.columns else "LONGITUDE"
    print("\n🌍 Latitude range:", df[lat_col].min(), "→", df[lat_col].max())
    print("🌍 Longitude range:", df[lon_col].min(), "→", df[lon_col].max())
