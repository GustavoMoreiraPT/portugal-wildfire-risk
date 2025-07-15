import pandas as pd
from pathlib import Path
from datetime import datetime

# 📁 Config
input_file = Path("data/fires/modis_2023_Portugal.csv")
output_file = Path("data/fires/cleaned_fire_events_2023.csv")
confidence_threshold = 50  # You can raise to 70–80 if you want high-confidence only

# 📥 Load
print(f"📂 Reading {input_file}...")
df = pd.read_csv(input_file)

# 🧼 Filter confidence
df = df[df["confidence"] >= confidence_threshold]

# 🕒 Combine date + time into UTC datetime
def parse_datetime(row):
    time_str = f"{int(row['acq_time']):04d}"  # Ensure 4-digit time like "0930"
    return pd.to_datetime(f"{row['acq_date']} {time_str[:2]}:{time_str[2:]}", utc=True)

print("🛠️  Parsing timestamps...")
df["datetime"] = df.apply(parse_datetime, axis=1)

# 🧹 Keep relevant columns only
df_clean = df[["datetime", "latitude", "longitude", "confidence", "satellite", "instrument"]]

# 💾 Save
output_file.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(output_file, index=False)

print(f"✅ Saved {len(df_clean)} fire events to {output_file}")
