import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_parquet("inference_outputs/fire_risk_2023.parquet")
df['time'] = pd.to_datetime(df['time'])

# Filter for December 24th
selected_day = "2023-12-24"
daily_df = df[df['time'].dt.date == pd.to_datetime(selected_day).date()]

# Pivot to heatmap format: max fire risk for each (lat, lon)
heatmap_df = daily_df.groupby(['latitude', 'longitude'])['fire_risk'].max().unstack()

# Plot fire risk heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(
    heatmap_df.sort_index(ascending=False),
    cmap="YlOrRd",
    cbar_kws={'label': '🔥 Max Fire Risk (0–1)'},
    linewidths=0.5
)
plt.title(f"🔥 Fire Risk Map for {selected_day}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.grid(False)
plt.show()
