import pandas as pd

# Load full dataset
df = pd.read_csv("data/fire_risk_training_data.csv")

# Split into fire and no-fire
fires = df[df["fire_occurred"] == 1]
no_fires = df[df["fire_occurred"] == 0]

print(f"🔥 Fire samples: {len(fires)}")
print(f"🌲 No-fire samples: {len(no_fires)}")

# Sample N no-fire examples (or all if fewer)
N = min(len(no_fires), max(1000, len(fires) * 10))
no_fires_sampled = no_fires.sample(n=N, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([fires, no_fires_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save
balanced_df.to_csv("data/fire_risk_balanced.csv", index=False)
print(f"✅ Saved balanced dataset to: data/fire_risk_balanced.csv")
