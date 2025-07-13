import pandas as pd

df = pd.read_csv("data/fire_risk_training_data.csv")

print("🔥 Fire vs No-Fire label counts:")
print(df["fire_occurred"].value_counts())

print("\n💯 Percentage fire samples:")
print(df["fire_occurred"].value_counts(normalize=True) * 100)
