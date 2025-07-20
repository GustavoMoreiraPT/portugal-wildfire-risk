import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="🔥 Portugal Fire Risk Viewer", layout="wide")
st.title("🔥 Wildfire Risk Predictions – Portugal 2023")
st.markdown("""
Select a date to visualize predicted wildfire risk across the country.
Predictions are based on ERA5 climate variables and a Random Forest classifier trained on 2023 data.
""")

# Load data (parquet must be pre-generated)
@st.cache_data
def load_fire_risk_data():
    df = pd.read_parquet("inference_outputs/fire_risk_2023.parquet")
    df["time"] = pd.to_datetime(df["time"])
    return df

fire_df = load_fire_risk_data()

# --- Sidebar controls ---
st.sidebar.header("🗓️ Select Date")
dates = fire_df["time"].dt.date.unique()
selected_date = st.sidebar.date_input("Choose a day in 2023", min_value=dates.min(), max_value=dates.max(), value=datetime(2023, 7, 20))

# --- Filter data ---
daily_df = fire_df[fire_df["time"].dt.date == selected_date]
if daily_df.empty:
    st.warning(f"No data found for {selected_date}.")
    st.stop()

# --- Pivot to 2D grid ---
heatmap_df = daily_df.groupby(['latitude', 'longitude'])['fire_risk'].max().unstack()

# --- Plot heatmap ---
st.subheader(f"📍 Fire Risk Map for {selected_date.strftime('%B %d, %Y')}")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    heatmap_df.sort_index(ascending=False),
    cmap="YlOrRd",
    cbar_kws={'label': '🔥 Fire Risk (0–1)'},
    linewidths=0.3,
    ax=ax
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
st.pyplot(fig)

# --- Download Option ---
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="💾 Download this day's predictions as CSV",
    data=daily_df.to_csv(index=False),
    file_name=f"fire_risk_{selected_date}.csv",
    mime="text/csv"
)
