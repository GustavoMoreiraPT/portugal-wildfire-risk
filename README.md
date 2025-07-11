# 🔥 Portugal Wildfire Risk

This project collects and visualizes active wildfire data in **Portugal** using NASA's FIRMS data (MODIS + VIIRS), downloaded via Kaggle. It filters and maps fire detections to help understand wildfire activity in the region.

---

## 🚀 Features

- Downloads recent fire data from Kaggle
- Filters data by geographic bounding box (Portugal)
- Creates an interactive fire map using Folium

---

## 🛠️ Setup

```bash
# Clone the repo and set up Python environment
git clone git@github.com:your-username/portugalWildFires.git
cd portugalWildFires
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

## 📦 Usage

# Download FIRMS fire data from Kaggle
python scripts/download_kaggle_firms.py

# Filter for Portugal
python scripts/filter_portugal_fires.py

# Generate map
python scripts/map_portugal_fires.py
