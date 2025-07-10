# scripts/download_kaggle_firms.py

import os
from dotenv import load_dotenv
import kaggle

# Load .env file
load_dotenv()

# Set Kaggle environment variables
os.environ['KAGGLE_USERNAME'] = os.getenv("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = os.getenv("KAGGLE_KEY")

# Dataset: NASA FIRMS Active Fire Dataset
dataset_slug = "vijayveersingh/nasa-firms-active-fire-dataset-modisviirs"
download_path = "data"

# Create data directory if needed
os.makedirs(download_path, exist_ok=True)

# Download dataset
print("📥 Downloading NASA FIRMS data from Kaggle...")
os.system(f'kaggle datasets download -d {dataset_slug} -p {download_path} --unzip')
print("✅ Download complete.")
