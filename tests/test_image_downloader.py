import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer
from app.preprocessing.image_downloader import ImageDownloader


loader = DataLoader()

df = loader.load_data()

engineer = FeatureEngineer(df)

df = engineer.prepare_dataset()

# Test with first 20 products
df = df.head(20)

downloader = ImageDownloader()

downloader.download_dataset(df)

print("\nDone.")