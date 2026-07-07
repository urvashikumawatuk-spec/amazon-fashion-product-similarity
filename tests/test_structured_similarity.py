import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import Config

from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer

from app.similarity.structured_similarity import StructuredSimilarity


loader = DataLoader()

df = loader.load_data()

engineer = FeatureEngineer(df)

df = engineer.prepare_dataset()

similarity = StructuredSimilarity(df)

product_id = df.iloc[0][Config.PRODUCT_ID]

print("\nQuery Product\n")

print(
    similarity.dataframe.iloc[0][
        [
            "product_name",
            "brand",
            "sales_price",
            "rating",
        ]
    ]
)

print("\nSimilar Products\n")

results = similarity.find_similar_products(
    product_id,
    top_k=5,
)

print(
    results[
        [
            "rank",
            "structured_similarity",
            "product_name",
            "brand",
            "sales_price",
            "rating",
        ]
    ]
)