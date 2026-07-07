from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer

from app.similarity.image_similarity import ImageSimilarity

loader = DataLoader()

df = loader.load_data()

engineer = FeatureEngineer(df)

df = engineer.prepare_dataset()

# Match the downloaded/embedded subset
df = df.head(20).reset_index(drop=True)

similarity = ImageSimilarity(df)

results = similarity.find_similar_products(
    product_index=0,
    top_k=5,
)

print("\nQuery Product\n")

print(
    df.iloc[0][
        [
            "product_name",
            "brand",
        ]
    ]
)

print("\nTop 5 Image Similar Products\n")

print(
    results[
        [
            "rank",
            "image_similarity",
            "product_name",
            "brand",
        ]
    ]
)