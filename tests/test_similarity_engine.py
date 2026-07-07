import numpy as np

from app.config import Config

from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer

from app.similarity.faiss_index import FaissIndex
from app.similarity.similarity_engine import SimilarityEngine


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

loader = DataLoader()

df = loader.load_data()

engineer = FeatureEngineer(df)

df = engineer.prepare_dataset()

# -------------------------------------------------------
# Use first 20 products
# -------------------------------------------------------

df = df.head(20).reset_index(drop=True)

# -------------------------------------------------------
# Build temporary text FAISS
# -------------------------------------------------------

text_embeddings = np.load(
    Config.TEXT_EMBEDDINGS_FILE
)

text_embeddings = text_embeddings[:20]

text_index = FaissIndex(
    Config.TEXT_FAISS_INDEX_FILE
)

text_index.build(text_embeddings)

text_index.save()

# -------------------------------------------------------
# Build temporary image FAISS
# -------------------------------------------------------

image_embeddings = np.load(
    Config.IMAGE_EMBEDDINGS_FILE
)

image_index = FaissIndex(
    Config.IMAGE_FAISS_INDEX_FILE
)

image_index.build(image_embeddings)

image_index.save()

# -------------------------------------------------------
# Engine
# -------------------------------------------------------

engine = SimilarityEngine(df)

query_product = df.iloc[0]["uniq_id"]

results = engine.find_similar_products(
    product_id=query_product,
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

print("\nSimilarity Engine Results\n")

print(
    results[
        [
            "rank",
            "fusion_score",
            "text_score",
            "image_score",
            "structured_score",
            "product_name",
        ]
    ]
)