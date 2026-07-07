import numpy as np

from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer

from app.similarity.faiss_index import FaissIndex
from app.similarity.text_similarity import TextSimilarity
from app.similarity.image_similarity import ImageSimilarity
from app.similarity.structured_similarity import StructuredSimilarity
from app.similarity.fusion_similarity import FusionSimilarity

from app.config import Config


loader = DataLoader()

df = loader.load_data()

engineer = FeatureEngineer(df)

df = engineer.prepare_dataset()

# Keep only the products that have image embeddings
df = df.head(20).reset_index(drop=True)

# -------------------------------------------------------
# Load text embeddings + index
# -------------------------------------------------------

text_embeddings = np.load(
    Config.TEXT_EMBEDDINGS_FILE
)

# Subset to match the 20-row dataframe
text_embeddings_subset = text_embeddings[:20]

text_index = FaissIndex(Config.TEXT_FAISS_INDEX_FILE)

text_index.build(text_embeddings_subset)

text_similarity = TextSimilarity(
    df,
    text_embeddings_subset,
    text_index,
)

# -------------------------------------------------------
# Load image embeddings + index
# -------------------------------------------------------

image_embeddings = np.load(
    Config.IMAGE_EMBEDDINGS_FILE
)

image_index = FaissIndex(Config.IMAGE_FAISS_INDEX_FILE)

image_index.load()

image_similarity = ImageSimilarity(
    df,
    image_embeddings,
    image_index,
)

# -------------------------------------------------------

structured_similarity = StructuredSimilarity(df)

fusion = FusionSimilarity(
    dataframe=df,
    text_similarity=text_similarity,
    image_similarity=image_similarity,
    structured_similarity=structured_similarity,
)

query_product = df.iloc[0]["uniq_id"]

results = fusion.find_similar_products(
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

print("\nFusion Results\n")

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