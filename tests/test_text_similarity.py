import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer

from app.similarity.text_embedding import TextEmbeddingGenerator
from app.similarity.faiss_index import FaissIndex
from app.similarity.text_similarity import TextSimilarity
from app.config import Config

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

def main():

    # ----------------------------
    # Load data
    # ----------------------------

    loader = DataLoader()

    df = loader.load_data()

    engineer = FeatureEngineer(df)

    df = engineer.prepare_dataset()

    # ----------------------------
    # Load embeddings
    # ----------------------------

    embedding_generator = TextEmbeddingGenerator()

    embeddings, _ = embedding_generator.load_embeddings()

    # ----------------------------
    # Load FAISS
    # ----------------------------

    faiss_index = FaissIndex()

    faiss_index.load()

    # ----------------------------
    # Similarity engine
    # ----------------------------

    similarity = TextSimilarity(
        dataframe=df,
        embeddings=embeddings,
        faiss_index=faiss_index,
    )

    product_id = df.iloc[0][Config.PRODUCT_ID]

    results = similarity.find_similar_products(
        product_id,
        top_k=5,
    )

    print(results)
    print("\nQuery Product\n")
    print(similarity.get_product(product_id)[Config.METADATA_COLUMNS])

    print("\nSimilar Products\n")
    print(results)






if __name__ == "__main__":
    main()