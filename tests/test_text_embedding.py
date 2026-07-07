import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.preprocessing.data_loader import ProductDataLoader
from app.preprocessing.feature_engineering import FeatureEngineer
from app.similarity.text_embedding import TextEmbeddingGenerator


def main():

    # Load dataset
    loader = ProductDataLoader()
    df = loader.load_data()

    print(f"Loaded {len(df)} products")

    # Feature engineering
    engineer = FeatureEngineer(df)
    df = engineer.prepare_dataset()

    print("Feature engineering completed.")

    # Generate embeddings
    generator = TextEmbeddingGenerator()

    embeddings = generator.fit(df)

    print(f"Embedding shape: {embeddings.shape}")

    # Save artifacts
    generator.save_embeddings()

    print("Embeddings saved successfully.")


if __name__ == "__main__":
    main()