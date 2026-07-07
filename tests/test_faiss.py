import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from app.similarity.text_embedding import TextEmbeddingGenerator
from app.similarity.faiss_index import FaissIndex


def main():

    generator = TextEmbeddingGenerator()

    embeddings, _ = generator.load_embeddings()

    print(f"Embeddings Shape : {embeddings.shape}")

    index = FaissIndex()

    index.build(embeddings)

    print(f"Vectors in index : {index.size}")

    index.save()

    print("Index saved successfully.")

    # Reload the index to verify persistence
    index.load()

    print("Index loaded successfully.")

    scores, indices = index.search(
        embeddings[0],
        top_k=5,
    )

    print("\nNearest Neighbours")
    print(indices)

    print("\nSimilarity Scores")
    print(scores)


if __name__ == "__main__":
    main()