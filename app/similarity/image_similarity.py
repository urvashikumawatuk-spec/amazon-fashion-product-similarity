import numpy as np
import pandas as pd

from app.config import Config
from app.similarity.faiss_index import FaissIndex


class ImageSimilarity:
    """
    Performs image similarity search using
    FAISS over image embeddings.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        embeddings: np.ndarray = None,
        faiss_index: FaissIndex = None,
    ):

        self.df = dataframe.reset_index(drop=True)

        if embeddings is None:
            self.embeddings = np.load(
                Config.IMAGE_EMBEDDINGS_FILE
            )
        else:
            self.embeddings = embeddings

        self.product_to_index = {
            product_id: idx
            for idx, product_id in enumerate(self.df["uniq_id"])
        }

        if faiss_index is None:
            self.index = FaissIndex()
            self.index.build(self.embeddings)
        else:
            self.index = faiss_index

    # ---------------------------------------------------------

    def find_similar_products(
        self,
        product_id: str,
        top_k: int = Config.TOP_K_DEFAULT,
    ) -> pd.DataFrame:

        query_embedding = self.embeddings[
            self.product_to_index[product_id]
        ]

        scores, indices = self.index.search(
            query_embedding,
            top_k + 1,
        )

        results = []

        rank = 1

        for score, idx in zip(scores, indices):

            if idx == product_id:
                continue

            product = self.df.iloc[idx]

            result = {
                "rank": rank,
                "image_similarity": float(score),
            }

            for column in Config.METADATA_COLUMNS:
                result[column] = product[column]

            results.append(result)

            rank += 1

            if len(results) == top_k:
                break

        return pd.DataFrame(results)