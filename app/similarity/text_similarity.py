from typing import List

import numpy as np
import pandas as pd

from app.config import Config
from app.similarity.faiss_index import FaissIndex


class TextSimilarity:
    """
    Performs text-based product similarity search using
    precomputed embeddings and a FAISS index.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        embeddings: np.ndarray,
        faiss_index: FaissIndex,
    ):

        self.dataframe = dataframe.reset_index(drop=True)

        self.embeddings = embeddings

        self.faiss_index = faiss_index

        # O(1) lookup from product id -> dataframe row
        self.product_to_index = {
            product_id: idx
            for idx, product_id in enumerate(
                self.dataframe[Config.PRODUCT_ID]
            )
        }

    # ---------------------------------------------------------
    # Private helper
    # ---------------------------------------------------------

    def _get_row_index(self, product_id: str) -> int:

        if product_id not in self.product_to_index:
            raise ValueError(
                f"Product '{product_id}' not found."
            )

        return self.product_to_index[product_id]

    # ---------------------------------------------------------
    # Main API
    # ---------------------------------------------------------

    def find_similar_products(
        self,
        product_id: str,
        top_k: int = Config.TOP_K_DEFAULT,
    ) -> pd.DataFrame:

        row_index = self._get_row_index(product_id)

        query_embedding = self.embeddings[row_index]

        scores, indices = self.faiss_index.search(
            query_embedding,
            top_k=top_k + 1,
        )

        results = []

        rank = 1

        for score, idx in zip(scores, indices):

            # Skip the query product
            if idx == row_index:
                continue

            product = self.dataframe.iloc[idx]

            result = {
                "rank": rank,
                "similarity_score": float(score),
            }

            for column in Config.METADATA_COLUMNS:
                result[column] = product[column]

            results.append(result)

            rank += 1

            if len(results) == top_k:
                break

        return pd.DataFrame(results)

    # ---------------------------------------------------------
    # Convenience method
    # ---------------------------------------------------------

    def get_product(self, product_id: str) -> pd.Series:

        row_index = self._get_row_index(product_id)

        return self.dataframe.iloc[row_index]