import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Config


class StructuredSimilarity:
    """
    Computes cosine similarity over structured product features.

    Categorical Features
    --------------------
    - Brand
    - Parent Category

    Numerical Features
    ------------------
    - Sales Price
    - Rating
    - Weight
    """

    def __init__(self, dataframe: pd.DataFrame):

        self.dataframe = dataframe.reset_index(drop=True).copy()

        for column in ["brand", "parent___child_category__all"]:
            self.dataframe[column] = (
                self.dataframe[column]
                .fillna("")
                .astype(str)
                .apply(lambda value: value.replace("{", "").replace("}", ""))
            )

        self.product_to_index = {
            pid: idx
            for idx, pid in enumerate(
                self.dataframe[Config.PRODUCT_ID]
            )
        }
     

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    Config.STRUCTURED_CATEGORICAL_FEATURES,
                ),
                (
                    "numeric",
                    StandardScaler(),
                    Config.STRUCTURED_NUMERIC_FEATURES,
                ),
            ]
        )

        self.feature_matrix = self.preprocessor.fit_transform(
            self.dataframe
        )


    # -----------------------------------------------------
    # Compute Similarity Between Two Products
    # -----------------------------------------------------

    def compute_similarity(
        self,
        product_id_1: str,
        product_id_2: str,
    ) -> float:
        """
        Compute cosine similarity between two products
        using their structured features.
        """

        idx1 = self.product_to_index[product_id_1]
        idx2 = self.product_to_index[product_id_2]

        score = cosine_similarity(
            self.feature_matrix[idx1],
            self.feature_matrix[idx2],
        )[0][0]

        return float(score)

    # -----------------------------------------------------
    # Find Top K Structured Similar Products
    # -----------------------------------------------------

    def find_similar_products(
        self,
        product_id: str,
        top_k: int = Config.TOP_K_DEFAULT,
    ) -> pd.DataFrame:

        idx = self.product_to_index[product_id]

        query_vector = self.feature_matrix[idx]

        scores = cosine_similarity(
            query_vector,
            self.feature_matrix,
        )[0]

        order = np.argsort(scores)[::-1]

        results = []

        rank = 1

        for neighbour_idx in order:

            if neighbour_idx == idx:
                continue

            product = self.dataframe.iloc[neighbour_idx]

            result = {
                "rank": rank,
                "structured_similarity": float(scores[neighbour_idx]),
            }

            for column in Config.METADATA_COLUMNS:
                result[column] = product[column]

            results.append(result)

            rank += 1

            if len(results) == top_k:
                break

        return pd.DataFrame(results)