from typing import Dict, Set

import pandas as pd

from app.config import Config


class FusionSimilarity:
    """
    Combines text, image and structured similarity scores
    into a single fusion score.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_similarity,
        image_similarity,
        structured_similarity,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.text_similarity = text_similarity
        self.image_similarity = image_similarity
        self.structured_similarity = structured_similarity

    # ---------------------------------------------------------

    def _text_candidates(
        self,
        product_id: str,
        top_k: int = 100,
    ) -> Dict[str, float]:

        results = self.text_similarity.find_similar_products(
            product_id=product_id,
            top_k=top_k,
        )

        return {
            row["uniq_id"]: row["similarity_score"]
            for _, row in results.iterrows()
        }

    # ---------------------------------------------------------

    def _image_candidates(
        self,
        product_id: str,
        top_k: int = 100,
    ) -> Dict[str, float]:

        results = self.image_similarity.find_similar_products(
            product_id=product_id,
            top_k=top_k,
        )

        return {
            row["uniq_id"]: row["image_similarity"]
            for _, row in results.iterrows()
        }

    # ---------------------------------------------------------

    @staticmethod
    def _candidate_pool(
        text_scores: Dict[str, float],
        image_scores: Dict[str, float],
    ) -> Set[str]:

        return set(text_scores.keys()) | set(image_scores.keys())

    # ---------------------------------------------------------

    def find_similar_products(
        self,
        product_id: str,
        top_k: int = Config.TOP_K_DEFAULT,
        candidate_pool_size: int = 100,
    ) -> pd.DataFrame:

        text_scores = self._text_candidates(
            product_id,
            candidate_pool_size,
        )

        image_scores = self._image_candidates(
            product_id,
            candidate_pool_size,
        )

        candidates = self._candidate_pool(
            text_scores,
            image_scores,
        )

        results = []

        for candidate_id in candidates:

            text_score = text_scores.get(candidate_id, 0.0)

            image_score = image_scores.get(candidate_id, 0.0)

            structured_score = (
                self.structured_similarity.compute_similarity(
                    product_id,
                    candidate_id,
                )
            )

            fusion_score = (

                Config.TEXT_SIMILARITY_WEIGHT * text_score

                +

                Config.IMAGE_SIMILARITY_WEIGHT * image_score

                +

                Config.STRUCTURED_SIMILARITY_WEIGHT * structured_score

            )

            product = self.df[
                self.df["uniq_id"] == candidate_id
            ].iloc[0]

            results.append({

                "uniq_id": candidate_id,

                "product_name": product["product_name"],

                "brand": product["brand"],

                "sales_price": product["sales_price"],

                "rating": product["rating"],

                "large": product["large"],

                "text_score": round(text_score, 4),

                "image_score": round(image_score, 4),

                "structured_score": round(structured_score, 4),

                "fusion_score": round(fusion_score, 4),

            })

        results = (
            pd.DataFrame(results)
            .sort_values(
                "fusion_score",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        results.insert(
            0,
            "rank",
            range(1, len(results) + 1),
        )

        return results.head(top_k)