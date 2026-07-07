import numpy as np
import pandas as pd

from app.config import Config

from app.similarity.faiss_index import FaissIndex
from app.similarity.text_similarity import TextSimilarity
from app.similarity.image_similarity import ImageSimilarity
from app.similarity.structured_similarity import StructuredSimilarity
from app.similarity.fusion_similarity import FusionSimilarity


class SimilarityEngine:
    """
    Main entry point for the product similarity system.

    Responsibilities
    ----------------
    - Load embeddings
    - Load FAISS indices
    - Initialize similarity modules
    - Expose a single API for product search
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):

        self.dataframe = dataframe.reset_index(drop=True)

        # --------------------------------------------------
        # Load Text Embeddings
        # --------------------------------------------------

        self.text_embeddings = np.load(
            Config.TEXT_EMBEDDINGS_FILE
        )

        # --------------------------------------------------
        # Load Image Embeddings
        # --------------------------------------------------

        self.image_embeddings = np.load(
            Config.IMAGE_EMBEDDINGS_FILE
        )

        # --------------------------------------------------
        # Text FAISS
        # --------------------------------------------------

        self.text_index = FaissIndex(
            Config.TEXT_FAISS_INDEX_FILE
        )

        self.text_index.load()

        # --------------------------------------------------
        # Image FAISS
        # --------------------------------------------------

        self.image_index = FaissIndex(
            Config.IMAGE_FAISS_INDEX_FILE
        )

        self.image_index.load()

        # --------------------------------------------------
        # Similarity Modules
        # --------------------------------------------------

        self.text_similarity = TextSimilarity(
            dataframe=self.dataframe,
            embeddings=self.text_embeddings,
            faiss_index=self.text_index,
        )

        self.image_similarity = ImageSimilarity(
            dataframe=self.dataframe,
            embeddings=self.image_embeddings,
            faiss_index=self.image_index,
        )

        self.structured_similarity = StructuredSimilarity(
            dataframe=self.dataframe,
        )

        self.fusion_similarity = FusionSimilarity(
            dataframe=self.dataframe,
            text_similarity=self.text_similarity,
            image_similarity=self.image_similarity,
            structured_similarity=self.structured_similarity,
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def find_similar_products(
        self,
        product_id: str,
        top_k: int = Config.TOP_K_DEFAULT,
    ) -> pd.DataFrame:

        return self.fusion_similarity.find_similar_products(
            product_id=product_id,
            top_k=top_k,
        )