from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import Config


class TextEmbeddingGenerator:
    """
    Generates and manages sentence embeddings for product text.

    Responsibilities
    ----------------
    - Load embedding model
    - Generate embeddings
    - Save embeddings
    - Load embeddings
    - Encode new query text

    Does NOT perform similarity search.
    """

    def __init__(self):

        self.model = SentenceTransformer(
            Config.EMBEDDING_MODEL,
            device=Config.DEVICE
        )

        self.embeddings = None
        self.product_ids = None

        Config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Generate embeddings for the dataset
    # ----------------------------------------------------------

    def fit(self, dataframe: pd.DataFrame) -> np.ndarray:

        texts = dataframe[Config.SEARCH_TEXT_COLUMN].fillna("").tolist()

        self.product_ids = np.asarray(
            dataframe[Config.PRODUCT_ID].values,
            dtype=str,
        )

        self.embeddings = self.encode(texts)

        return self.embeddings

    # ----------------------------------------------------------
    # Encode text
    # ----------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)

    # ----------------------------------------------------------
    # Encode a single query
    # ----------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype(np.float32)

    # ----------------------------------------------------------
    # Save embeddings
    # ----------------------------------------------------------

    def save_embeddings(self) -> None:

        if self.embeddings is None:
            raise ValueError("Embeddings have not been generated.")

        np.save(
            Config.TEXT_EMBEDDINGS_FILE,
            self.embeddings,
        )

        np.save(
            Config.PRODUCT_IDS_FILE,
            self.product_ids.astype(str),
        )

    # ----------------------------------------------------------
    # Load embeddings
    # ----------------------------------------------------------

    def load_embeddings(self):

        embeddings_path = Path(Config.TEXT_EMBEDDINGS_FILE)
        ids_path = Path(Config.PRODUCT_IDS_FILE)

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"{embeddings_path} not found."
            )

        if not ids_path.exists():
            raise FileNotFoundError(
                f"{ids_path} not found."
            )

        self.embeddings = np.load(embeddings_path)

        self.product_ids = np.load(ids_path, allow_pickle=True)
        self.product_ids = np.asarray(self.product_ids, dtype=str)

        return self.embeddings, self.product_ids