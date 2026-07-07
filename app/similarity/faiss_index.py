from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

from app.config import Config


class FaissIndex:
    """
    Builds and manages a FAISS HNSW index for approximate nearest
    neighbour search over product embeddings.

    Responsibilities
    ----------------
    - Build the FAISS index
    - Save the index
    - Load the index
    - Search nearest neighbours

    This class is completely independent of products,
    metadata and embeddings generation.
    """

    def __init__(self, index_path=None):

        self.index = None
        self.index_path = index_path or Config.FAISS_INDEX_FILE

    # ---------------------------------------------------------
    # Build Index
    # ---------------------------------------------------------

    # def build(self, embeddings: np.ndarray) -> None:
    #     """
    #     Build a HNSW index from normalized embeddings.
    #     """

    #     dimension = embeddings.shape[1]
    #     embeddings = np.asarray(embeddings, dtype=np.float32)

    #     index = faiss.IndexFlatIP(dimension)
    #     index.add(embeddings)

    #     self.index = index
    def build(self, embeddings: np.ndarray) -> None:
        """
        Build a FAISS HNSW index from normalized embeddings.
        """

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2-dimensional array.")

        dimension = embeddings.shape[1]

        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass

        index = faiss.IndexHNSWFlat(
            dimension,
            Config.HNSW_M,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efConstruction = Config.HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = Config.HNSW_EF_SEARCH
        index.add(embeddings)

        self.index = index
    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = Config.TOP_K_DEFAULT,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index.

        Returns
        -------
        scores : np.ndarray
        indices : np.ndarray
        """

        if self.index is None:
            raise ValueError("FAISS index has not been built.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        return scores[0], indices[0]

    # ---------------------------------------------------------
    # Save Index
    # ---------------------------------------------------------

    def save(self) -> None:

        if self.index is None:
            raise ValueError("No FAISS index to save.")

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

    # ---------------------------------------------------------
    # Load Index
    # ---------------------------------------------------------

    def load(self, index_path=None) -> None:

        if index_path:
            self.index_path = index_path

        index_path = Path(self.index_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found."
            )

        self.index = faiss.read_index(
            str(index_path)
        )

    # ---------------------------------------------------------
    # Number of vectors
    # ---------------------------------------------------------

    @property
    def size(self) -> int:

        if self.index is None:
            return 0

        return self.index.ntotal