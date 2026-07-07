import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from app.config import Config
from app.similarity.image_embedding import (
    ImageEmbeddingGenerator,
)

generator = ImageEmbeddingGenerator()

generator.embed_directory()

embeddings = np.load(
    Config.IMAGE_EMBEDDINGS_FILE
)

print()

print(embeddings.shape)

print(embeddings.dtype)

print(
    np.linalg.norm(
        embeddings[0]
    )
)