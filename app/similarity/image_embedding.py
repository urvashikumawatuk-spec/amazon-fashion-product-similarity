from pathlib import Path
from typing import List

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from torchvision import models
from torchvision.models import ResNet50_Weights

from app.config import Config


class ImageEmbeddingGenerator:
    """
    Generates image embeddings using a pretrained ResNet50.

    Each image is converted into a normalized
    2048-dimensional embedding.
    """

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        weights = ResNet50_Weights.DEFAULT

        model = models.resnet50(weights=weights)

        # remove classification layer
        model.fc = torch.nn.Identity()

        self.model = model.to(self.device)
        self.model.eval()

        self.transform = weights.transforms()

    # ---------------------------------------------------------

    def embed_image(
        self,
        image_path: Path,
    ) -> np.ndarray:

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():

            embedding = self.model(image)

        embedding = embedding.cpu().numpy()[0]

        embedding = embedding.astype(np.float32)

        embedding /= np.linalg.norm(embedding)

        return embedding

    # ---------------------------------------------------------

    def embed_directory(self):

        image_dir = Path(Config.IMAGE_DIR)

        image_files = sorted(
            image_dir.glob("*.jpg")
        )

        embeddings: List[np.ndarray] = []

        product_ids = []

        for image_path in tqdm(image_files):

            try:

                embedding = self.embed_image(
                    image_path
                )

                embeddings.append(embedding)

                product_ids.append(
                    image_path.stem
                )

            except Exception:

                continue

        embeddings = np.vstack(
            embeddings
        ).astype(np.float32)

        np.save(
            Config.IMAGE_EMBEDDINGS_FILE,
            embeddings,
        )

        np.save(
            Config.DATA_DIR / "image_product_ids.npy",
            np.array(product_ids),
        )

        print()

        print(
            f"Generated {len(product_ids)} embeddings."
        )

        print(
            embeddings.shape
        )