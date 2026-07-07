from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import logging
import requests
import pandas as pd
from tqdm import tqdm

from app.config import Config


logging.basicConfig(level=logging.INFO)


class ImageDownloader:
    """
    Downloads and caches product images locally.

    Images are downloaded only once. Existing files are skipped.
    """

    def __init__(self):

        self.image_dir = Path(Config.IMAGE_DIR)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------

    def _download_single(self, row) -> bool:
        """
        Download one image.

        Returns
        -------
        bool
            True if downloaded successfully.
        """

        product_id = row[Config.PRODUCT_ID]
        image_url = row[Config.IMAGE_URL_COLUMN]

        if not image_url:
            return False

        image_path = self.image_dir / f"{product_id}.jpg"

        # Already downloaded
        if image_path.exists():
            return True

        try:

            response = requests.get(
                image_url,
                timeout=Config.IMAGE_DOWNLOAD_TIMEOUT,
            )

            response.raise_for_status()

            image_path.write_bytes(response.content)

            return True

        except Exception as e:

            logging.warning(
                f"Failed to download {product_id}: {e}"
            )

            return False

    # -----------------------------------------------------

    def download_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Download all product images.
        """

        rows = [row for _, row in dataframe.iterrows()]

        with ThreadPoolExecutor(
            max_workers=Config.IMAGE_DOWNLOAD_WORKERS
        ) as executor:

            results = list(
                tqdm(
                    executor.map(
                        self._download_single,
                        rows,
                    ),
                    total=len(rows),
                    desc="Downloading Images",
                )
            )

        downloaded = sum(results)

        print()

        print(f"Downloaded / Available : {downloaded}")

        print(f"Missing / Failed       : {len(results)-downloaded}")