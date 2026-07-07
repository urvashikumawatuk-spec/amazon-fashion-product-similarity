from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import Config


class ProductDataLoader:
    """
    Loads and validates the Amazon Fashion dataset.

    Responsibilities:
    - Read the LDJSON dataset
    - Validate required columns
    - Provide basic dataset statistics

    This class intentionally DOES NOT perform preprocessing,
    feature engineering, or missing value handling.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Config.DATA_FILE
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from disk.

        Returns
        -------
        pd.DataFrame
            Raw dataframe.
        """

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {self.data_path}"
            )

        self.df = pd.read_json(
            self.data_path,
            lines=True
        )

        self._validate_columns()

        return self.df

    def _validate_columns(self) -> None:
        """
        Ensure all required columns are present.
        """

        missing_columns = [
            column
            for column in Config.REQUIRED_COLUMNS
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return dataframe if already loaded,
        otherwise load it.
        """

        if self.df is None:
            return self.load_data()

        return self.df

    def dataset_summary(self) -> dict:
        """
        Return basic dataset information.
        """

        df = self.get_dataframe()

        return {
            "num_products": len(df),
            "num_features": len(df.columns),
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / (1024 ** 2),
                2
            ),
        }

    def missing_value_summary(self) -> pd.Series:
        """
        Return missing values for each column.
        """

        df = self.get_dataframe()

        return (
            df.isna()
              .sum()
              .sort_values(ascending=False)
        )


DataLoader = ProductDataLoader