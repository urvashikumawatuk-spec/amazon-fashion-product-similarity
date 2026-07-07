import ast
import re
from typing import Any

import numpy as np
import pandas as pd

from app.config import Config


class FeatureEngineer:
    """
    Performs feature engineering for the product similarity system.

    Responsibilities
    ----------------
    - Handle missing values
    - Parse product weight
    - Extract category information
    - Build searchable text representation

    This class intentionally DOES NOT:
    - Generate embeddings
    - Encode categorical variables
    - Compute similarity
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    # ==========================================================
    # Public API
    # ==========================================================

    def prepare_dataset(self) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        """

        self._clean_numeric_features()
        self._clean_categorical_features()
        self._clean_text_features()

        self._parse_weight_column()
        self._extract_category_text()
        self._build_search_text()
        self._create_image_url() 

        return self.df

    # ==========================================================
    # Cleaning
    # ==========================================================

    def _clean_numeric_features(self) -> None:

        for column in Config.NUMERIC_FEATURES:

            median = self.df[column].median()

            self.df[column] = self.df[column].fillna(median)

    def _clean_categorical_features(self) -> None:

        for column in Config.CATEGORICAL_FEATURES:

            self.df[column] = (
                self.df[column]
                .fillna(Config.UNKNOWN_CATEGORY)
                .astype(str)
                .str.strip()
            )

    def _clean_text_features(self) -> None:

        for column in Config.TEXT_COLUMNS:

            self.df[column] = (
                self.df[column]
                .fillna(Config.EMPTY_TEXT)
                .astype(str)
                .apply(self._normalize_text)
            )

    def _create_image_url(self) -> None:
        """
        Create a unified image URL using the highest
        resolution image available.
        """

        self.df[Config.IMAGE_URL_COLUMN] = (
            self.df["large"]
            .fillna(self.df["medium"])
            .fillna(self.df["image_urls__small"])
            .fillna("")
            .astype(str)
        )
    # ==========================================================
    # Weight Processing
    # ==========================================================

    def _parse_weight_column(self) -> None:
        """
        Convert product weight to grams and create
        a numeric weight feature.
        """

        parsed_weight = (
            self.df[Config.WEIGHT_COLUMN]
            .apply(self._convert_weight_to_grams)
        )

        self.df[Config.WEIGHT_OUTPUT_COLUMN] = parsed_weight

        median_weight = (
            self.df[Config.WEIGHT_OUTPUT_COLUMN]
            .median()
        )

        self.df[Config.WEIGHT_OUTPUT_COLUMN] = (
            self.df[Config.WEIGHT_OUTPUT_COLUMN]
            .fillna(median_weight)
        )

    def _convert_weight_to_grams(self, weight: Any) -> float:

        if pd.isna(weight):
            return np.nan

        weight = str(weight).strip().lower()

        if weight == Config.UNKNOWN_WEIGHT:
            return np.nan

        numbers = re.findall(r"\d+\.?\d*", weight)

        if not numbers:
            return np.nan

        value = float(numbers[0])

        if "kg" in weight:
            return value * 1000

        if "g" in weight:
            return value

        if "lb" in weight:
            return value * 453.592

        if "oz" in weight:
            return value * 28.3495

        return value

    # ==========================================================
    # Category Processing
    # ==========================================================

    def _extract_category_text(self) -> None:

        self.df[Config.CATEGORY_TEXT_COLUMN] = (
            self.df[Config.CATEGORY_COLUMN]
            .apply(self._category_to_text)
        )

    def _category_to_text(self, category: Any) -> str:

        if pd.isna(category):
            return ""

        try:

            if isinstance(category, str):
                category = ast.literal_eval(category)

            if isinstance(category, dict):

                keys = list(category.keys())

                return self._normalize_text(" ".join(keys))

        except Exception:
            pass

        return ""

    # ==========================================================
    # Search Text
    # ==========================================================

    def _build_search_text(self) -> None:

        base_text = (
            self.df[Config.TEXT_COLUMNS]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

        self.df[Config.SEARCH_TEXT_COLUMN] = (

            base_text

            + " "

            + self.df[Config.CATEGORY_TEXT_COLUMN]

        ).apply(self._normalize_text)

    

    # ==========================================================
    # Utilities
    # ==========================================================

    @staticmethod
    def _normalize_text(text: str) -> str:

        if pd.isna(text):
            return ""

        text = str(text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()