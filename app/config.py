from pathlib import Path


class Config:
    """
    Configuration for the Amazon Fashion Product Similarity System.
    """

    # ==========================================================
    # Dataset
    # ==========================================================

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR / "app" / "data"

    DATA_FILE = (
        DATA_DIR
        / "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
    )

    RANDOM_STATE = 42

    # ==========================================================
    # Identifier
    # ==========================================================

    PRODUCT_ID = "uniq_id"

    # ==========================================================
    # Metadata (Returned in API Response)
    # ==========================================================

    METADATA_COLUMNS = [
        "uniq_id",
        "asin",
        "product_name",
        "brand",
        "sales_price",
        "rating",
        "product_url",
        "large",
    ]

    # ==========================================================
    # Structured Features
    # ==========================================================

    NUMERIC_FEATURES = [
        "sales_price",
        "rating",
    ]

    CATEGORICAL_FEATURES = [
        "brand",
    ]

    WEIGHT_COLUMN = "weight"

    UNKNOWN_WEIGHT = "999999999"

    WEIGHT_OUTPUT_COLUMN = "weight_in_grams"

    # ==========================================================
    # Text Features
    # ==========================================================

    TEXT_COLUMNS = [
        "product_name",
        "brand",
        "meta_keywords",
    ]

    CATEGORY_COLUMN = "parent___child_category__all"

    CATEGORY_TEXT_COLUMN = "category_text"

    SEARCH_TEXT_COLUMN = "search_text"

    # ==========================================================
    # Structured Features
    # ==========================================================

    STRUCTURED_CATEGORICAL_FEATURES = [
    "brand",
    "parent___child_category__all",
    ]

    STRUCTURED_NUMERIC_FEATURES = [
        "sales_price",
        "rating",
        WEIGHT_OUTPUT_COLUMN,
    ]

    # ==========================================================
    # Image
    # ==========================================================

    IMAGE_COLUMN = "large"
    IMAGE_URL_COLUMN = 'image_url'

    # ==========================================================
    # Image Download
    # ==========================================================

    IMAGE_DIR = DATA_DIR / "images"

    IMAGE_DOWNLOAD_TIMEOUT = 10

    IMAGE_DOWNLOAD_WORKERS = 8

    # ==========================================================
    # Image Embeddings
    # ==========================================================

    IMAGE_EMBEDDING_MODEL = "resnet50"

    IMAGE_EMBEDDING_DIM = 2048

    IMAGE_EMBEDDINGS_FILE = (
        DATA_DIR / "image_embeddings.npy"
    ) 
    # ==========================================================
    # Required Columns
    # ==========================================================

    REQUIRED_COLUMNS = (
        [PRODUCT_ID]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [WEIGHT_COLUMN]
        + TEXT_COLUMNS
        + [CATEGORY_COLUMN]
        + [IMAGE_COLUMN]
    )

    # ==========================================================
    # Missing Value Handling
    # ==========================================================

    UNKNOWN_CATEGORY = "Unknown"

    EMPTY_TEXT = ""

    NUMERIC_IMPUTATION = "median"

    # ==========================================================
    # Similarity Fusion Weights
    # ==========================================================

    STRUCTURED_SIMILARITY_WEIGHT = 0.20

    TEXT_SIMILARITY_WEIGHT = 0.45

    IMAGE_SIMILARITY_WEIGHT = 0.35

    # ==========================================================
    # Search
    # ==========================================================

    TOP_K_DEFAULT = 10

    # ==========================================================
    # Embedding Model
    # ==========================================================

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    EMBEDDING_DIMENSION = 384

    EMBEDDING_BATCH_SIZE = 128

    DEVICE = "cpu"

    # ==========================================================
    # Artifacts
    # ==========================================================

    ARTIFACT_DIR = BASE_DIR / "artifacts"

    TEXT_EMBEDDINGS_FILE = ARTIFACT_DIR / "text_embeddings.npy"

    PRODUCT_IDS_FILE = ARTIFACT_DIR / "product_ids.npy"

    # ==========================================================
    # FAISS
    # ==========================================================

    FAISS_INDEX_FILE = ARTIFACT_DIR / "text_faiss.index"

    TEXT_FAISS_INDEX_FILE = ARTIFACT_DIR / "text_faiss.index"

    IMAGE_FAISS_INDEX_FILE = ARTIFACT_DIR / "image_faiss.index"

    HNSW_M = 64

    HNSW_EF_CONSTRUCTION = 400

    HNSW_EF_SEARCH = 128