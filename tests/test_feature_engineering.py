from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer
from app.config import Config


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

loader = DataLoader()

df = loader.load_data()

print(f"Original Shape: {df.shape}")

# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

engineer = FeatureEngineer(df)

processed_df = engineer.prepare_dataset()

print(f"Processed Shape: {processed_df.shape}")

# ---------------------------------------------------------
# Check Required Columns
# ---------------------------------------------------------

print("\nChecking required columns...")

missing_columns = [
    column
    for column in Config.REQUIRED_COLUMNS
    if column not in processed_df.columns
]

assert len(missing_columns) == 0, (
    f"Missing columns: {missing_columns}"
)

print("✓ Required columns exist.")

# ---------------------------------------------------------
# Engineered Columns
# ---------------------------------------------------------

engineered_columns = [
    Config.WEIGHT_OUTPUT_COLUMN,
    "category_text",
    "search_text",
    "image_url",
]

print("\nChecking engineered columns...")

for column in engineered_columns:
    assert column in processed_df.columns, f"{column} not created."

print("✓ Engineered columns created.")

# ---------------------------------------------------------
# Search Text
# ---------------------------------------------------------

print("\nExample search text:\n")

print(processed_df.loc[0, "search_text"])

assert (
    processed_df["search_text"]
    .str.len()
    .gt(0)
    .all()
)

print("✓ Search text generated.")

# ---------------------------------------------------------
# Weight Parsing
# ---------------------------------------------------------

print("\nParsed weights:")

print(
    processed_df[
        [
            Config.WEIGHT_COLUMN,
            Config.WEIGHT_OUTPUT_COLUMN,
        ]
    ].head()
)

assert processed_df[Config.WEIGHT_OUTPUT_COLUMN].dtype in [
    "float64",
    "float32",
]

print("✓ Weight parsed.")

# ---------------------------------------------------------
# Image URL
# ---------------------------------------------------------

print("\nExample image URLs:")

print(processed_df["image_url"].head())

assert (
    processed_df["image_url"]
    .notna()
    .all()
)

print("✓ Image URL created.")

# ---------------------------------------------------------
# Null Check
# ---------------------------------------------------------

print("\nChecking required columns for nulls...")

required = [
    "product_name",
    "search_text",
    "image_url",
]

print(
    processed_df[required]
    .isnull()
    .sum()
)

print("✓ Required fields validated.")

# ---------------------------------------------------------
# Success
# ---------------------------------------------------------

print("\nFeature Engineering Test Passed Successfully!")