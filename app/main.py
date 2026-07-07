from fastapi import FastAPI

from app.preprocessing.data_loader import DataLoader
from app.preprocessing.feature_engineering import FeatureEngineer

from app.similarity.similarity_engine import SimilarityEngine

from app.api.routes import router

app = FastAPI(
    title="Amazon Fashion Product Similarity API",
    version="1.0.0",
)

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

loader = DataLoader()

df = loader.load_data()

engineer = FeatureEngineer(df)

df = engineer.prepare_dataset()

# -------------------------------------------------------
# Similarity Engine
# -------------------------------------------------------

engine = SimilarityEngine(df)

app.state.engine = engine

# -------------------------------------------------------
# Routes
# -------------------------------------------------------

app.include_router(router)