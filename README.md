# SAP Labs Assignment – Multimodal Product Similarity Search

## Overview

This project implements a **multimodal product similarity search system** for the Amazon Fashion dataset as part of the SAP Labs assignment.

The goal is to retrieve products that are similar to a given query product by leveraging information from multiple modalities instead of relying solely on textual matching.

The system combines:

* Semantic text understanding using Sentence Transformers
* Visual similarity using ResNet50 image embeddings
* Structured attribute similarity using engineered numerical and categorical features
* FAISS-based approximate nearest neighbour search for scalable retrieval
* Weighted multimodal fusion for final ranking

The entire retrieval pipeline is exposed through a production-style FastAPI application and containerized using Docker.

---

# Problem Statement

Given a product, retrieve the **Top-K most similar products** by considering:

* Product title
* Brand
* Categories
* Product metadata
* Product image
* Structured attributes such as price, rating and weight

Instead of depending on a single modality, the final ranking is obtained by combining information from all available sources.

---

# Architecture

```
                        Query Product
                              │
                              ▼
                   Feature Engineering
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
 Text Embedding         Image Embedding     Structured Features
 (MiniLM)               (ResNet50)          (Price, Rating,
                                                Weight,
                                                Category)
        │                     │                     │
        ▼                     ▼                     ▼
     FAISS                 FAISS            Cosine Similarity
        │                     │                     │
        └────────────── Candidate Union ───────────┘
                              │
                              ▼
                  Weighted Fusion Re-ranking
                              │
                              ▼
                   Top-K Similar Products
                              │
                              ▼
                         FastAPI Endpoint
```

---
```
project/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── data/
│   │
│   ├── preprocessing/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   └── image_downloader.py
│   │
│   ├── similarity/
│   │   ├── faiss_index.py
│   │   ├── fusion_similarity.py
│   │   ├── image_embedding.py
│   │   ├── image_similarity.py
│   │   ├── similarity_engine.py
│   │   ├── structured_similarity.py
│   │   ├── text_embedding.py
│   │   └── text_similarity.py
│   │
│   ├── __init__.py
│   ├── config.py
│   └── main.py
│
├── artifacts/
│   ├── image_embeddings.npy
│   ├── image_faiss.index
│   ├── text_embeddings.npy
│   └── text_faiss.index
│
├── tests/
│   ├── test_api.py
│   ├── test_faiss.py
│   ├── test_feature_engineering.py
│   ├── test_fusion_similarity.py
│   ├── test_image_downloader.py
│   ├── test_image_embedding.py
│   ├── test_image_similarity.py
│   ├── test_similarity_engine.py
│   ├── test_structured_similarity.py
│   └── test_text_similarity.py
│
├── .dockerignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

# Dataset

The project uses the **Amazon Fashion Products Dataset (~30K products)**.

Important columns used during retrieval include:

| Feature                      | Purpose                        |
| ---------------------------- | ------------------------------ |
| uniq_id                      | Product identifier             |
| product_name                 | Primary textual representation |
| brand                        | Structured feature             |
| sales_price                  | Numeric similarity             |
| rating                       | Numeric similarity             |
| weight                       | Numeric similarity             |
| parent___child_category__all | Category representation        |
| meta_keywords                | Additional textual context     |
| product_details__k_v_pairs   | Product metadata               |
| image_urls                   | Image retrieval                |

---

# Data Preprocessing

The raw dataset required significant preprocessing before it could be used for similarity search.

The preprocessing pipeline performs:

### Numeric Cleaning

* Converts price and rating to numeric values
* Parses product weight
* Handles invalid values such as `999999999`
* Creates a single canonical `weight_numeric` feature

### Categorical Cleaning

* Cleans brand names
* Parses nested category dictionaries
* Extracts human-readable category names

### Text Processing

A unified search text is created by combining:

* Product name
* Brand
* Categories
* Meta keywords
* Parsed product metadata

This produces richer semantic embeddings than using only the product title.

### Image Processing

Image URLs are selected using a fallback strategy:

```
Large Image
      ↓
Medium Image
      ↓
Small Image
```

Images are downloaded concurrently using `ThreadPoolExecutor`.

---

# Feature Engineering

Three different feature spaces are created.

## Text Features

The final search text includes:

* Product Name
* Brand
* Categories
* Meta Keywords
* Product Details

These are embedded using a Sentence Transformer.

---

## Structured Features

Structured similarity uses:

Categorical

* Brand
* Category

Numerical

* Sales Price
* Rating
* Product Weight

Categorical variables are one-hot encoded while numerical variables are standardized before computing cosine similarity.

---

## Image Features

Images are converted into feature vectors using a pretrained ResNet50 model with the classification layer removed.

Each image is represented as a normalized **2048-dimensional embedding**.

---

# Text Similarity

Model:

```
SentenceTransformer
all-MiniLM-L6-v2
```

Reasons for selection:

* Strong semantic understanding
* Small model size
* Fast inference
* Excellent performance for retrieval applications

Embeddings:

```
30000 × 384
```

Similarity Search:

* FAISS IndexFlatIP
* Cosine similarity using normalized vectors

---

# Image Similarity

Model:

```
torchvision.models.resnet50
```

Classifier removed:

```
model.fc = Identity()
```

Embedding dimension:

```
2048
```

Similarity retrieval is performed using FAISS.

---

# Structured Similarity

Structured similarity is computed independently.

Pipeline:

```
ColumnTransformer
    │
    ├── OneHotEncoder
    └── StandardScaler
            │
            ▼
Cosine Similarity
```

This captures similarities based on structured product attributes.

---

# FAISS Index

Both text and image retrieval use FAISS.

Advantages:

* Extremely fast nearest neighbour search
* Efficient vector indexing
* Easily scalable to millions of products
* Low memory overhead

Separate indices are maintained for:

* Text embeddings
* Image embeddings

---

# Multimodal Fusion

Instead of relying on one modality, the system combines all similarity signals.

Pipeline:

```
Query Product

↓

Text Retrieval

↓

Image Retrieval

↓

Candidate Union

↓

Structured Similarity

↓

Weighted Fusion

↓

Final Ranking
```

Fusion score:

```
Fusion Score =
0.40 × Text Similarity
+
0.35 × Image Similarity
+
0.25 × Structured Similarity
```

This late fusion strategy allows each modality to contribute independently while maintaining modularity.

---

# Similarity Engine

`SimilarityEngine` serves as the single public interface for the retrieval system.

Responsibilities:

* Load embeddings
* Load FAISS indices
* Initialize similarity modules
* Execute multimodal retrieval
* Return ranked products

The FastAPI layer contains no retrieval logic.

---

# API

The project exposes a REST API using FastAPI.

## Find Similar Products

```
POST /find_similar_products
```

Request

```json
{
    "product_id": "<product_id>",
    "top_k": 5
}
```

Response

```json
{
    "results": [
        {
            "rank": 1,
            "product_id": "...",
            "fusion_score": 0.94,
            "text_score": 0.91,
            "image_score": 0.95,
            "structured_score": 0.88
        }
    ]
}
```

Interactive documentation is available at:

```
http://localhost:8000/docs
```

---

# Testing

The project includes unit and integration tests covering:

* Feature Engineering
* Text Embeddings
* Image Downloader
* Image Embeddings
* FAISS Index
* Text Similarity
* Image Similarity
* Structured Similarity
* Fusion Similarity
* Similarity Engine
* FastAPI API

All tests pass successfully.

---

# Running the Project

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Start the API

```
uvicorn app.main:app --reload
```

Swagger UI:

```
http://localhost:8000/docs
```

---

# Running with Docker

Build:

```
docker build -t sap-product-similarity .
```

Run:

```
docker run -p 8000:8000 sap-product-similarity
```

---

# Design Decisions

Several engineering decisions were made while designing the system.

### Why Sentence Transformers?

Traditional TF-IDF cannot capture semantic similarity. Sentence Transformers generate contextual embeddings that better represent product descriptions.

### Why ResNet50?

ResNet50 provides strong visual representations while remaining computationally efficient and widely adopted.

### Why FAISS?

FAISS enables efficient vector search and scales significantly better than exhaustive nearest-neighbour comparisons.

### Why Late Fusion?

Each modality captures different aspects of product similarity.

* Text captures semantics.
* Images capture visual appearance.
* Structured features capture business attributes.

Late fusion combines their strengths while allowing each retrieval module to evolve independently.

### Why Exclude Seller Information?

Seller details, discounts, inventory and offers describe the marketplace listing rather than intrinsic product characteristics. Including them could bias similarity towards transient business information instead of the products themselves.

---

# Scalability

The architecture is designed with scalability in mind.

Possible production enhancements include:

* Replace `IndexFlatIP` with IVF or HNSW FAISS indices for larger datasets.
* Store embeddings in a vector database such as Milvus or Pinecone.
* Periodically refresh embeddings as new products are added.
* Deploy retrieval as a dedicated microservice.
* Cache frequently requested products.
* Parallelize embedding generation using GPU inference.

---

# Future Improvements

Potential extensions include:

* Cross-modal transformer models (e.g., CLIP) for joint text-image embeddings.
* Learning-to-rank models for adaptive fusion weights.
* Personalized recommendations using user interaction history.
* Hybrid keyword and semantic retrieval.
* Incremental FAISS index updates for real-time catalog changes.
* Evaluation using retrieval metrics such as Recall@K, MAP, and NDCG.

---

# Technologies Used

* Python
* FastAPI
* Pandas
* NumPy
* Scikit-learn
* Sentence Transformers
* PyTorch
* Torchvision
* FAISS
* Pillow
* Docker
* Pytest

---

# Conclusion

This project demonstrates a modular and scalable approach to multimodal product retrieval by combining semantic text embeddings, visual representations, and structured product attributes into a unified ranking framework. The architecture emphasizes clean separation of concerns, reusable components, efficient retrieval with FAISS, and production-ready deployment through FastAPI and Docker.




