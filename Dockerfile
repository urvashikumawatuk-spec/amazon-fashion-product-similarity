# =========================================================
# SAP Labs Product Similarity Search
# Dockerfile
# =========================================================

FROM python:3.11-slim

# ----------------------------
# Environment Variables
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------
# Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Install System Dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Install Python Dependencies
# ----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy Entire Project
# ----------------------------
COPY . .

# ----------------------------
# Expose FastAPI Port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Start FastAPI Application
# ----------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]