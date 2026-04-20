FROM python:3.11-slim

WORKDIR /app

# System deps for building native extensions (torch, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps — use Docker-specific requirements that skip heavy ML-only
# packages (cornac, matplotlib, seaborn, pytest) not needed by the demo backend.
# Install CPU-only torch first (much smaller download than full torch+CUDA),
# then the rest of the deps.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --timeout 300 \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout 300 -r requirements-docker.txt

# Copy application code
COPY backend/ backend/
COPY frontend_jquery/ frontend_jquery/
COPY ml/models/ ml/models/
COPY ml/evaluation/ ml/evaluation/
COPY ml/data/__init__.py ml/data/__init__.py
COPY ml/scripts/benchmarks/benchmark_common.py ml/scripts/benchmarks/benchmark_common.py
COPY ml/scripts/benchmarks/cooc_rerank.py ml/scripts/benchmarks/cooc_rerank.py
COPY ml/__init__.py ml/__init__.py
COPY ml/scripts/__init__.py ml/scripts/__init__.py
COPY ml/scripts/benchmarks/__init__.py ml/scripts/benchmarks/__init__.py

# Copy pre-built artifacts (embeddings, catalog, cooc)
COPY artifacts/demo/ artifacts/demo/

# Create data dir for SQLite
RUN mkdir -p data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.demo.main:app", "--host", "0.0.0.0", "--port", "8000"]
