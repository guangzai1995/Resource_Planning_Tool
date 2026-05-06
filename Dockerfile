# ── Stage 1: Build frontend ───────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --registry https://registry.npmmirror.com

COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python runtime ───────────────────────────────
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY backend/pyproject.toml ./
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple ".[prod]" 2>/dev/null || \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        fastapi>=0.115 \
        "uvicorn[standard]>=0.30" \
        sqlalchemy>=2.0 \
        alembic \
        pydantic>=2.0 \
        pydantic-settings \
        cachetools \
        scipy \
        numpy \
        pandas \
        openpyxl \
        structlog \
        python-multipart \
        aiofiles

# Copy backend source
COPY backend/app ./app

# Copy frontend build output
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy data (optional, may be mounted at runtime)
RUN mkdir -p /data /model

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

ENV PYTHONPATH=/app
ENV SQLITE_PATH=/data/rpt.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
