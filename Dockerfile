FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/mibici_balance_pipeline.joblib \
    SCHEMA_PATH=/app/models/model_schema.json \
    MODEL_VERSION=2.0.0 \
    PYTHONPATH=/app

# libgomp1 es obligatorio para HistGradientBoostingRegressor (OpenMP) en imagenes slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY mibici/ ./mibici/
COPY configs/ ./configs/
COPY models/mibici_balance_pipeline.joblib ./models/mibici_balance_pipeline.joblib
COPY models/model_schema.json ./models/model_schema.json

RUN pip install --no-cache-dir -e . --no-deps && \
    useradd --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "mibici.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]