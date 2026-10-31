FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

RUN useradd --create-home appuser && mkdir -p /app/var && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Ingestion runs at startup because it downloads the embedding model into the volume.
CMD ["sh", "-c", "uv run --no-sync lira-ingest && uv run --no-sync uvicorn app.interfaces.api:app --host 0.0.0.0 --port 8000"]
