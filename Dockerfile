FROM python:3.14-slim AS builder

WORKDIR /build

RUN pip install uv

COPY pyproject.toml .
COPY uv.lock .
COPY src/ src/

RUN uv sync --no-dev --frozen

FROM python:3.14-slim AS runtime

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /build/.venv /app/.venv
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV HF_HOME="/app/.cache/huggingface"

RUN mkdir -p /app/.cache && \
    HF_HOME=/app/.cache/huggingface python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && \
    chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

EXPOSE 8001

CMD ["uvicorn", "stylemind.main:app", "--host", "0.0.0.0", "--port", "8001"]
