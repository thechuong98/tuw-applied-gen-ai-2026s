# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Bring in the uv binary (pinned) from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first for layer caching — only the manifest + lock are needed here.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# App code + config.
COPY app ./app
COPY config.yaml .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    CONFIG_PATH=/app/config.yaml
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
