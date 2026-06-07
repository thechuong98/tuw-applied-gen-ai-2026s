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

# Download the spaCy English model into the project venv. Without this the NER
# layer silently falls back to regex-only (PERSON/GPE/LOC/ORG detection is lost).
# Installed as a wheel via `uv pip` because the uv-managed venv has no pip for
# `spacy download` to call. Pin matches spacy 3.8.x in uv.lock.
RUN uv pip install --python /app/.venv/bin/python \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# App code + config.
COPY app ./app
COPY config.yaml .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    CONFIG_PATH=/app/config.yaml
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
