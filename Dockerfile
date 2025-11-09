# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# System deps (git, wget often needed by HF; build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
  git wget curl ca-certificates build-essential \
  && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.1+cu121 \
  && rm -rf /var/lib/apt/lists/*

# App dirs
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

# Python deps (cached by layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime env
ENV DJANGO_SETTINGS_MODULE=core.settings \
  HF_HOME=/root/.cache/huggingface \
  MODELS_DIR=/models

COPY . .

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "core.asgi:application", "-k", "uvicorn.workers.UvicornWorker"]
