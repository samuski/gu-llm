#!/usr/bin/env bash
set -euo pipefail

# (optional) load token from a mounted secret file instead of env
[ -n "${HF_TOKEN_FILE:-}" ] && [ -f "$HF_TOKEN_FILE" ] && export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"

# ensure models dir exists
: "${MODELS_DIR:=/models}"
mkdir -p "$MODELS_DIR"



# --- Hugging Face login (non-interactive) ---
if [ -n "${HF_TOKEN:-}" ]; then
  # strip accidental CR/LF from Windows .env
  export HF_TOKEN="$(printf %s "$HF_TOKEN" | tr -d '\r\n')"
  huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

# download model once
if [ -n "${HF_MODEL_REPO:-}" ]; then
  DEST="${MODELS_DIR}/${HF_MODEL_ALIAS:-default}"
  if [ ! -d "$DEST" ]; then
    echo "[entrypoint] Downloading ${HF_MODEL_REPO} -> ${DEST}"
    python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo  = os.environ["HF_MODEL_REPO"]
dest  = os.path.join(os.environ.get("MODELS_DIR","/models"),
                     os.environ.get("HF_MODEL_ALIAS","default"))
token = os.environ.get("HF_TOKEN")
snapshot_download(repo_id=repo, local_dir=dest,
                  local_dir_use_symlinks=False, token=token)
print("Downloaded to", dest)
PY
  else
    echo "[entrypoint] Model already present at ${DEST}"
  fi
fi

# migrate
python manage.py migrate --noinput

exec "$@"
