#!/usr/bin/env bash
set -euo pipefail

/app/download_models.sh

# --warm: модели грузятся заранее; остальные аргументы (например --serve) передаются как есть
exec python /app/model_server.py --warm "$@"
