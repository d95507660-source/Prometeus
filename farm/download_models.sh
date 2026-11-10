#!/usr/bin/env bash
# Скачивает модели из gs://$GCS_MODELS_BUCKET/models/ в $MODELS_DIR.
# Если бакет смонтирован как том Cloud Run (MODELS_DIR уже заполнен) — ничего не качает.
set -euo pipefail

: "${GCS_MODELS_BUCKET:=prometheus-models}"
: "${MODELS_DIR:=/models}"
REQUIRED=(wan22_ti2v_5b xtts_v2 voices)

mkdir -p "$MODELS_DIR"

missing=0
for d in "${REQUIRED[@]}"; do
  [ -d "${MODELS_DIR}/${d}" ] || missing=1
done

if [ "$missing" -eq 1 ] || [ "${FORCE_MODEL_SYNC:-0}" = "1" ]; then
  echo "[download_models] gs://${GCS_MODELS_BUCKET}/models/ -> ${MODELS_DIR}/"
  # в образе нет gcloud/gsutil, поэтому качаем через google-cloud-storage
  python /app/download_models.py --bucket "$GCS_MODELS_BUCKET" --dest "$MODELS_DIR"
else
  echo "[download_models] модели уже на месте, пропуск"
fi

status=0
for d in "${REQUIRED[@]}"; do
  if [ ! -d "${MODELS_DIR}/${d}" ]; then
    echo "[download_models] ОШИБКА: нет ${MODELS_DIR}/${d}" >&2
    status=1
  fi
done
[ -d "${MODELS_DIR}/longcat_avatar" ] || echo "[download_models] нет longcat_avatar — будет наложение звука без липсинка"

if [ "$status" -ne 0 ] && [ "${ALLOW_MISSING_MODELS:-0}" != "1" ]; then
  exit 1
fi
echo "[download_models] готово"
