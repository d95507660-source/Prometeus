FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements-gpu.txt

COPY model_server.py download_models.py download_models.sh entrypoint.sh /app/
RUN chmod +x /app/download_models.sh /app/entrypoint.sh \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /models \
    && chown -R app:app /models /app

USER app

ENV MODELS_DIR=/models \
    HF_HOME=/models/hf_cache

ENTRYPOINT ["/app/entrypoint.sh"]
