# Базовый образ: лёгкий Python 3.11
FROM python:3.11-slim

# PYTHONUNBUFFERED=1 — логи print() сразу видны в Railway, без задержки.
# PIP_NO_CACHE_DIR=1 — не храним кэш pip, образ получается меньше.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg нужен yt-dlp, чтобы конвертировать аудио в mp3.
# --no-install-recommends — не ставим лишние пакеты, образ меньше.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Deno — JavaScript-движок. С конца 2025 года yt-dlp без него
# не может полноценно скачивать с YouTube (нужно решать JS-задачки YouTube).
# Берём готовый бинарник из официального образа Deno.
COPY --from=denoland/deno:bin-2.9.5 /deno /usr/local/bin/deno

WORKDIR /app

# Сначала копируем только requirements.txt и ставим зависимости.
# Так Docker закэширует этот слой, и пересборка после правки кода будет быстрой.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Теперь копируем весь код
COPY . .

# Railway сам передаёт порт в переменной $PORT.
# Если её нет (локальный запуск Docker) — используем 8000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
