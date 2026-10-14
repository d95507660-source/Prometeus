"""
FastAPI-приложение бэкенда Content Repurposing Engine.

- При старте запускаются два фоновых цикла (см. worker.py):
    Worker  — забирает задачи из Supabase и обрабатывает их;
    Cleaner — раз в 5 минут переводит зависшие задачи в error.
- GET  /health        — проверка, что сервер жив (её дёргает Railway).
- POST /test/process  — отладочная ручка: ссылка → транскрипт → 5 твитов,
                        без Supabase. Оставлена для ручной отладки пайплайна.
"""

# .env загружаем ДО импорта сервисов, чтобы переменные окружения были доступны.
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from services.downloader import check_video  # noqa: E402
from services.generator import generate_tweets  # noqa: E402
from services.supabase_client import get_client, is_configured  # noqa: E402
from services.transcriber import get_transcript  # noqa: E402
from worker import Cleaner, Worker, error_to_message  # noqa: E402

worker = Worker()
cleaner = Cleaner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Код до yield выполняется при старте сервера, после yield — при остановке.
    (lifespan — современная замена устаревших @app.on_event("startup"/"shutdown").)
    """
    # --- Старт ---
    if is_configured():
        get_client()  # создаём клиента сразу: ошибка в ключах будет видна в логах при старте
        worker.start()
        cleaner.start()
    else:
        # Решение: без Supabase сервер всё равно стартует — чтобы /test/process
        # работал для отладки. /health покажет worker_running: false.
        print("[main] SUPABASE_URL / SUPABASE_SERVICE_KEY не заданы — воркер НЕ запущен")

    yield

    # --- Остановка ---
    await worker.stop()
    await cleaner.stop()


app = FastAPI(title="Content Repurposing Engine — backend", lifespan=lifespan)

# CORS: разрешаем запросы с любых сайтов (пока MVP).
# Фронтенд работает с Supabase напрямую, к бэкенду ходит только Railway health-check.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Пайплайн для отладочной ручки
# ---------------------------------------------------------------------------


def process_video(url: str) -> dict:
    """
    Полный пайплайн без Supabase:
    1. Проверяем видео (длина, не стрим, не плейлист) — получаем метаданные.
    2. Получаем транскрипт (субтитры, иначе Whisper).
    3. Генерируем 5 твитов через Claude.
    """
    print(f"[pipeline] Старт: {url}")
    info = check_video(url)
    transcript, source = get_transcript(url, info)
    tweets = generate_tweets(transcript)
    print(f"[pipeline] Готово: {url} (источник транскрипта: {source})")
    return {
        "transcript": transcript,
        "source": source,
        "duration_sec": float(info.get("duration") or 0),
        "tweets": tweets,
    }


# ---------------------------------------------------------------------------
# HTTP-ручки
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    """Railway дёргает эту ручку, чтобы понять, что сервис жив."""
    return {"status": "ok", "worker_running": worker.running}


@app.post("/test/process")
def test_process(request: ProcessRequest):
    """
    Отладочная ручка: синхронно обрабатывает видео и возвращает результат.

    Функция объявлена через обычный `def`, а не `async def`: FastAPI запускает
    её в отдельном потоке, и долгая обработка не блокирует сервер.
    """
    try:
        return process_video(request.url)
    except Exception as e:
        status, message = error_to_message(e)
        return JSONResponse(status_code=status, content={"error": message})
