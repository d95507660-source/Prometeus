"""
FastAPI-приложение бэкенда Content Repurposing Engine.

Сейчас здесь:
- GET  /health        — проверка, что сервер жив (её дёргает Railway).
- POST /test/process  — ТЕСТОВАЯ ручка: ссылка → транскрипт → 5 твитов.

Вся логика обработки вынесена в функцию process_video(url).
На следующем шаге фоновый цикл (опрос Supabase) будет вызывать
ту же самую process_video() — ручку /test/process можно будет просто удалить.
"""

# .env загружаем ДО импорта сервисов, чтобы переменные окружения были доступны.
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from services.downloader import VideoError, check_video  # noqa: E402
from services.generator import GenerationError, generate_tweets  # noqa: E402
from services.transcriber import TranscriptionError, get_transcript  # noqa: E402

app = FastAPI(title="Content Repurposing Engine — backend")

# CORS: разрешаем запросы с любых сайтов — ТОЛЬКО для локального теста.
# Когда будет фронтенд, здесь нужно оставить только его домен.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Основной пайплайн — его же будет вызывать фоновый воркер
# ---------------------------------------------------------------------------


def process_video(url: str) -> dict:
    """
    Полный пайплайн обработки одного видео:
    1. Проверяем видео (длина, не стрим, не плейлист) — получаем метаданные.
    2. Получаем транскрипт (субтитры, иначе Whisper).
    3. Генерируем 5 твитов через Claude.

    Возвращает словарь с результатом. Ошибки НЕ ловит — пусть их обрабатывает
    вызывающий код (ручка API или фоновый воркер), каждый по-своему.
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


def error_to_message(err: Exception) -> tuple[int, str]:
    """
    Превращает исключение в (HTTP-код, понятное сообщение).
    Эту же функцию воркер будет использовать для поля jobs.error_message.
    """
    if isinstance(err, VideoError):
        return 422, str(err)  # проблема с видео/ссылкой — пользователь может исправить
    if isinstance(err, (TranscriptionError, GenerationError)):
        return 502, str(err)  # сбой внешнего сервиса (OpenAI/Anthropic)
    # Неожиданная ошибка — подробности только в лог, пользователю общий текст
    print(f"[pipeline] Неожиданная ошибка: {type(err).__name__}: {err}")
    return 500, "Внутренняя ошибка сервера. Попробуйте ещё раз."


# ---------------------------------------------------------------------------
# HTTP-ручки
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    """Railway дёргает эту ручку, чтобы понять, что сервис жив."""
    return {"status": "ok"}


@app.post("/test/process")
def test_process(request: ProcessRequest):
    """
    Тестовая ручка: синхронно обрабатывает видео и возвращает результат.

    ВАЖНО: функция объявлена через обычный `def`, а не `async def`.
    Так FastAPI запускает её в отдельном потоке, и долгая обработка
    (скачивание, Whisper — до нескольких минут) не блокирует сервер:
    /health продолжает отвечать.
    """
    try:
        return process_video(request.url)
    except Exception as e:
        status, message = error_to_message(e)
        return JSONResponse(status_code=status, content={"error": message})
