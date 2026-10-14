"""
Транскрипция видео.

Стратегия (от дешёвого к дорогому):
1. Субтитры YouTube — бесплатно и мгновенно.
2. Если субтитров нет — скачиваем аудио и отправляем в OpenAI Whisper API.

Решение: субтитры пробуем ВСЕГДА, а не только на «первых 3 видео».
Так дешевле навсегда, а качество субтитров автора обычно не хуже Whisper.
"""

import os

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from services import downloader


class TranscriptionError(RuntimeError):
    """Ошибка распознавания речи (проблема на стороне Whisper/OpenAI)."""


def _openai_client() -> OpenAI:
    """Создаём клиента OpenAI. Ключ берётся из переменной окружения."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise TranscriptionError("Не задан OPENAI_API_KEY — Whisper недоступен.")
    # timeout=300 — длинное аудио распознаётся до нескольких минут.
    # max_retries=2 — SDK сам повторит запрос при сетевой ошибке или 429/5xx.
    return OpenAI(api_key=api_key, timeout=300, max_retries=2)


def transcribe(audio_path: str) -> str:
    """Отправляет аудиофайл в Whisper API (модель whisper-1) и возвращает текст."""
    client = _openai_client()
    try:
        with open(audio_path, "rb") as f:
            # Язык не указываем — Whisper определит его сам (ru, en и др.)
            result = client.audio.transcriptions.create(model="whisper-1", file=f)
    except RateLimitError as e:
        print(f"[transcriber] Whisper rate limit / нет денег на балансе: {e}")
        raise TranscriptionError("Сервис распознавания речи перегружен. Попробуйте позже.")
    except APIStatusError as e:
        print(f"[transcriber] Whisper HTTP {e.status_code}: {e}")
        raise TranscriptionError("Ошибка сервиса распознавания речи.")
    except APIConnectionError as e:
        print(f"[transcriber] Нет связи с OpenAI: {e}")
        raise TranscriptionError("Нет связи с сервисом распознавания речи.")

    text = (result.text or "").strip()
    if not text:
        raise TranscriptionError("В видео не удалось распознать речь.")
    print(f"[transcriber] Whisper вернул {len(text)} символов")
    return text


def get_transcript(url: str, info: dict | None = None) -> tuple[str, str]:
    """
    Главная функция: возвращает (текст_транскрипта, источник).
    Источник — 'subtitles' или 'whisper'.

    info — результат downloader.check_video(). Если передать, не будет
    повторного запроса метаданных к YouTube (быстрее и меньше шанс блокировки).
    """
    if info is None:
        info = downloader.check_video(url)

    # Шаг 1: пробуем субтитры
    subtitles = downloader.get_subtitles(url, info)
    if subtitles:
        return subtitles, "subtitles"

    # Шаг 2: скачиваем аудио и распознаём через Whisper
    audio_path = None
    try:
        audio_path, _duration = downloader.download_audio(url, info)
        text = transcribe(audio_path)
        return text, "whisper"
    finally:
        # Временный файл удаляем ВСЕГДА — даже если Whisper упал с ошибкой.
        # Иначе диск сервера постепенно забьётся.
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"[transcriber] Временный файл удалён: {audio_path}")
