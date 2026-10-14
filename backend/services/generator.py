"""
Генерация 5 твитов из транскрипта через Claude API.

Надёжность:
- Ответ Claude парсим как JSON; если он всё-таки обернул его в ```json — вырезаем.
- Каждый твит проверяем через Pydantic (все 4 поля — непустые строки).
- full_text длиннее 280 символов обрезаем до 277 + "...".
- Если ответ не распарсился — делаем ОДНУ повторную попытку
  (модели иногда ошибаются в формате, вторая попытка почти всегда успешна).
"""

import json
import os
import re

import anthropic
from pydantic import BaseModel, ValidationError, field_validator

from prompts.tweets_prompt import TWEETS_SYSTEM_PROMPT, TWEETS_USER_TEMPLATE

# Модель можно поменять через .env, не трогая код.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

TWEET_MAX_LEN = 280
TWEETS_COUNT = 5

# Сколько раз пробуем получить корректный ответ.
MAX_ATTEMPTS = 2


class GenerationError(ValueError):
    """Не удалось сгенерировать твиты (ошибка API или некорректный ответ модели)."""


class TweetSchema(BaseModel):
    """Схема одного твита. Pydantic проверит, что все поля есть и это строки."""

    hook: str
    body: str
    cta: str
    full_text: str

    @field_validator("hook", "body", "cta", "full_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        # Убираем лишние пробелы по краям
        return value.strip()

    @field_validator("full_text")
    @classmethod
    def limit_length(cls, value: str) -> str:
        # Жёсткий лимит Twitter. Если модель превысила — обрезаем аккуратно.
        if not value:
            raise ValueError("full_text пустой")
        if len(value) > TWEET_MAX_LEN:
            print(f"[generator] full_text {len(value)} символов > {TWEET_MAX_LEN}, обрезаем")
            value = value[: TWEET_MAX_LEN - 3].rstrip() + "..."
        return value


def _anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise GenerationError("Не задан ANTHROPIC_API_KEY.")
    # max_retries=2 — SDK сам повторит запрос при 429/5xx и сетевых ошибках.
    return anthropic.Anthropic(api_key=api_key, timeout=120, max_retries=2)


def _extract_json(text: str):
    """
    Пытаемся достать JSON из ответа модели.
    1) Сначала просто json.loads().
    2) Если не вышло — ищем блок ```json ... ``` (markdown-обёртку).
    3) Если и так нет — берём кусок от первой [ до последней ].
    Если ничего не сработало — ValueError.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("Ответ модели не содержит корректного JSON")


def _parse_tweets(raw_text: str) -> list[dict]:
    """Парсит и валидирует ответ модели. Возвращает список из 5 словарей."""
    data = _extract_json(raw_text)

    # Иногда модель возвращает {"tweets": [...]} вместо массива — принимаем и так.
    if isinstance(data, dict):
        data = data.get("tweets")
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-массив")

    if len(data) < TWEETS_COUNT:
        raise ValueError(f"Модель вернула {len(data)} твитов вместо {TWEETS_COUNT}")

    # Если твитов больше 5 — берём первые 5.
    tweets = [TweetSchema.model_validate(item).model_dump() for item in data[:TWEETS_COUNT]]
    return tweets


def _max_transcript_chars() -> int:
    return int(os.getenv("MAX_TRANSCRIPT_CHARS", "60000"))


def generate_tweets(transcript: str) -> list[dict]:
    """
    Главная функция: транскрипт → список из 5 твитов
    [{hook, body, cta, full_text}, ...].
    """
    transcript = transcript.strip()
    if not transcript:
        raise GenerationError("Пустой транскрипт — не из чего делать твиты.")

    # Решение: ограничиваем длину транскрипта (по умолчанию 60 000 символов —
    # это примерно час речи). Для 5 твитов этого с запасом хватает, а платим
    # за каждый токен. Обрезку пишем в лог, чтобы было видно.
    limit = _max_transcript_chars()
    if len(transcript) > limit:
        print(f"[generator] Транскрипт {len(transcript)} символов, обрезаем до {limit}")
        transcript = transcript[:limit]

    client = _anthropic_client()
    model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
    user_message = TWEETS_USER_TEMPLATE.format(transcript=transcript)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,  # 5 твитов в JSON — это ~1500 токенов, берём с запасом
                system=TWEETS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.RateLimitError as e:
            print(f"[generator] Claude rate limit: {e}")
            raise GenerationError("Сервис генерации перегружен. Попробуйте позже.")
        except anthropic.APIStatusError as e:
            print(f"[generator] Claude HTTP {e.status_code}: {e}")
            raise GenerationError("Ошибка сервиса генерации твитов.")
        except anthropic.APIConnectionError as e:
            print(f"[generator] Нет связи с Anthropic: {e}")
            raise GenerationError("Нет связи с сервисом генерации твитов.")

        # Ответ упёрся в лимит токенов — JSON будет оборван.
        if response.stop_reason == "max_tokens":
            last_error = ValueError("Ответ модели оборвался (max_tokens)")
            print(f"[generator] Попытка {attempt}: {last_error}")
            continue

        # Собираем весь текст из ответа (обычно это один текстовый блок)
        raw_text = "".join(block.text for block in response.content if block.type == "text")

        try:
            tweets = _parse_tweets(raw_text)
            print(f"[generator] Готово: {len(tweets)} твитов (попытка {attempt}, модель {model})")
            return tweets
        except (ValueError, ValidationError) as e:
            # ValidationError от Pydantic — тоже подкласс ValueError, но указываем явно для ясности
            last_error = e
            print(f"[generator] Попытка {attempt}: не удалось разобрать ответ: {e}")
            print(f"[generator] Начало ответа: {raw_text[:300]!r}")

    raise GenerationError(f"Модель вернула некорректный ответ: {last_error}")
