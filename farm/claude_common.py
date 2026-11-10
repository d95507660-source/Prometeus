"""claude_common.py — общий вызов Claude API для claude_analyst.py и claude_builder.py.

* модель claude-sonnet-5-5;
* ответ — строго JSON по схеме (structured outputs, output_config.format), без разбора markdown;
* temperature НЕ передаётся: на Claude Sonnet 5.5 любое значение, кроме значения по умолчанию,
  даёт 400 (а с API-версии 2026-10-21 temperature отклоняется на всех моделях). Объём рассуждений
  регулируется output_config.effort;
* повторы: SDK сам повторяет 408/409/429/5xx и сетевые ошибки с экспоненциальной задержкой —
  max_retries=3 на клиенте; свой цикл поверх не нужен;
* отказ классификатора (stop_reason="refusal") — включён серверный fallback
  (fallbacks="default", beta server-side-fallback-2026-07-01). Он повторяет на другой модели только
  отказы категорий cyber / frontier_llm; остальные возвращаются как ClaudeCallError.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SEC = 600.0


class ClaudeCallError(RuntimeError):
    """Запрос не дал пригодного ответа (ошибка API после повторов SDK, отказ, обрыв по max_tokens)."""

    def __init__(self, message: str, *, retriable: bool = False, request_id: str | None = None):
        super().__init__(message)
        self.retriable = retriable
        self.request_id = request_id


class ClaudeConfigError(ClaudeCallError):
    """Нет ключа или доступа — повтор не поможет."""


@dataclass
class CallMeta:
    model: str
    request_id: str | None
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    fallback_used: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def require_api_key() -> str:
    """Проверка на старте: ключ только из окружения, в лог не пишется."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise ClaudeConfigError("ANTHROPIC_API_KEY не задан")
    return key


def make_client(max_retries: int = DEFAULT_MAX_RETRIES, timeout: float = DEFAULT_TIMEOUT_SEC):
    import anthropic
    return anthropic.Anthropic(api_key=require_api_key(), max_retries=max_retries, timeout=timeout)


def _fallback_used(msg: Any) -> bool:
    usage = getattr(msg, "usage", None)
    for entry in (getattr(usage, "iterations", None) or []):
        if getattr(entry, "type", None) == "fallback_message":
            return True
    return False


def structured_call(client: Any, *, system: str, user: str, schema: dict, max_tokens: int,
                    effort: str = "medium", stream: bool = False) -> tuple[dict, CallMeta]:
    """Один запрос с JSON-схемой ответа. Возвращает (данные, метаданные)."""
    import anthropic

    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort,
                          "format": {"type": "json_schema", "schema": schema}},
    }
    try:
        if stream:  # длинный ответ: стрим не упирается в HTTP-таймауты
            with client.beta.messages.stream(**kwargs) as s:
                msg = s.get_final_message()
        else:
            msg = client.beta.messages.create(**kwargs)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise ClaudeConfigError(f"Claude API: доступ запрещён ({e.status_code})") from e
    except anthropic.BadRequestError as e:
        raise ClaudeCallError(f"Claude API: неверный запрос: {e.message}"[:500]) from e
    except anthropic.RateLimitError as e:
        raise ClaudeCallError("Claude API: лимит запросов (повторы SDK исчерпаны)", retriable=True) from e
    except anthropic.APIStatusError as e:
        raise ClaudeCallError(f"Claude API: HTTP {e.status_code} (повторы SDK исчерпаны)",
                              retriable=e.status_code >= 500) from e
    except anthropic.APIConnectionError as e:
        raise ClaudeCallError("Claude API: нет соединения (повторы SDK исчерпаны)", retriable=True) from e

    request_id = getattr(msg, "_request_id", None)
    usage = getattr(msg, "usage", None)
    meta = CallMeta(model=str(getattr(msg, "model", MODEL)), request_id=request_id,
                    stop_reason=getattr(msg, "stop_reason", None),
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                    fallback_used=_fallback_used(msg))
    if meta.stop_reason == "refusal":
        details = getattr(msg, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise ClaudeCallError(f"Claude отказался выполнить запрос (категория: {category})",
                              request_id=request_id)
    if meta.stop_reason == "max_tokens":
        raise ClaudeCallError(f"ответ обрезан на max_tokens={max_tokens}", request_id=request_id)
    text = next((b.text for b in (msg.content or []) if getattr(b, "type", None) == "text"), None)
    if text is None:
        raise ClaudeCallError(f"в ответе нет текста (stop_reason={meta.stop_reason})",
                              request_id=request_id)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ClaudeCallError(f"ответ не JSON: {e}", request_id=request_id) from e
    if not isinstance(data, dict):
        raise ClaudeCallError("ответ JSON не объект", request_id=request_id)
    log.info("Claude %s: request_id=%s in=%s out=%s fallback=%s", meta.model, request_id,
             meta.input_tokens, meta.output_tokens, meta.fallback_used)
    return data, meta
