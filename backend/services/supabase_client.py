"""
Клиент Supabase для бэкенда (singleton — один на всё приложение).

Используем SERVICE KEY (service_role): он обходит RLS и может вызывать
служебные RPC (claim_job, release_failed_job, cleanup_stuck_jobs).
Этот ключ — полный доступ к базе. Он живёт ТОЛЬКО на сервере (Railway)
и никогда не должен попасть во фронтенд или в git.
"""

import os
import threading

from supabase import Client, create_client

_client: Client | None = None
# Замок нужен, потому что get_client() могут вызвать одновременно
# из нескольких потоков (воркер обрабатывает задачи в отдельном потоке).
_lock = threading.Lock()


class SupabaseConfigError(RuntimeError):
    """Не заданы переменные окружения для подключения к Supabase."""


def is_configured() -> bool:
    """Заданы ли переменные для подключения (без создания клиента)."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))


def get_client() -> Client:
    """Возвращает общий клиент Supabase, создавая его при первом вызове."""
    global _client
    if _client is not None:
        return _client

    with _lock:
        # Повторная проверка: пока мы ждали замок, клиент мог создать другой поток
        if _client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")
            if not url or not key:
                raise SupabaseConfigError("Не заданы SUPABASE_URL и/или SUPABASE_SERVICE_KEY")
            _client = create_client(url, key)
    return _client
