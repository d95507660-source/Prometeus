"""
Фоновый воркер: забирает задачи из Supabase и обрабатывает их.

Как это работает:
- Worker каждые 5 секунд вызывает RPC claim_job (атомарный захват задачи).
  Нашлась задача → обрабатываем, нет → ждём 5 секунд.
- Cleaner каждые 5 минут вызывает RPC cleanup_stuck_jobs — переводит
  «зависшие» задачи (processing дольше 15 минут) в error.

Задачи обрабатываются строго по одной. Для MVP этого достаточно, а сервер
на Railway не упадёт от нехватки памяти из-за нескольких ffmpeg одновременно.
Если понадобится больше — можно запустить несколько копий сервиса:
claim_job гарантирует, что одну задачу не возьмут дважды.
"""

import asyncio

from services.downloader import VideoError
from services.generator import GenerationError, generate_tweets
from services.supabase_client import get_client
from services.transcriber import TranscriptionError, get_transcript

# Пауза между проверками очереди, когда задач нет
POLL_INTERVAL_SEC = 5
# Как часто запускается уборщик зависших задач
CLEANUP_INTERVAL_SEC = 5 * 60


def error_to_message(err: Exception) -> tuple[int, str]:
    """
    Превращает исключение в (HTTP-код, понятное сообщение для пользователя).
    Используется и воркером (поле jobs.error_message), и тестовой ручкой.

    Решение: для неожиданных ошибок пользователю показываем общий текст,
    а технические подробности (пути, имена переменных, трассировки) — только в лог.
    """
    if isinstance(err, VideoError):
        return 422, str(err)  # проблема с видео/ссылкой — пользователь может исправить
    if isinstance(err, (TranscriptionError, GenerationError)):
        return 502, str(err)  # сбой внешнего сервиса (OpenAI/Anthropic)
    print(f"[worker] Неожиданная ошибка: {type(err).__name__}: {err}")
    return 500, "Внутренняя ошибка сервера. Попробуйте ещё раз."


def process_job(job: dict) -> None:
    """
    Обрабатывает одну задачу. СИНХРОННАЯ функция — вызывается через
    asyncio.to_thread(), потому что yt-dlp, ffmpeg и HTTP-запросы блокирующие.

    Попытка одна, без повторов: при любой ошибке задача получает статус
    error, а пользователю возвращается попытка (RPC release_failed_job).
    """
    job_id = job["id"]
    url = job["url"]
    supabase = get_client()
    print(f"[worker] Задача {job_id}: старт, {url}")

    try:
        # 1-2. Транскрипт (субтитры или Whisper). Временный mp3 удаляет сам transcriber.
        transcript, source = get_transcript(url)
        print(f"[worker] Задача {job_id}: транскрипт {len(transcript)} симв. ({source})")

        # 3. Генерация 5 твитов
        tweets = generate_tweets(transcript)

        # 4. Сохраняем результат.
        # Условие status='processing': если уборщик уже пометил задачу как error
        # (обработка шла дольше 15 минут) — не перезаписываем, попытка уже возвращена.
        result = (
            supabase.table("jobs")
            .update(
                {
                    "status": "done",
                    "transcript": transcript,
                    "transcript_source": source,
                    "output_tweets": tweets,
                    "error_message": None,
                }
            )
            .eq("id", job_id)
            .eq("status", "processing")
            .execute()
        )
        if result.data:
            print(f"[worker] Задача {job_id}: готово")
        else:
            print(f"[worker] Задача {job_id}: результат не сохранён — задача уже не в processing")

    except Exception as e:
        _status, message = error_to_message(e)
        print(f"[worker] Задача {job_id}: ошибка — {type(e).__name__}: {e}")
        try:
            supabase.rpc(
                "release_failed_job",
                {"p_job_id": job_id, "p_error": message[:500]},
            ).execute()
            print(f"[worker] Задача {job_id}: статус error, попытка возвращена пользователю")
        except Exception as release_err:
            # Даже если не удалось записать ошибку (например, упала сеть до Supabase),
            # задачу через 15 минут подберёт уборщик.
            print(f"[worker] Задача {job_id}: не удалось записать ошибку в базу: {release_err}")


def _claim_job() -> dict | None:
    """Атомарно забирает одну pending-задачу. Возвращает dict или None."""
    result = get_client().rpc("claim_job").execute()
    # claim_job возвращает setof jobs → в Python это список из 0 или 1 элемента
    if result.data:
        return result.data[0]
    return None


def _cleanup_stuck_jobs() -> int:
    """Вызывает уборщика в базе. Возвращает количество очищенных задач."""
    result = get_client().rpc("cleanup_stuck_jobs").execute()
    return int(result.data or 0)


class _BackgroundLoop:
    """Общая часть Worker и Cleaner: запуск, остановка, флаг «работает»."""

    name = "loop"

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    def start(self) -> asyncio.Task:
        """Запускает бесконечный цикл как фоновую asyncio-задачу."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=self.name)
        return self._task

    async def stop(self) -> None:
        """Останавливает цикл (вызывается при выключении сервера)."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        raise NotImplementedError


class Worker(_BackgroundLoop):
    """Цикл обработки задач из очереди."""

    name = "worker"

    async def _run(self) -> None:
        print("[worker] Воркер запущен")
        while True:
            try:
                # Запрос к Supabase тоже блокирующий — выносим в поток,
                # чтобы /health отвечал без задержек.
                job = await asyncio.to_thread(_claim_job)
                if job is None:
                    await asyncio.sleep(POLL_INTERVAL_SEC)
                    continue

                # Обработка в отдельном потоке: yt-dlp/ffmpeg/Whisper не блокируют сервер.
                # После задачи сразу проверяем очередь снова — без паузы.
                #
                # Если сервер выключают посреди обработки, поток доработает
                # не всегда, и задача останется в processing — её через
                # 15 минут переведёт в error уборщик.
                await asyncio.to_thread(process_job, job)

            except asyncio.CancelledError:
                print("[worker] Воркер остановлен")
                raise
            except Exception as e:
                # Ошибка связи с Supabase и т.п. Не роняем цикл — ждём и пробуем снова.
                print(f"[worker] Ошибка цикла: {type(e).__name__}: {e}")
                await asyncio.sleep(POLL_INTERVAL_SEC)


class Cleaner(_BackgroundLoop):
    """Цикл уборки зависших задач."""

    name = "cleaner"

    async def _run(self) -> None:
        print("[cleaner] Уборщик запущен")
        while True:
            try:
                count = await asyncio.to_thread(_cleanup_stuck_jobs)
                if count:
                    print(f"[cleaner] Зависших задач переведено в error: {count}")
                else:
                    print("[cleaner] Зависших задач нет")
            except asyncio.CancelledError:
                print("[cleaner] Уборщик остановлен")
                raise
            except Exception as e:
                print(f"[cleaner] Ошибка: {type(e).__name__}: {e}")

            await asyncio.sleep(CLEANUP_INTERVAL_SEC)
