"""forge_stage.py — этап forge через очередь SQLite: enqueue → параллельно claim / run / complete.

Вызов из orchestrator.py (вместо синхронного цикла по устройствам):
    from forge_stage import stage_forge
    code = stage_forge(cfg, registry, resume=args.resume, device_ids=args.device)
    if code:
        sys.exit(code)

Отдельно:  python forge_stage.py [--resume] [--reset-budget] [--device ID ...] [--day YYYY-MM-DD]

Поведение:
  * обычный запуск — ставит в очередь задачи дня (повторно уже существующие не добавляются),
    затем выполняет всё незавершённое за этот день;
  * --resume — ничего не добавляет, доделывает незавершённое за все дни;
  * упавшие задачи повторяются до forge.retry_max_attempts с паузой backoff·2^(n-1);
  * лог прогресса каждые 10 завершений: "10/60 done, 2 failed";
  * доля окончательно упавших > forge.stop_on_failure_ratio → новые задачи не берутся, код 3;
  * бюджет (budget_guard, если в config.json есть блок budget): перед запуском Cloud Run Job
    резервируется оценка (сцены × среднее время на сцену × $/час / 3600), после — пишется факт
    (длительность выполнения из GCP, иначе — время вызова). Лимит месяца исчерпан → задача
    возвращается в очередь без списания попытки, новые не берутся, код 4. Следующий запуск
    продолжит в новом месяце биллинга или после --reset-budget;
  * Ctrl+C — новые задачи не берутся, текущие дожидаются (код 130, очередь сохранена);
    второй Ctrl+C — текущие задачи возвращаются в очередь и процесс выходит сразу.
    ВНИМАНИЕ: уже запущенное выполнение Cloud Run при этом продолжает работать в GCP;
    после --resume задача запустится снова (имя прошлого выполнения — в jobs.gcp_execution_name).

Коды выхода: 0 — успех, 3 — порог падений превышен, 4 — пауза по бюджету, 130 — прервано.
"""
from __future__ import annotations

import logging
import os
import re
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

from job_queue import Job, JobQueue, account_age_days, priority_for_age

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE_RATIO = 3
EXIT_BUDGET_PAUSED = 4
EXIT_INTERRUPTED = 130
SUCCESS_STATUSES = ("ok", "skipped")
NON_RETRIABLE_STATUSES = ("realism_rejected",)  # детерминированы: повтор даст то же самое
NO_GPU_STATUSES = ("skipped", "realism_rejected")  # Cloud Run Job не запускался — трат нет


@dataclass
class Outcome:
    ok: bool
    error: str | None = None
    retriable: bool = True
    gcp_execution_name: str | None = None
    paused: bool = False  # задача не запускалась (бюджет) — вернуть в очередь без попытки


@dataclass
class RunSettings:
    max_workers: int = 8
    stop_on_failure_ratio: float = 0.2
    retry_backoff_sec: float = 60.0
    heartbeat_sec: float = 30.0
    stale_after_sec: float = 180.0
    poll_sec: float = 2.0
    progress_every: int = 10


@dataclass
class RunReport:
    exit_code: int
    stats: dict[str, int]
    processed: int
    interrupted: bool
    ratio_exceeded: bool
    budget_paused: bool = False


class QueueRunner:
    """Пул исполнителей поверх JobQueue. process(job) → Outcome, исключения ловятся."""

    def __init__(self, queue: JobQueue, process: Callable[[Job], Outcome], settings: RunSettings,
                 day: str | None = None, scope_days: list[str] | None = None,
                 stop_event: threading.Event | None = None):
        self.q = queue
        self.process = process
        self.s = settings
        self.day = day
        self.scope_days = [day] if day else list(scope_days or [])
        self.stop = stop_event or threading.Event()
        self.interrupted = False
        self.ratio_exceeded = False
        self.budget_paused = False
        self._lock = threading.Lock()
        self._processed = 0
        host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())[:40] or "host"
        run_id = uuid.uuid4().hex[:8]
        self.worker_ids = [f"{host}-{os.getpid()}-{run_id}-w{i}"
                           for i in range(max(1, int(settings.max_workers)))]

    # --- статистика ----------------------------------------------------------

    def _scope_stats(self) -> dict[str, int]:
        if not self.scope_days:
            return self.q.stats()
        total: dict[str, int] = {}
        for d in self.scope_days:
            for k, v in self.q.stats(d).items():
                total[k] = total.get(k, 0) + v
        return total

    def _after_complete(self) -> None:
        with self._lock:
            self._processed += 1
            n = self._processed
        st = self._scope_stats()
        if n % max(1, self.s.progress_every) == 0:
            log.info("%d/%d done, %d failed", st["done"], st["total"], st["failed_final"])
        if st["total"] and st["failed_final"] / st["total"] > self.s.stop_on_failure_ratio:
            if not self.ratio_exceeded:
                log.error("окончательно упало %d из %d (> %.0f%%) — новые задачи не беру",
                          st["failed_final"], st["total"], self.s.stop_on_failure_ratio * 100)
            self.ratio_exceeded = True
            self.stop.set()

    def _pause_for_budget(self, job: Job) -> None:
        self.q.release(job.id)  # pending, попытка не засчитана
        with self._lock:
            first = not self.budget_paused
            self.budget_paused = True
        if first:
            log.warning("лимит бюджета исчерпан — новые задачи не запускаю, они остаются в очереди. "
                        "Продолжение: новый месяц биллинга или --reset-budget")
        self.stop.set()

    # --- исполнитель ---------------------------------------------------------

    def _run_one(self, job: Job) -> Outcome:
        try:
            out = self.process(job)
            return out if isinstance(out, Outcome) else Outcome(False, f"process вернул {out!r}")
        except Exception as e:
            log.exception("задача %d (%s/%s/source_%d) упала", job.id, job.device_id, job.day,
                          job.source_idx)
            return Outcome(False, f"{type(e).__name__}: {e}")

    def _worker(self, worker_id: str) -> None:
        self.q.register_worker(worker_id)
        try:
            while not self.stop.is_set():
                job = self.q.claim(worker_id, day=self.day)
                if job is None:
                    if self.q.retry_failed(backoff_sec=self.s.retry_backoff_sec, day=self.day):
                        continue
                    if self.q.has_unfinished(self.day):
                        # есть отложенные повторы или задачи у других исполнителей
                        self.stop.wait(self.s.poll_sec)
                        continue
                    return
                log.info("[%s] задача %d: %s/%s/source_%d (приоритет %d, попытка %d/%d)",
                         worker_id.rsplit("-", 1)[-1], job.id, job.device_id, job.day,
                         job.source_idx, job.priority, job.attempts, job.max_attempts)
                out = self._run_one(job)
                if out.paused:
                    self._pause_for_budget(job)
                    continue
                recorded = self.q.complete(job.id, "done" if out.ok else "failed", error=out.error,
                                           gcp_execution_name=out.gcp_execution_name,
                                           worker_id=worker_id, retriable=out.retriable)
                if not recorded:
                    log.warning("задача %d: результат не записан (её уже вернули в очередь)", job.id)
                if not out.ok:
                    log.warning("задача %d: ошибка (%s): %s", job.id,
                                "будет повтор" if out.retriable and job.attempts < job.max_attempts
                                else "окончательно", (out.error or "")[:300])
                self._after_complete()
        finally:
            self.q.unregister_worker(worker_id)

    def _heartbeat(self, done: threading.Event) -> None:
        while not done.wait(self.s.heartbeat_sec):
            try:
                self.q.heartbeat(self.worker_ids)
                self.q.recover_stale(self.s.stale_after_sec)
            except Exception as e:  # БД занята и т.п. — следующий тик повторит
                log.warning("heartbeat: %s", e)

    @staticmethod
    def _wait_all(futs: list[Future]) -> None:
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_EXCEPTION)
            for f in done:
                if f.exception() is not None:
                    log.error("исполнитель упал: %r", f.exception())

    def request_stop(self) -> None:
        """То же, что первый Ctrl+C: новые задачи не брать, текущие доделать."""
        self.interrupted = True
        self.stop.set()

    def run(self) -> RunReport:
        recovered = self.q.recover_stale(self.s.stale_after_sec)
        if recovered:
            log.warning("возвращено в очередь задач от упавшего процесса: %d", recovered)
        hb_done = threading.Event()
        hb = threading.Thread(target=self._heartbeat, args=(hb_done,), name="queue-heartbeat",
                              daemon=True)
        hb.start()
        started = time.time()
        try:
            with ThreadPoolExecutor(max_workers=len(self.worker_ids),
                                    thread_name_prefix="forge-q") as ex:
                futs = [ex.submit(self._worker, w) for w in self.worker_ids]
                try:
                    self._wait_all(futs)
                except KeyboardInterrupt:
                    log.warning("Ctrl+C: новые задачи не беру, жду текущие. "
                                "Ещё раз Ctrl+C — выйти сразу (задачи вернутся в очередь)")
                    self.request_stop()
                    try:
                        self._wait_all(futs)
                    except KeyboardInterrupt:
                        n = self.q.release_worker_jobs(self.worker_ids)
                        log.warning("выход без ожидания, возвращено в очередь: %d. "
                                    "Продолжить: --resume", n)
                        logging.shutdown()
                        os._exit(EXIT_INTERRUPTED)  # потоки ждут Cloud Run — join их не дождётся
        finally:
            hb_done.set()
        st = self._scope_stats()
        log.info("очередь: %d/%d done, %d failed (окончательно), %d pending, за %.0f с",
                 st["done"], st["total"], st["failed_final"], st["pending"], time.time() - started)
        if self.ratio_exceeded:
            code = EXIT_FAILURE_RATIO
        elif self.interrupted:
            code = EXIT_INTERRUPTED
            log.warning("остановлено, очередь сохранена. Продолжить: --resume")
        elif self.budget_paused:
            code = EXIT_BUDGET_PAUSED
        else:
            code = EXIT_OK
        return RunReport(code, st, self._processed, self.interrupted, self.ratio_exceeded,
                         self.budget_paused)


# ---------------------------------------------------------------------------
# Этап forge
# ---------------------------------------------------------------------------

def settings_from_config(cfg: Any) -> RunSettings:
    f = cfg.forge
    return RunSettings(max_workers=int(f.max_parallel_jobs),
                       stop_on_failure_ratio=float(f.stop_on_failure_ratio),
                       retry_backoff_sec=float(f.retry_backoff_sec))


def max_scenes(cfg: Any) -> int:
    """Оценка числа сцен сверху: forge выбирает от scenes_per_video[0] до [1], не больше,
    чем помещается в video_duration_sec. Точное число известно только после сценария (LLM)."""
    f = cfg.forge
    by_duration = max(1, max(f.video_duration_sec) // max(1, f.scene_duration_sec))
    return max(1, min(int(f.scenes_per_video[1]), by_duration))


def _video_scenes(res: Any) -> int | None:
    scenes = getattr(res, "scenes", None) or []
    n = sum(1 for s in scenes if getattr(s, "role", "scene") != "master")
    return n or None


def make_process(forge: Any, devices_by_id: dict[str, dict], guard: Any = None,
                 n_scenes_estimate: int = 1) -> Callable[[Job], Outcome]:
    def process(job: Job) -> Outcome:
        device = devices_by_id.get(job.device_id)
        if device is None:
            return Outcome(False, "устройство неактивно или отброшено реестром", retriable=False)
        key = f"job{job.id}.a{job.attempts}"  # повтор — отдельный запуск и отдельная трата
        if guard is not None:
            est = guard.estimate_cost(n_scenes_estimate)
            if not guard.try_reserve(key, est):
                return Outcome(False, "лимит бюджета", paused=True)
        t0 = time.monotonic()
        try:
            res = forge.generate_one(device, job.source_idx, job.day)
        except BaseException:
            if guard is not None:  # не знаем, успел ли запуститься Cloud Run — считаем по времени
                sec = time.monotonic() - t0
                guard.record_spend(guard.cost_for_seconds(sec), key)
            raise
        status = getattr(res, "status", "unknown")
        if guard is not None:
            if status in NO_GPU_STATUSES:
                guard.release(key)
            else:
                gpu_sec = getattr(res, "gpu_seconds", None)
                source = "GCP"
                if not isinstance(gpu_sec, (int, float)) or gpu_sec <= 0:
                    gpu_sec, source = time.monotonic() - t0, "время вызова"
                guard.record_spend(guard.cost_for_seconds(gpu_sec), key, gpu_sec=gpu_sec,
                                   scenes=_video_scenes(res) or n_scenes_estimate)
                log.debug("задача %d: %.0f с GPU (%s)", job.id, gpu_sec, source)
        ok = status in SUCCESS_STATUSES
        return Outcome(ok=ok, error=None if ok else (getattr(res, "error", None) or status),
                       retriable=status not in NON_RETRIABLE_STATUSES,
                       gcp_execution_name=getattr(res, "gcp_execution_name", None))
    return process


def stage_forge(cfg: Any, registry: Any = None, resume: bool = False,
                device_ids: list[str] | None = None, day: str | None = None,
                forge: Any = None, settings: RunSettings | None = None,
                stop_event: threading.Event | None = None, reset_budget: bool = False,
                guard: Any = None) -> int:
    """Возвращает код выхода (0 / 3 / 4 / 130). forge, settings, stop_event, guard — для тестов."""
    if registry is None:
        from devices_registry import DevicesRegistry
        registry = DevicesRegistry.from_config(cfg)
    local_gpu = getattr(getattr(cfg, "gcp", None), "mode", "cloud") == "local"
    if guard is None and not local_gpu:
        from budget_guard import guard_for
        guard = guard_for(cfg)
    elif guard is None and getattr(getattr(cfg, "budget", None), "enabled", False):
        log.info("gcp.mode=local: облачный бюджет не применяется (генерация на своём GPU)")
    if reset_budget:
        if guard is None:
            log.warning("--reset-budget: в config.json нет блока budget — нечего сбрасывать")
        else:
            guard.reset("--reset-budget")
    queue = JobQueue(cfg.queue_db_path)
    log.info("очередь: %s", cfg.queue_db_path)
    # все активные — чтобы при --resume доделать задачи и тех устройств, что не в --device
    devices_by_id = {d["device_id"]: d for d in registry.get_active_devices(device_ids=None)}

    if resume:
        if device_ids:
            log.warning("--resume доделывает всю очередь, --device игнорируется")
        scope_days = queue.unfinished_days()
        run_day = None
        log.info("resume: незавершённые дни %s", scope_days or "нет")
    else:
        run_day = day or cfg.today()
        scope_days = [run_day]
        devices = registry.get_active_devices(device_ids=device_ids)
        from account_health import health_for
        health = health_for(cfg)
        if health is not None:
            banned = [d["device_id"] for d in devices if not health.device_usable(d)]
            if banned:
                log.warning("forge: устройства со всеми аккаунтами в бане не генерируются: %s", banned)
                devices = [d for d in devices if d["device_id"] not in banned]
        if not devices:
            log.warning("forge: нет активных устройств")
        items = []
        for d in devices:
            prio = priority_for_age(account_age_days(queue, d, run_day), cfg.forge.new_account_days)
            for idx in range(1, cfg.forge.videos_per_device_per_day + 1):
                items.append((d["device_id"], run_day, idx, prio))
        added = queue.enqueue_many(items, max_attempts=cfg.forge.retry_max_attempts)
        log.info("очередь %s: добавлено %d, уже было %d", run_day, added, len(items) - added)

    n_scenes = max_scenes(cfg)
    if guard is not None:
        st = guard.status()
        est = guard.estimate_cost(n_scenes)
        log.info("бюджет %s: потрачено $%.2f из $%.2f (резерв $%.2f), оценка задачи $%.3f",
                 st["month"], st["spent_usd"], st["limit_usd"], st["reserved_usd"], est)
        if not guard.can_spend(est):
            log.warning("лимит бюджета исчерпан — задачи остаются в очереди. Продолжение: новый "
                        "месяц биллинга или --reset-budget")
            return EXIT_BUDGET_PAUSED

    if forge is None:
        from forge import Forge
        forge = Forge(cfg, registry)
    runner = QueueRunner(queue, make_process(forge, devices_by_id, guard, n_scenes),
                         settings or settings_from_config(cfg), day=run_day,
                         scope_days=scope_days, stop_event=stop_event)
    report = runner.run()
    # внешний stop_event (тесты, web_panel) трактуем как Ctrl+C
    if (stop_event is not None and stop_event.is_set() and not report.ratio_exceeded
            and not report.budget_paused):
        report.exit_code = EXIT_INTERRUPTED
        log.warning("остановлено извне, очередь сохранена. Продолжить: --resume")
    return report.exit_code


def main() -> None:
    from config_loader import cli_config

    def extra(ap):
        ap.add_argument("--device", action="append", default=None)
        ap.add_argument("--day", default=None, help="YYYY-MM-DD (по умолчанию сегодня)")
        ap.add_argument("--resume", action="store_true",
                        help="продолжить незавершённую очередь (без постановки новых задач)")
        ap.add_argument("--reset-budget", action="store_true",
                        help="обнулить трату текущего месяца в data/budget.json перед запуском")

    cfg, a = cli_config("forge_stage", extra=extra)
    sys.exit(stage_forge(cfg, resume=a.resume, device_ids=a.device, day=a.day,
                         reset_budget=a.reset_budget))


if __name__ == "__main__":
    main()
