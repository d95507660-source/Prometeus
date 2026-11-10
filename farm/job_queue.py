"""job_queue.py — очередь генерации в SQLite (data/queue.db).

Таблицы:
  jobs          — одна строка на (device_id, day, source_idx); повторный enqueue не дублирует.
  workers       — потоки-исполнители и их heartbeat (по нему находим «зависшие» задачи).
  devices_seen  — дата, когда устройство впервые попало в очередь (возраст для приоритета,
                  если в devices.json нет даты создания аккаунта).

Жизненный цикл задачи:
  pending ──claim──▶ running ──complete(done)──▶ done
                        │
                        └─complete(failed)──▶ failed ──retry_failed──▶ pending (с backoff)
  failed при attempts >= max_attempts — окончательно (retry_failed его не трогает).
  running без живого heartbeat (процесс убит) → recover_stale → pending.

Модуль не зависит от остального проекта: только stdlib.
Параллелизм: WAL + BEGIN IMMEDIATE — захват атомарен и между потоками, и между процессами.
"""
from __future__ import annotations

import logging
import os
import re
import socket
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WORKER_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")

STATUSES = ("pending", "running", "done", "failed")
PRIORITY_NEW = 10
PRIORITY_NORMAL = 5
NEW_ACCOUNT_DAYS = 14

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id          TEXT    NOT NULL,
    day                TEXT    NOT NULL,
    source_idx         INTEGER NOT NULL,
    priority           INTEGER NOT NULL DEFAULT 5,
    status             TEXT    NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts           INTEGER NOT NULL DEFAULT 0,
    max_attempts       INTEGER NOT NULL DEFAULT 3,
    created_at         REAL    NOT NULL,
    started_at         REAL,
    finished_at        REAL,
    not_before         REAL    NOT NULL DEFAULT 0,
    error              TEXT,
    gcp_execution_name TEXT,
    worker_id          TEXT,
    UNIQUE (device_id, day, source_idx)
);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs (status, priority DESC, created_at, id);
CREATE INDEX IF NOT EXISTS jobs_day ON jobs (day, status);
CREATE TABLE IF NOT EXISTS workers (
    worker_id      TEXT PRIMARY KEY,
    host           TEXT,
    pid            INTEGER,
    started_at     REAL,
    heartbeat_at   REAL,
    current_job_id INTEGER,
    status         TEXT
);
CREATE TABLE IF NOT EXISTS devices_seen (
    device_id  TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Job:
    id: int
    device_id: str
    day: str
    source_idx: int
    priority: int
    status: str
    attempts: int
    max_attempts: int
    created_at: float
    started_at: float | None
    finished_at: float | None
    error: str | None
    gcp_execution_name: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(**{k: row[k] for k in cls.__dataclass_fields__})


def _check_device_id(device_id: str) -> str:
    if not isinstance(device_id, str) or not DEVICE_ID_RE.fullmatch(device_id):
        raise ValueError(f"некорректный device_id: {device_id!r}")
    return device_id


def _check_day(day: str) -> str:
    if not isinstance(day, str) or not DAY_RE.fullmatch(day):
        raise ValueError(f"некорректная дата (YYYY-MM-DD): {day!r}")
    return day


def _check_worker(worker_id: str) -> str:
    if not isinstance(worker_id, str) or not WORKER_ID_RE.fullmatch(worker_id):
        raise ValueError(f"некорректный worker_id: {worker_id!r}")
    return worker_id


def priority_for_age(age_days: int | None, new_days: int = NEW_ACCOUNT_DAYS) -> int:
    """Прогрев: аккаунты моложе new_days дней — PRIORITY_NEW, остальные — PRIORITY_NORMAL."""
    if age_days is not None and age_days < new_days:
        return PRIORITY_NEW
    return PRIORITY_NORMAL


class JobQueue:
    def __init__(self, db_path: str | Path, busy_timeout_sec: float = 30.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_sec = busy_timeout_sec
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    # --- соединения ------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # новое соединение на вызов: sqlite3-соединения нельзя делить между потоками
        c = sqlite3.connect(str(self.db_path), timeout=self.busy_timeout_sec,
                            isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA busy_timeout = %d" % int(self.busy_timeout_sec * 1000))
            c.execute("PRAGMA synchronous = NORMAL")
            yield c
        finally:
            c.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE: сразу берём блокировку записи — два захвата не пересекутся."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    # --- устройства ----------------------------------------------------------

    def first_seen(self, device_id: str, today: str) -> str:
        """Дата первого появления устройства в очереди (запоминается навсегда)."""
        _check_device_id(device_id)
        _check_day(today)
        with self._tx() as c:
            c.execute("INSERT OR IGNORE INTO devices_seen (device_id, first_seen) VALUES (?, ?)",
                      (device_id, today))
            return c.execute("SELECT first_seen FROM devices_seen WHERE device_id = ?",
                             (device_id,)).fetchone()["first_seen"]

    # --- постановка ----------------------------------------------------------

    def enqueue(self, device_id: str, day: str, source_idx: int, priority: int = PRIORITY_NORMAL,
                max_attempts: int = 3) -> bool:
        """True — задача добавлена; False — уже была (у pending обновляется только приоритет)."""
        return self.enqueue_many([(device_id, day, source_idx, priority)], max_attempts) == 1

    def enqueue_many(self, items: list[tuple[str, str, int, int]], max_attempts: int = 3) -> int:
        """items: (device_id, day, source_idx, priority). Возвращает число новых задач."""
        if max_attempts < 1:
            raise ValueError("max_attempts должен быть >= 1")
        rows = []
        for device_id, day, source_idx, priority in items:
            rows.append((_check_device_id(device_id), _check_day(day), int(source_idx), int(priority)))
        now = time.time()
        added = 0
        with self._tx() as c:
            for device_id, day, source_idx, priority in rows:
                cur = c.execute(
                    "INSERT OR IGNORE INTO jobs (device_id, day, source_idx, priority, status, "
                    "max_attempts, created_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                    (device_id, day, source_idx, priority, max_attempts, now))
                if cur.rowcount == 1:
                    added += 1
                else:
                    c.execute("UPDATE jobs SET priority = ? WHERE device_id = ? AND day = ? "
                              "AND source_idx = ? AND status = 'pending'",
                              (priority, device_id, day, source_idx))
        return added

    # --- исполнители ---------------------------------------------------------

    def register_worker(self, worker_id: str) -> None:
        _check_worker(worker_id)
        now = time.time()
        with self._tx() as c:
            c.execute("INSERT INTO workers (worker_id, host, pid, started_at, heartbeat_at, status) "
                      "VALUES (?, ?, ?, ?, ?, 'idle') ON CONFLICT(worker_id) DO UPDATE SET "
                      "host = excluded.host, pid = excluded.pid, started_at = excluded.started_at, "
                      "heartbeat_at = excluded.heartbeat_at, status = 'idle', current_job_id = NULL",
                      (worker_id, socket.gethostname(), os.getpid(), now, now))

    def heartbeat(self, worker_ids: list[str]) -> None:
        if not worker_ids:
            return
        ids = [_check_worker(w) for w in worker_ids]
        with self._tx() as c:
            c.executemany("UPDATE workers SET heartbeat_at = ? WHERE worker_id = ?",
                          [(time.time(), w) for w in ids])

    def unregister_worker(self, worker_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE workers SET status = 'stopped', current_job_id = NULL, "
                      "heartbeat_at = ? WHERE worker_id = ?", (time.time(), _check_worker(worker_id)))

    # --- захват и завершение -------------------------------------------------

    def claim(self, worker_id: str, day: str | None = None) -> Job | None:
        """Атомарно берёт задачу с наибольшим приоритетом (внутри приоритета — FIFO)."""
        _check_worker(worker_id)
        if day is not None:
            _check_day(day)
        now = time.time()
        sql = ("SELECT id FROM jobs WHERE status = 'pending' AND not_before <= ?"
               + (" AND day = ?" if day else "")
               + " ORDER BY priority DESC, created_at, id LIMIT 1")
        args: tuple[Any, ...] = (now, day) if day else (now,)
        with self._tx() as c:
            row = c.execute(sql, args).fetchone()
            if row is None:
                return None
            c.execute("UPDATE jobs SET status = 'running', attempts = attempts + 1, started_at = ?, "
                      "finished_at = NULL, worker_id = ? WHERE id = ?", (now, worker_id, row["id"]))
            c.execute("UPDATE workers SET current_job_id = ?, status = 'busy', heartbeat_at = ? "
                      "WHERE worker_id = ?", (row["id"], now, worker_id))
            return Job.from_row(c.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())

    def complete(self, job_id: int, status: str, error: str | None = None,
                 gcp_execution_name: str | None = None, worker_id: str | None = None,
                 retriable: bool = True) -> bool:
        """status: 'done' | 'failed'. retriable=False — ошибка детерминирована, повтор не поможет
        (attempts выставляется в max_attempts). False в ответе — задача уже не running
        (например, её забрал recover_stale) и результат не записан."""
        if status not in ("done", "failed"):
            raise ValueError(f"complete: status должен быть done|failed, получено {status!r}")
        now = time.time()
        with self._tx() as c:
            cur = c.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, error = ?, "
                "gcp_execution_name = COALESCE(?, gcp_execution_name), "
                "attempts = CASE WHEN ? THEN attempts ELSE MAX(attempts, max_attempts) END "
                "WHERE id = ? AND status = 'running'" + (" AND worker_id = ?" if worker_id else ""),
                (status, now, (error or None) and str(error)[:2000], gcp_execution_name,
                 1 if (retriable or status == "done") else 0, int(job_id),
                 *((worker_id,) if worker_id else ())))
            if worker_id:
                c.execute("UPDATE workers SET current_job_id = NULL, status = 'idle', heartbeat_at = ? "
                          "WHERE worker_id = ?", (now, worker_id))
            return cur.rowcount == 1

    def release(self, job_id: int) -> bool:
        """Вернуть running → pending, не засчитывая попытку (остановка по Ctrl+C)."""
        with self._tx() as c:
            cur = c.execute("UPDATE jobs SET status = 'pending', attempts = MAX(attempts - 1, 0), "
                            "started_at = NULL, worker_id = NULL WHERE id = ? AND status = 'running'",
                            (int(job_id),))
            return cur.rowcount == 1

    def release_worker_jobs(self, worker_ids: list[str]) -> int:
        n = 0
        for w in worker_ids:
            with self._conn() as c:
                ids = [r["id"] for r in c.execute(
                    "SELECT id FROM jobs WHERE status = 'running' AND worker_id = ?", (_check_worker(w),))]
            n += sum(self.release(i) for i in ids)
        return n

    # --- повторы и восстановление ---------------------------------------------

    def retry_failed(self, max_attempts: int | None = None, backoff_sec: float = 0.0,
                     day: str | None = None) -> int:
        """failed с attempts < лимита → pending. Лимит — max_attempts из аргумента или из задачи.
        backoff: задача станет доступна через backoff_sec * 2^(attempts-1)."""
        if day is not None:
            _check_day(day)
        now = time.time()
        with self._tx() as c:
            sql = ("SELECT id, attempts FROM jobs WHERE status = 'failed' "
                   "AND attempts < COALESCE(?, max_attempts)" + (" AND day = ?" if day else ""))
            rows = c.execute(sql, (max_attempts, day) if day else (max_attempts,)).fetchall()
            for r in rows:
                delay = backoff_sec * (2 ** max(0, r["attempts"] - 1)) if backoff_sec > 0 else 0.0
                c.execute("UPDATE jobs SET status = 'pending', not_before = ?, worker_id = NULL "
                          "WHERE id = ?", (now + delay, r["id"]))
            return len(rows)

    def recover_stale(self, stale_after_sec: float = 120.0) -> int:
        """running, чей исполнитель не подаёт heartbeat stale_after_sec секунд (процесс убит),
        → pending (попытка засчитывается; если лимит исчерпан — failed)."""
        limit = time.time() - stale_after_sec
        with self._tx() as c:
            rows = c.execute(
                "SELECT j.id, j.attempts, j.max_attempts FROM jobs j "
                "LEFT JOIN workers w ON w.worker_id = j.worker_id "
                "WHERE j.status = 'running' AND (w.worker_id IS NULL OR w.heartbeat_at < ?)",
                (limit,)).fetchall()
            for r in rows:
                if r["attempts"] >= r["max_attempts"]:
                    c.execute("UPDATE jobs SET status = 'failed', finished_at = ?, "
                              "error = 'исполнитель пропал (stale), попытки исчерпаны' WHERE id = ?",
                              (time.time(), r["id"]))
                else:
                    c.execute("UPDATE jobs SET status = 'pending', worker_id = NULL, started_at = NULL "
                              "WHERE id = ?", (r["id"],))
            return len(rows)

    # --- статистика ----------------------------------------------------------

    def stats(self, day: str | None = None) -> dict[str, int]:
        """pending/running/done/failed + failed_final (попытки исчерпаны) + total."""
        if day is not None:
            _check_day(day)
        where, args = (" WHERE day = ?", (day,)) if day else ("", ())
        with self._conn() as c:
            out = {s: 0 for s in STATUSES}
            for r in c.execute(f"SELECT status, COUNT(*) AS n FROM jobs{where} GROUP BY status", args):
                out[r["status"]] = r["n"]
            out["failed_final"] = c.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status = 'failed' AND attempts >= max_attempts"
                + (" AND day = ?" if day else ""), args).fetchone()["n"]
        out["total"] = sum(out[s] for s in STATUSES)
        return out

    def has_unfinished(self, day: str | None = None) -> bool:
        """Есть ли pending (в т.ч. отложенные backoff), running или failed с попытками."""
        if day is not None:
            _check_day(day)
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM jobs WHERE (status IN ('pending', 'running') OR "
                "(status = 'failed' AND attempts < max_attempts))"
                + (" AND day = ?" if day else "") + " LIMIT 1", (day,) if day else ()).fetchone()
            return row is not None

    def unfinished_days(self) -> list[str]:
        """Дни, где ещё есть что делать (для --resume)."""
        with self._conn() as c:
            return [r["day"] for r in c.execute(
                "SELECT DISTINCT day FROM jobs WHERE status IN ('pending', 'running') OR "
                "(status = 'failed' AND attempts < max_attempts) ORDER BY day")]

    def jobs(self, day: str | None = None, status: str | None = None) -> list[Job]:
        conds, args = [], []
        if day is not None:
            conds.append("day = ?")
            args.append(_check_day(day))
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"неизвестный статус {status!r}")
            conds.append("status = ?")
            args.append(status)
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(conds) if conds else "") + " ORDER BY id"
        with self._conn() as c:
            return [Job.from_row(r) for r in c.execute(sql, args)]


def account_age_days(queue: JobQueue, device: dict, today: str,
                     date_fields: tuple[str, ...] = ("warmup_started", "created_at")) -> int:
    """Возраст аккаунта в днях. Источник: device[warmup_started|created_at] → самая ранняя
    accounts.*.created_at → дата первого появления устройства в очереди.
    [ПРОВЕРИТЬ: как называется поле даты создания в вашем devices.json]"""
    candidates: list[str] = []
    for f in date_fields:
        if device.get(f):
            candidates.append(str(device[f]))
    for acc in (device.get("accounts") or {}).values():
        if isinstance(acc, dict) and acc.get("created_at"):
            candidates.append(str(acc["created_at"]))
    parsed: list[date_cls] = []
    for s in candidates:
        try:
            parsed.append(date_cls.fromisoformat(s[:10]))
        except ValueError:
            log.warning("%s: непонятная дата %r — пропускаю", device.get("device_id"), s)
    start = min(parsed) if parsed else date_cls.fromisoformat(
        queue.first_seen(device["device_id"], today))
    return max(0, (date_cls.fromisoformat(today) - start).days)
