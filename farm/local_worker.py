"""local_worker.py — генерация на своём GPU (RTX 4090) вместо Cloud Run Job.

Запуск (на машине с GPU):
    LOCAL_WORKER_TOKEN=... python local_worker.py [--config config.json]
Модели грузятся ОДИН раз при старте (model_server.warm), дальше каждая задача вызывает
model_server.run_pipeline прямо в этом процессе — без контейнера, без GCS.

API (FastAPI, адрес и порт из local.server_url, по умолчанию 127.0.0.1:8081; заголовок
Authorization: Bearer <LOCAL_WORKER_TOKEN>):
  POST /submit          — поставить задачу (повторная отправка того же device/day/source_idx не
                          создаёт дубль; упавшая задача перезапускается с новым payload)
  GET  /status/{id}     — статус, попытки, ошибка, использованный лимит кадров
  GET  /result/{id}     — готовый mp4
  GET  /health          — GPU, видеопамять, очередь, режим остановки

Очередь — та же job_queue (SQLite, local.queue_db), payload задач — в соседней таблице.
Видеопамять: перед задачей torch.cuda.mem_get_info(); свободно меньше local.vram_safety_mb —
задача возвращается в очередь без списания попытки, воркер ждёт local.vram_wait_sec.
Нехватка памяти (CUDA OOM) во время задачи → failed и повтор с меньшим лимитом кадров по
лесенке local.oom_max_frames (например 121 → 81 → 49). На последней ступени OOM — окончательно.
Меньше кадров = короче сцена (при 24 к/с: 81 кадр ≈ 3,4 с, 49 ≈ 2 с).

SIGTERM / Ctrl+C: новые задачи не берутся и /submit отвечает 503, текущая задача
дорабатывает, затем процесс завершается (uvicorn вызывает shutdown приложения). После мягкой
остановки uvicorn повторно посылает себе тот же сигнал, поэтому код выхода 143 (SIGTERM) / 130
(Ctrl+C) — это нормальная остановка; systemd считает выход по SIGTERM успешным.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from job_queue import Job, JobQueue

log = logging.getLogger("local_worker")

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MB = 1024 * 1024

PAYLOAD_SCHEMA = """
CREATE TABLE IF NOT EXISTS payloads (
    job_id      INTEGER PRIMARY KEY,
    payload     TEXT    NOT NULL,
    max_frames  INTEGER,
    oom_count   INTEGER NOT NULL DEFAULT 0,
    output_path TEXT,
    run_sec     REAL
);
"""


def is_oom(exc: BaseException) -> bool:
    """torch.cuda.OutOfMemoryError или RuntimeError с «CUDA out of memory»."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


@dataclass
class WorkerSettings:
    vram_safety_mb: int = 2000
    max_concurrent_jobs: int = 1
    oom_max_frames: tuple[int, ...] = (81, 49)
    vram_wait_sec: float = 15.0
    default_max_frames: int = 121
    heartbeat_sec: float = 30.0
    stale_after_sec: float = 300.0


class LocalWorker:
    def __init__(self, queue_db: Path, output_dir: Path, masters_dir: Path,
                 settings: WorkerSettings, *, pipeline: Callable[..., Path],
                 warm: Callable[[], None], cuda: Any, persona_seed: Callable[[dict, str], int],
                 masters_io: Callable[..., tuple]):
        self.q = JobQueue(queue_db)
        self.db_path = Path(queue_db)
        self.output_dir = Path(output_dir)
        self.masters_dir = Path(masters_dir)
        self.s = settings
        self.pipeline = pipeline
        self._warm = warm
        self.cuda = cuda
        self.persona_seed = persona_seed
        self.masters_io = masters_io
        self.max_attempts = len(settings.oom_max_frames) + 2
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []
        self._busy = 0
        self._busy_lock = threading.Lock()
        self.draining = False
        self.warmed = False
        host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())[:40] or "host"
        self.worker_ids = [f"local-{host}-{os.getpid()}-w{i}" for i in range(settings.max_concurrent_jobs)]
        with self._db() as c:
            c.executescript(PAYLOAD_SCHEMA)

    # --- фабрика с настоящими model_server и torch ---------------------------

    @classmethod
    def from_config(cls, cfg: Any) -> "LocalWorker":
        # model_server читает MODELS_DIR и WAN_* при импорте — окружение до импорта
        os.environ.setdefault("MODELS_DIR", str(cfg.local.models_dir))
        os.environ["STORAGE_BACKEND"] = "local"
        import torch
        import model_server
        if not torch.cuda.is_available():
            raise SystemExit("CUDA недоступна: local_worker рассчитан на GPU")
        lc = cfg.local
        return cls(cfg.local_queue_db_path, cfg.local_output_dir, model_server.MODELS_DIR / "masters",
                   WorkerSettings(vram_safety_mb=lc.vram_safety_mb,
                                  max_concurrent_jobs=lc.max_concurrent_jobs,
                                  oom_max_frames=tuple(lc.oom_max_frames),
                                  vram_wait_sec=lc.vram_wait_sec,
                                  default_max_frames=model_server.WAN_MAX_FRAMES),
                   pipeline=model_server.run_pipeline, warm=model_server.warm, cuda=torch.cuda,
                   persona_seed=model_server._persona_seed, masters_io=model_server.masters_dir_io)

    # --- БД payload ------------------------------------------------------------

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Соединение на операцию (sqlite3 нельзя делить между потоками), всегда закрывается."""
        c = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    def _payload_row(self, job_id: int) -> sqlite3.Row | None:
        with self._db() as c:
            return c.execute("SELECT * FROM payloads WHERE job_id = ?", (job_id,)).fetchone()

    def _job(self, job_id: int) -> Job | None:
        with self._db() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def output_path(self, device_id: str, day: str, source_idx: int) -> Path:
        if not ID_RE.fullmatch(device_id) or not DAY_RE.fullmatch(day):
            raise ValueError("некорректный device_id или day")
        return self.output_dir / device_id / day / f"source_{int(source_idx)}.mp4"

    # --- API-операции ------------------------------------------------------------

    def submit(self, req: dict) -> dict:
        if self.draining:
            raise RuntimeError("воркер останавливается — новые задачи не принимаются")
        device_id, day, idx = req["device_id"], req["day"], int(req["source_idx"])
        self.q.enqueue(device_id, day, idx, priority=5, max_attempts=self.max_attempts)
        payload = json.dumps({k: v for k, v in req.items() if k not in ("day", "source_idx")},
                             ensure_ascii=False)
        with self._db() as c:
            c.execute("BEGIN IMMEDIATE")
            job = c.execute("SELECT id, status FROM jobs WHERE device_id = ? AND day = ? "
                            "AND source_idx = ?", (device_id, day, idx)).fetchone()
            job_id, status = job["id"], job["status"]
            if status == "failed":  # forge прислал задачу снова — начинаем заново
                c.execute("UPDATE jobs SET status = 'pending', attempts = 0, error = NULL, "
                          "not_before = 0, max_attempts = ? WHERE id = ?", (self.max_attempts, job_id))
                c.execute("DELETE FROM payloads WHERE job_id = ?", (job_id,))
                status = "pending"
            if status == "pending":
                c.execute("INSERT INTO payloads (job_id, payload, max_frames) VALUES (?, ?, ?) "
                          "ON CONFLICT(job_id) DO UPDATE SET payload = excluded.payload",
                          (job_id, payload, req.get("max_frames")))
            c.execute("COMMIT")
        self._wake.set()
        log.info("submit %s/%s/source_%d → задача %d (%s)", device_id, day, idx, job_id, status)
        return {"job_id": job_id, "status": status}

    def status(self, job_id: int) -> dict | None:
        job = self._job(job_id)
        if job is None:
            return None
        p = self._payload_row(job_id)
        out = Path(p["output_path"]) if p and p["output_path"] else None
        return {"job_id": job.id, "device_id": job.device_id, "day": job.day,
                "source_idx": job.source_idx, "status": job.status, "attempts": job.attempts,
                "max_attempts": job.max_attempts,
                "final": job.status == "done" or (job.status == "failed"
                                                  and job.attempts >= job.max_attempts),
                "error": job.error, "started_at": job.started_at, "finished_at": job.finished_at,
                "max_frames": p["max_frames"] if p else None, "oom_count": p["oom_count"] if p else 0,
                "run_sec": p["run_sec"] if p else None,
                "output_ready": bool(out and job.status == "done" and out.exists())}

    def result_path(self, job_id: int) -> Path | None:
        st = self.status(job_id)
        if not st or not st["output_ready"]:
            return None
        return Path(self._payload_row(job_id)["output_path"])

    def vram(self) -> dict:
        free, total = self.cuda.mem_get_info()
        return {"free_mb": int(free // MB), "total_mb": int(total // MB)}

    def health(self) -> dict:
        try:
            gpu = self.vram()
        except Exception as e:
            gpu = {"error": str(e)[:200]}
        with self._busy_lock:
            busy = self._busy
        return {"ok": self.warmed and not self.draining, "mode": "local", "models_loaded": self.warmed,
                "draining": self.draining, "busy_jobs": busy, "gpu": gpu,
                "vram_safety_mb": self.s.vram_safety_mb, "queue": self.q.stats()}

    # --- жизненный цикл ----------------------------------------------------------

    def start(self) -> None:
        t0 = time.monotonic()
        self._warm()  # модели — один раз на процесс
        self.warmed = True
        log.info("модели загружены за %.0f с", time.monotonic() - t0)
        recovered = self.q.recover_stale(0)  # процесс один — всё running осталось от прошлого запуска
        if recovered:
            log.warning("возвращено в очередь задач от прошлого запуска: %d", recovered)
        for w in self.worker_ids:
            t = threading.Thread(target=self._loop, args=(w,), name=w, daemon=True)
            t.start()
            self._threads.append(t)
        threading.Thread(target=self._heartbeat, name="local-heartbeat", daemon=True).start()

    def shutdown(self, timeout: float | None = None) -> None:
        """Мягкая остановка: новые задачи не брать, текущие дождаться."""
        if not self.draining:
            log.warning("остановка: жду завершения текущих задач")
        self.draining = True
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout)
        log.info("воркер остановлен")

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.s.heartbeat_sec):
            try:
                self.q.heartbeat(self.worker_ids)
            except Exception as e:
                log.warning("heartbeat: %s", e)

    def _loop(self, worker_id: str) -> None:
        self.q.register_worker(worker_id)
        try:
            while not self._stop.is_set():
                job = self.q.claim(worker_id)
                if job is None:
                    if self.q.retry_failed(backoff_sec=0):
                        continue
                    self._wake.wait(1.0)
                    self._wake.clear()
                    continue
                self._run(job, worker_id)
        finally:
            self.q.unregister_worker(worker_id)

    def _vram_ok(self) -> bool:
        info = self.vram()
        if info["free_mb"] >= self.s.vram_safety_mb:
            return True
        self.cuda.empty_cache()
        info = self.vram()
        if info["free_mb"] >= self.s.vram_safety_mb:
            return True
        log.warning("мало видеопамяти: свободно %d МБ < %d МБ — жду %.0f с", info["free_mb"],
                    self.s.vram_safety_mb, self.s.vram_wait_sec)
        return False

    def _next_frames(self, current: int) -> int | None:
        return next((f for f in self.s.oom_max_frames if f < current), None)

    def _run(self, job: Job, worker_id: str) -> None:
        row = self._payload_row(job.id)
        if row is None:
            self.q.complete(job.id, "failed", error="нет payload задачи", worker_id=worker_id,
                            retriable=False)
            return
        if not self._vram_ok():
            self.q.release(job.id)  # попытка не списывается
            self._stop.wait(self.s.vram_wait_sec)
            return
        payload = json.loads(row["payload"])
        frames = int(row["max_frames"] or self.s.default_max_frames)
        payload["max_frames"] = frames
        out = self.output_path(job.device_id, job.day, job.source_idx)
        with self._busy_lock:
            self._busy += 1
        t0 = time.monotonic()
        log.info("[%s] задача %d %s/%s/source_%d: попытка %d/%d, max_frames=%d, VRAM свободно %d МБ",
                 worker_id.rsplit("-", 1)[-1], job.id, job.device_id, job.day, job.source_idx,
                 job.attempts, job.max_attempts, frames, self.vram()["free_mb"])
        try:
            lock = self.persona_seed(payload, job.device_id)
            fetch, store = self.masters_io(self.masters_dir, job.device_id, lock)
            with tempfile.TemporaryDirectory(prefix="lw_") as tmp_s:
                final = self.pipeline(payload, job.device_id, job.source_idx,
                                      payload.get("voice_id") or "voice_01",
                                      bool(payload.get("split_screen")), Path(tmp_s), fetch, store)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp_out = out.with_name(f".{out.name}.tmp")
                shutil.copyfile(final, tmp_out)
                os.replace(tmp_out, out)
            run_sec = time.monotonic() - t0
            with self._db() as c:
                c.execute("UPDATE payloads SET output_path = ?, run_sec = ?, max_frames = ? "
                          "WHERE job_id = ?", (str(out), run_sec, frames, job.id))
            self.q.complete(job.id, "done", worker_id=worker_id,
                            gcp_execution_name=f"local:{job.id}")
            log.info("задача %d готова за %.0f с → %s", job.id, run_sec, out)
        except Exception as e:
            if is_oom(e):
                nxt = self._next_frames(frames)
                with self._db() as c:
                    c.execute("UPDATE payloads SET oom_count = oom_count + 1, max_frames = ? "
                              "WHERE job_id = ?", (nxt or frames, job.id))
                if nxt is None:
                    log.error("задача %d: нехватка VRAM даже при max_frames=%d — окончательно",
                              job.id, frames)
                else:
                    log.warning("задача %d: нехватка VRAM при max_frames=%d — повтор с %d",
                                job.id, frames, nxt)
                self.q.complete(job.id, "failed", error=f"CUDA OOM при max_frames={frames}",
                                worker_id=worker_id, retriable=nxt is not None)
            else:
                log.exception("задача %d упала", job.id)
                self.q.complete(job.id, "failed", error=f"{type(e).__name__}: {e}"[:500],
                                worker_id=worker_id)
        finally:
            try:
                self.cuda.empty_cache()
            except Exception:
                pass
            with self._busy_lock:
                self._busy -= 1


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

# Модель запроса — на уровне модуля: с `from __future__ import annotations` FastAPI не видит
# классы, объявленные внутри функции, и принимает тело запроса за query-параметр.
class SubmitRequest(BaseModel):
    device_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    day: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    source_idx: int = Field(ge=1, le=100)
    scenes: list[dict] = Field(min_length=1, max_length=9)
    voice_id: str = Field(default="voice_01", pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    split_screen: bool = False
    narration: str = Field(default="", max_length=500)
    seed: int | None = Field(default=None, ge=0, le=2 ** 63 - 1)
    negative_prompt: str = Field(default="", max_length=1000)
    persona_lock: int | None = Field(default=None, ge=0, le=2 ** 32 - 1)
    max_frames: int | None = Field(default=None, ge=5, le=1000)



def create_app(worker: LocalWorker, token: str | None):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(worker.start)
        yield
        await asyncio.to_thread(worker.shutdown)  # дождаться текущей задачи

    app = FastAPI(title="local_worker", lifespan=lifespan)

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post("/submit", status_code=202, dependencies=[Depends(auth)])
    def submit(req: SubmitRequest):
        try:
            return worker.submit(req.model_dump(exclude_none=True))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/status/{job_id}", dependencies=[Depends(auth)])
    def status(job_id: int):
        st = worker.status(job_id)
        if st is None:
            raise HTTPException(status_code=404, detail="нет такой задачи")
        return st

    @app.get("/result/{job_id}", dependencies=[Depends(auth)])
    def result(job_id: int):
        p = worker.result_path(job_id)
        if p is None:
            raise HTTPException(status_code=409, detail="результат ещё не готов")
        return FileResponse(str(p), media_type="video/mp4", filename=p.name)

    @app.get("/health", dependencies=[Depends(auth)])
    def health():
        return worker.health()

    return app


def main() -> None:
    from config_loader import cli_config

    def extra(ap):
        ap.add_argument("--host", default=None, help="по умолчанию — хост из local.server_url")

    cfg, a = cli_config("local_worker", extra=extra)
    token = os.getenv("LOCAL_WORKER_TOKEN", "").strip() or None
    if not token and os.getenv("ALLOW_NO_AUTH") != "1":
        raise SystemExit("задай LOCAL_WORKER_TOKEN (или ALLOW_NO_AUTH=1 только для локальной отладки)")
    url = urlsplit(cfg.local.server_url)
    host, port = a.host or url.hostname or "127.0.0.1", url.port or 8081
    worker = LocalWorker.from_config(cfg)
    import uvicorn
    # uvicorn сам ловит SIGTERM/SIGINT: перестаёт принимать запросы и вызывает lifespan-shutdown
    server = uvicorn.Server(uvicorn.Config(create_app(worker, token), host=host, port=port,
                                           log_config=None))
    server.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
