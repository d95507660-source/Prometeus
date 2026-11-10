"""tests/test_queue.py — очередь генерации на моках GCP (без сети, без GPU).

Запуск:  python -m pytest -q tests/test_queue.py
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_loader  # noqa: E402
import forge_stage  # noqa: E402
from job_queue import PRIORITY_NEW, PRIORITY_NORMAL, JobQueue, account_age_days, priority_for_age  # noqa: E402

N_ACCOUNTS = 30
PER_DAY = 2
N_JOBS = N_ACCOUNTS * PER_DAY
DAY = "2026-11-10"


# ---------------------------------------------------------------------------
# Моки
# ---------------------------------------------------------------------------

class FakeCfg:
    def __init__(self, root: Path, **forge_overrides):
        self.root = root
        self.forge = config_loader.ForgeConfig(videos_per_device_per_day=PER_DAY, max_parallel_jobs=8,
                                               retry_backoff_sec=0, **forge_overrides)

    @property
    def queue_db_path(self) -> Path:
        return self.root / self.forge.queue_db

    def today(self) -> str:
        return DAY


@dataclass
class FakeRegistry:
    devices: list[dict]

    def get_active_devices(self, niches=None, device_ids=None):
        return [d for d in self.devices if not device_ids or d["device_id"] in device_ids]


@dataclass
class FakeForge:
    """Вместо Cloud Run: fail_times[(device, idx)] = сколько первых попыток падают (-1 — всегда)."""
    fail_times: dict[tuple[str, int], int] = field(default_factory=dict)
    on_call: object = None
    delay: float = 0.0
    calls: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def generate_one(self, device: dict, source_idx: int, day: str):
        key = (device["device_id"], source_idx)
        with self.lock:
            self.calls[key] += 1
            n = self.calls[key]
            total = sum(self.calls.values())
        if self.on_call:
            self.on_call(total)
        if self.delay:
            time.sleep(self.delay)
        limit = self.fail_times.get(key, 0)
        if limit == -1 or n <= limit:
            return SimpleNamespace(status="gcp_failed", error=f"mock FAILED {key} #{n}",
                                   gcp_execution_name=f"executions/mock-{key[0]}-{key[1]}-{n}")
        return SimpleNamespace(status="ok", error=None,
                               gcp_execution_name=f"executions/mock-{key[0]}-{key[1]}-{n}")


def make_devices(n: int = N_ACCOUNTS) -> list[dict]:
    return [{"device_id": f"acc_{i:02d}", "niche": "fitness_female",
             "accounts": {"tiktok": {"username": f"a{i}.lift"}}} for i in range(n)]


def every_fifth(devices: list[dict]) -> list[tuple[str, int]]:
    """Ровно 20 % задач (12 из 60)."""
    keys = [(d["device_id"], i) for d in devices for i in range(1, PER_DAY + 1)]
    return keys[::5]


FAST = forge_stage.RunSettings(max_workers=8, stop_on_failure_ratio=0.2, retry_backoff_sec=0,
                               heartbeat_sec=0.2, stale_after_sec=5, poll_sec=0.02)


@pytest.fixture
def env(tmp_path):
    cfg = FakeCfg(tmp_path)
    devices = make_devices()
    return cfg, FakeRegistry(devices), devices


def run(cfg, registry, fake, **kw) -> int:
    return forge_stage.stage_forge(cfg, registry, forge=fake, settings=kw.pop("settings", FAST), **kw)


# ---------------------------------------------------------------------------
# Критерии приёмки
# ---------------------------------------------------------------------------

def test_flaky_20_percent_retried_until_done(env):
    cfg, registry, devices = env
    flaky = every_fifth(devices)
    fake = FakeForge(fail_times={k: 2 for k in flaky})  # 2 падения, 3-я попытка успешна
    assert run(cfg, registry, fake) == 0
    q = JobQueue(cfg.queue_db_path)
    st = q.stats(DAY)
    assert st["total"] == N_JOBS and st["done"] == N_JOBS and st["failed"] == 0
    by_key = {(j.device_id, j.source_idx): j for j in q.jobs(DAY)}
    for k in flaky:
        assert by_key[k].attempts == 3 and fake.calls[k] == 3
        assert by_key[k].gcp_execution_name.endswith("-3")
    assert sum(fake.calls.values()) == N_JOBS + 2 * len(flaky)


def test_permanent_failures_retried_to_max_attempts(env):
    cfg, registry, devices = env
    broken = every_fifth(devices)
    fake = FakeForge(fail_times={k: -1 for k in broken})
    code = run(cfg, registry, fake)
    q = JobQueue(cfg.queue_db_path)
    st = q.stats(DAY)
    assert code == 0  # 12/60 = 20 % — ровно порог, не больше
    assert st["done"] == N_JOBS - len(broken) and st["failed_final"] == len(broken)
    for j in q.jobs(DAY, status="failed"):
        assert j.attempts == j.max_attempts == 3 and "mock FAILED" in j.error
    assert all(fake.calls[k] == 3 for k in broken)


def test_stops_with_code_3_when_ratio_exceeded(env):
    cfg, registry, devices = env
    keys = [(d["device_id"], i) for d in devices for i in range(1, PER_DAY + 1)]
    fake = FakeForge(fail_times={k: -1 for k in keys[::3]})  # 20 из 60 = 33 %
    assert run(cfg, registry, fake) == 3
    st = JobQueue(cfg.queue_db_path).stats(DAY)
    assert st["failed_final"] / st["total"] > 0.2
    assert st["running"] == 0


def test_rerun_does_not_duplicate(env):
    cfg, registry, _ = env
    assert run(cfg, registry, FakeForge()) == 0
    again = FakeForge()
    assert run(cfg, registry, again) == 0
    assert sum(again.calls.values()) == 0
    assert len(JobQueue(cfg.queue_db_path).jobs()) == N_JOBS


def test_interrupt_then_resume_finishes_without_duplicates(env):
    cfg, registry, _ = env
    stop = threading.Event()
    first = FakeForge(on_call=lambda n: stop.set() if n >= 10 else None, delay=0.01)
    assert run(cfg, registry, first, stop_event=stop) == 130
    q = JobQueue(cfg.queue_db_path)
    st = q.stats(DAY)
    assert 10 <= st["done"] < N_JOBS and st["running"] == 0
    assert st["pending"] == N_JOBS - st["done"]

    second = FakeForge()
    assert run(cfg, registry, second, resume=True) == 0
    assert q.stats(DAY)["done"] == N_JOBS
    merged = first.calls + second.calls
    assert len(merged) == N_JOBS and all(v == 1 for v in merged.values())


def test_hard_kill_recovered_on_resume(env):
    """Процесс убит (kill -9): задача осталась running, heartbeat старый → resume её доделает."""
    cfg, registry, _ = env
    q = JobQueue(cfg.queue_db_path)
    q.enqueue("acc_00", DAY, 1)
    q.register_worker("dead-worker")
    job = q.claim("dead-worker")
    with q._tx() as c:  # «процесс умер давно»
        c.execute("UPDATE workers SET heartbeat_at = 0 WHERE worker_id = 'dead-worker'")
    fake = FakeForge()
    assert run(cfg, registry, fake, resume=True) == 0
    j = q.jobs(DAY)[0]
    assert j.id == job.id and j.status == "done" and j.attempts == 2


def test_release_on_second_ctrl_c_refunds_attempt(tmp_path):
    q = JobQueue(tmp_path / "q.db")
    q.enqueue("acc_00", DAY, 1)
    q.register_worker("w1")
    q.claim("w1")
    assert q.release_worker_jobs(["w1"]) == 1
    j = q.jobs()[0]
    assert j.status == "pending" and j.attempts == 0


def test_progress_logged_every_10(env, caplog):
    cfg, registry, _ = env
    with caplog.at_level(logging.INFO, logger="forge_stage"):
        assert run(cfg, registry, FakeForge()) == 0
    progress = [r.getMessage() for r in caplog.records if r.getMessage().endswith("failed")
                and "/60 done" in r.getMessage()]
    assert len(progress) == N_JOBS // 10
    assert any(m.startswith("60/60 done, 0 failed") for m in progress)


# ---------------------------------------------------------------------------
# Очередь
# ---------------------------------------------------------------------------

def test_priority_new_accounts_first_then_fifo(tmp_path):
    q = JobQueue(tmp_path / "q.db")
    q.enqueue_many([("old_1", DAY, 1, PRIORITY_NORMAL), ("new_1", DAY, 1, PRIORITY_NEW),
                    ("old_2", DAY, 1, PRIORITY_NORMAL), ("new_2", DAY, 1, PRIORITY_NEW)])
    q.register_worker("w")
    order = []
    while (j := q.claim("w")) is not None:
        order.append(j.device_id)
    assert order == ["new_1", "new_2", "old_1", "old_2"]


def test_account_age_and_priority(tmp_path):
    q = JobQueue(tmp_path / "q.db")
    old = {"device_id": "old", "created_at": "2026-01-01"}
    acc_old = {"device_id": "acc", "accounts": {"tiktok": {"created_at": "2026-10-01"}}}
    fresh = {"device_id": "fresh"}  # даты нет → дата первого появления в очереди
    assert priority_for_age(account_age_days(q, old, DAY)) == PRIORITY_NORMAL
    assert priority_for_age(account_age_days(q, acc_old, DAY)) == PRIORITY_NORMAL
    assert account_age_days(q, fresh, DAY) == 0
    assert priority_for_age(account_age_days(q, fresh, "2026-11-20")) == PRIORITY_NEW
    assert priority_for_age(account_age_days(q, fresh, "2026-11-30")) == PRIORITY_NORMAL


def test_concurrent_claim_is_atomic(tmp_path):
    q = JobQueue(tmp_path / "q.db")
    q.enqueue_many([(f"d{i:03d}", DAY, 1, 5) for i in range(200)])
    got: list[int] = []
    lock = threading.Lock()

    def worker(w: str) -> None:
        q.register_worker(w)
        while (j := q.claim(w)) is not None:
            with lock:
                got.append(j.id)
            q.complete(j.id, "done", worker_id=w)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 200 and len(set(got)) == 200


def test_retry_failed_respects_limit_and_backoff(tmp_path):
    q = JobQueue(tmp_path / "q.db")
    q.enqueue("d1", DAY, 1, max_attempts=2)
    q.register_worker("w")
    j = q.claim("w")
    q.complete(j.id, "failed", error="boom", worker_id="w")
    assert q.retry_failed(backoff_sec=3600) == 1
    assert q.claim("w") is None  # отложено backoff
    with q._tx() as c:
        c.execute("UPDATE jobs SET not_before = 0")
    j = q.claim("w")
    assert j.attempts == 2
    q.complete(j.id, "failed", error="boom", worker_id="w")
    assert q.retry_failed() == 0 and q.stats()["failed_final"] == 1


def test_non_retriable_failure_is_final(tmp_path):
    q = JobQueue(tmp_path / "q.db")
    q.enqueue("d1", DAY, 1, max_attempts=3)
    q.register_worker("w")
    j = q.claim("w")
    q.complete(j.id, "failed", error="realism_rejected", worker_id="w", retriable=False)
    assert q.retry_failed() == 0 and q.stats()["failed_final"] == 1


def test_bad_ids_rejected(tmp_path):
    q = JobQueue(tmp_path / "q.db")
    with pytest.raises(ValueError):
        q.enqueue("../etc", DAY, 1)
    with pytest.raises(ValueError):
        q.enqueue("ok", "2026-11-10; DROP TABLE jobs", 1)
    with pytest.raises(ValueError):
        q.claim("bad worker id")


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

def test_config_json_queue_keys_are_loaded(tmp_path, caplog):
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    niches = {n: {"marker": m} for n, m in raw["devices_registry"]["niche_markers"].items()}
    (tmp_path / "niches.json").write_text(json.dumps({"niches": niches}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="config_loader"):
        cfg = config_loader.load_config(tmp_path / "config.json")
    assert not any("неизвестные ключи forge" in r.getMessage() for r in caplog.records)
    f = cfg.forge
    assert (f.max_parallel_jobs, f.queue_db, f.retry_max_attempts, f.stop_on_failure_ratio) == \
        (8, "data/queue.db", 3, 0.2)
    assert cfg.queue_db_path == tmp_path / "data" / "queue.db"


def test_config_validation_rejects_bad_ratio(tmp_path):
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    raw["forge"]["stop_on_failure_ratio"] = 1.5
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    niches = {n: {"marker": m} for n, m in raw["devices_registry"]["niche_markers"].items()}
    (tmp_path / "niches.json").write_text(json.dumps({"niches": niches}), encoding="utf-8")
    with pytest.raises(ValueError, match="stop_on_failure_ratio"):
        config_loader.load_config(tmp_path / "config.json")
