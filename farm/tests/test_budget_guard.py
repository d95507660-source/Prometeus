"""tests/test_budget_guard.py — месячный лимит, резервы, интеграция с очередью forge (без GCP).

Запуск:  python -m pytest -q tests/test_budget_guard.py
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import budget_guard as bg  # noqa: E402
import config_loader  # noqa: E402
import forge_stage  # noqa: E402
from job_queue import JobQueue  # noqa: E402

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
RATE = 0.672


class Clock:
    def __init__(self, t: datetime = T0):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def budget(**kw) -> config_loader.BudgetConfig:
    return config_loader.BudgetConfig(enabled=True, **{"monthly_limit_usd": 10.0, **kw})


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def guard(tmp_path, clock) -> bg.BudgetGuard:
    return bg.BudgetGuard(budget(), tmp_path / "budget.json", now=clock)


def state(g: bg.BudgetGuard) -> dict:
    return json.loads(g.path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Основное
# ---------------------------------------------------------------------------

def test_can_spend_against_limit(guard):
    assert guard.can_spend(10.0) and not guard.can_spend(10.01)
    guard.record_spend(9.5, "j1")
    assert guard.can_spend(0.5) and not guard.can_spend(0.51)


def test_record_spend_atomic_file_and_idempotent(guard, caplog):
    assert guard.record_spend(1.25, "j1") == 1.25
    with caplog.at_level(logging.WARNING, logger="budget_guard"):
        assert guard.record_spend(1.25, "j1") == 1.25  # тот же job_id не удваивает
    st = state(guard)
    assert st["spent_usd"] == 1.25 and st["recorded_jobs"] == ["j1"] and st["month"] == "2026-11"
    assert not list(guard.path.parent.glob(".budget.json.*.tmp"))
    assert any("уже записана" in r.getMessage() for r in caplog.records)


def test_estimate_formula_and_calibration(guard):
    # до статистики: сцены × default_scene_gpu_sec(120) × 0.672 / 3600
    assert guard.estimate_cost(3) == pytest.approx(3 * 120 * RATE / 3600)
    guard.record_spend(guard.cost_for_seconds(600), "j1", gpu_sec=600, scenes=4)  # 150 с на сцену
    assert guard.avg_scene_gpu_sec() == 150
    assert guard.estimate_cost(2) == pytest.approx(2 * 150 * RATE / 3600)
    assert guard.cost_for_seconds(3600) == pytest.approx(RATE)


def test_reservations_are_atomic_across_threads(guard):
    ok: list[bool] = []
    lock = threading.Lock()

    def worker(i):
        r = guard.try_reserve(f"j{i}", 1.0)
        with lock:
            ok.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ok.count(True) == 10  # лимит $10, каждый резерв $1
    assert guard.status()["reserved_usd"] == 10.0 and not guard.can_spend(0.01)


def test_record_replaces_reservation_and_release_frees_it(guard):
    assert guard.try_reserve("a", 6.0) and not guard.try_reserve("b", 5.0)
    guard.record_spend(2.0, "a")  # факт дешевле оценки — освободилось место
    assert guard.try_reserve("b", 5.0)
    guard.release("b")
    assert guard.status() == {"month": "2026-11", "spent_usd": 2.0, "reserved_usd": 0.0,
                              "limit_usd": 10.0, "remaining_usd": 8.0, "percent": 20.0,
                              "alerts_sent": []}


def test_stale_reservation_expires(guard, monkeypatch):
    assert guard.try_reserve("dead", 9.0)
    real = bg.time.time
    monkeypatch.setattr(bg.time, "time", lambda: real() + bg.RESERVATION_TTL_SEC + 1)
    assert guard.can_spend(5.0)


def test_new_billing_month_resets_in_pacific_time(tmp_path, clock):
    g = bg.BudgetGuard(budget(), tmp_path / "budget.json", now=clock)
    g.record_spend(9.0, "nov")
    clock.t = datetime(2026, 12, 1, 5, 0, tzinfo=timezone.utc)  # в Лос-Анджелесе ещё 30 ноября
    assert g.status()["month"] == "2026-11" and not g.can_spend(2.0)
    clock.t = datetime(2026, 12, 1, 9, 0, tzinfo=timezone.utc)  # 1 декабря 01:00 PST
    assert g.can_spend(10.0)
    st = state(g)
    assert st["month"] == "2026-12" and st["spent_usd"] == 0 and st["history"] == {"2026-11": 9.0}


def test_seed_current_spent_only_for_fresh_state(tmp_path, clock):
    g = bg.BudgetGuard(budget(current_spent_usd=4.0), tmp_path / "budget.json", now=clock)
    assert g.status()["spent_usd"] == 4.0
    g.record_spend(1.0, "j")
    assert bg.BudgetGuard(budget(current_spent_usd=4.0), g.path, now=clock).status()["spent_usd"] == 5.0


def test_alerts_once_per_threshold(guard, caplog):
    with caplog.at_level(logging.WARNING, logger="budget_guard"):
        guard.record_spend(4.0, "a")   # 40 %
        guard.record_spend(1.0, "b")   # 50 %
        guard.record_spend(1.0, "c")   # 60 %
        guard.record_spend(3.5, "d")   # 95 %
        guard.record_spend(0.5, "e")   # 100 %
    alerts = [r.getMessage() for r in caplog.records if "БЮДЖЕТ" in r.getMessage()]
    assert [a.split("потрачено ")[1].split("%")[0] for a in alerts] == ["50", "95", "100"]
    assert state(guard)["alerts_sent"] == [50, 90, 100]


def test_reset_keeps_reservations(guard):
    guard.record_spend(10.0, "a")
    guard.try_reserve("running", 0.0)
    assert not guard.can_spend(0.01)
    guard.reset("тест")
    st = state(guard)
    assert st["spent_usd"] == 0 and "running" in st["reserved"] and st["resets"][0]["spent_before"] == 10.0


def test_corrupt_state_is_treated_as_exhausted(guard, caplog):
    guard.path.parent.mkdir(parents=True, exist_ok=True)
    guard.path.write_text("{oops", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="budget_guard"):
        assert not guard.can_spend(0.01)
    assert list(guard.path.parent.glob("budget.json.corrupt-*"))
    guard.reset("после разбора")
    assert guard.can_spend(1.0)


@pytest.mark.parametrize("call", [
    lambda g: g.can_spend(-1), lambda g: g.can_spend(float("nan")), lambda g: g.record_spend(-0.1, "x"),
    lambda g: g.record_spend(1, "bad id/../x"), lambda g: g.try_reserve("", 1), lambda g: g.estimate_cost(0),
    lambda g: g.record_spend(True, "x"),
])
def test_bad_input_rejected(guard, call):
    with pytest.raises(ValueError):
        call(guard)


def test_gcp_budget_command():
    cmd = bg.gcp_budget_command(budget(monthly_limit_usd=300.0), "01ABCD-23EF45-6789AB")
    assert cmd == ["gcloud", "billing", "budgets", "create", "--billing-account=01ABCD-23EF45-6789AB",
                   "--display-name=prometheus monthly", "--budget-amount=300USD",
                   "--threshold-rule=percent=0.5", "--threshold-rule=percent=0.9",
                   "--threshold-rule=percent=1"]
    with pytest.raises(ValueError):
        bg.gcp_budget_command(budget(), "acc; rm -rf /")


def test_guard_for_disabled_or_missing():
    assert bg.guard_for(SimpleNamespace()) is None
    assert bg.guard_for(SimpleNamespace(budget=config_loader.BudgetConfig())) is None


# ---------------------------------------------------------------------------
# Интеграция с forge_stage
# ---------------------------------------------------------------------------

DAY = "2026-11-10"


class FakeCfg:
    def __init__(self, root: Path, per_day: int = 2):
        self.root = root
        self.forge = config_loader.ForgeConfig(videos_per_device_per_day=per_day, max_parallel_jobs=1,
                                               retry_backoff_sec=0, scenes_per_video=[1, 2],
                                               video_duration_sec=[10, 15], scene_duration_sec=5)

    @property
    def queue_db_path(self) -> Path:
        return self.root / "queue.db"

    def today(self) -> str:
        return DAY


@dataclass
class FakeRegistry:
    devices: list

    def get_active_devices(self, niches=None, device_ids=None):
        return [d for d in self.devices if not device_ids or d["device_id"] in device_ids]


@dataclass
class FakeForge:
    gpu_seconds: float | None = 360.0
    status: str = "ok"
    raise_exc: bool = False
    delay: float = 0.0
    calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def generate_one(self, device, idx, day):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raise_exc:
            raise RuntimeError("Cloud Run упал")
        return SimpleNamespace(status=self.status, error=None if self.status == "ok" else self.status,
                               gcp_execution_name="executions/x", gpu_seconds=self.gpu_seconds,
                               scenes=[SimpleNamespace(role="master"), SimpleNamespace(role="scene"),
                                       SimpleNamespace(role="scene")])


SETTINGS = forge_stage.RunSettings(max_workers=1, retry_backoff_sec=0, heartbeat_sec=0.2, poll_sec=0.01)


@pytest.fixture
def env(tmp_path, clock):
    cfg = FakeCfg(tmp_path)
    reg = FakeRegistry([{"device_id": f"acc_{i:02d}", "niche": "n", "accounts": {}} for i in range(5)])
    g = bg.BudgetGuard(budget(monthly_limit_usd=0.3), tmp_path / "budget.json", now=clock)
    return cfg, reg, g


def run(cfg, reg, fake, g, **kw):
    return forge_stage.stage_forge(cfg, reg, forge=fake, settings=SETTINGS, guard=g, **kw)


def test_pauses_when_limit_reached_and_keeps_queue(env):
    cfg, reg, g = env
    fake = FakeForge()  # 360 с GPU на задачу = $0.0672
    assert run(cfg, reg, fake, g) == forge_stage.EXIT_BUDGET_PAUSED
    q = JobQueue(cfg.queue_db_path)
    st = q.stats(DAY)
    assert fake.calls == 4 and st["done"] == 4 and st["pending"] == 6 and st["failed"] == 0
    assert all(j.attempts == 0 for j in q.jobs(DAY, status="pending"))  # попытка не списана
    bs = g.status()
    assert bs["spent_usd"] == pytest.approx(4 * 360 * RATE / 3600) and bs["reserved_usd"] == 0
    assert g.avg_scene_gpu_sec() == 180  # 360 с на 2 сцены


def test_exhausted_at_start_does_not_call_forge(env):
    cfg, reg, g = env
    g.record_spend(0.3, "earlier")
    fake = FakeForge()
    assert run(cfg, reg, fake, g) == forge_stage.EXIT_BUDGET_PAUSED
    assert fake.calls == 0 and JobQueue(cfg.queue_db_path).stats(DAY)["pending"] == 10


def test_continues_next_month_and_after_reset(env, clock):
    cfg, reg, g = env
    assert run(cfg, reg, FakeForge(), g) == 4
    assert run(cfg, reg, FakeForge(), g, resume=True) == 4  # тот же месяц — ничего нового
    clock.t = T0 + timedelta(days=30)                         # декабрь
    assert run(cfg, reg, FakeForge(), g, resume=True) == 4
    assert JobQueue(cfg.queue_db_path).stats(DAY)["done"] == 8
    fake = FakeForge()
    assert run(cfg, reg, fake, g, resume=True, reset_budget=True) == 0  # оставшиеся 2 влезли
    assert fake.calls == 2 and JobQueue(cfg.queue_db_path).stats(DAY)["done"] == 10


def test_no_gpu_statuses_cost_nothing(env):
    cfg, reg, g = env
    assert run(cfg, reg, FakeForge(status="skipped"), g) == 0
    assert g.status()["spent_usd"] == 0 and g.status()["reserved_usd"] == 0


def test_wall_clock_used_when_gcp_duration_missing(env):
    cfg, reg, g = env
    assert run(cfg, reg, FakeForge(gpu_seconds=None, delay=0.05), g) == 0
    spent = g.status()["spent_usd"]
    assert 0 < spent < 0.01  # учтено фактическое время вызова, а не 0 и не оценка


def test_failed_gcp_job_is_still_billed(env):
    cfg, reg, g = env
    cfg.forge.retry_max_attempts = 1
    assert run(cfg, reg, FakeForge(status="gcp_failed", gpu_seconds=36.0), g) == forge_stage.EXIT_FAILURE_RATIO
    assert g.status()["spent_usd"] > 0


def test_exception_records_spend_and_fails_job(env):
    cfg, reg, g = env
    cfg.forge.retry_max_attempts = 1
    run(cfg, reg, FakeForge(raise_exc=True), g)
    assert g.status()["reserved_usd"] == 0
    assert JobQueue(cfg.queue_db_path).stats(DAY)["failed"] >= 1


def test_without_budget_block_runs_as_before(tmp_path):
    cfg = FakeCfg(tmp_path)
    reg = FakeRegistry([{"device_id": "acc_00", "niche": "n", "accounts": {}}])
    fake = FakeForge()
    assert forge_stage.stage_forge(cfg, reg, forge=fake, settings=SETTINGS) == 0 and fake.calls == 2


def test_orchestrator_continues_after_budget_pause(tmp_path, monkeypatch):
    import orchestrator as orch_mod
    o = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    o.cfg = FakeCfg(tmp_path)
    o.registry = FakeRegistry([{"device_id": "acc_00", "niche": "n", "accounts": {}}])
    monkeypatch.setattr(forge_stage, "stage_forge", lambda *a, **kw: forge_stage.EXIT_BUDGET_PAUSED)
    called = []
    monkeypatch.setitem(orch_mod.STAGES, "ghost", lambda ctx: called.append(1) or ({}, True, True))
    results = o.run("forge,ghost", None)
    assert [r.name for r in results] == ["forge", "ghost"] and called
    assert results[0].payload["budget_paused"] is True and not results[0].ok
    assert orch_mod.exit_code_for(results) == 1


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

def _write_cfg(tmp_path: Path, mutate=None) -> Path:
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(raw)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    niches = {n: {"marker": m} for n, m in raw["devices_registry"]["niche_markers"].items()}
    (tmp_path / "niches.json").write_text(json.dumps({"niches": niches}), encoding="utf-8")
    return tmp_path / "config.json"


def test_config_budget_loaded(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="config_loader"):
        cfg = config_loader.load_config(_write_cfg(tmp_path))
    assert not any("неизвестные ключи" in r.getMessage() for r in caplog.records)
    b = cfg.budget
    assert (b.enabled, b.monthly_limit_usd, b.cost_per_gpu_hour, b.alert_thresholds) == \
        (True, 300.0, 0.672, [50, 90, 100])
    assert cfg.budget_state_path == tmp_path / "data" / "budget.json"
    assert isinstance(bg.guard_for(cfg), bg.BudgetGuard)


def test_config_without_budget_block_disables_guard(tmp_path):
    cfg = config_loader.load_config(_write_cfg(tmp_path, lambda r: r.pop("budget")))
    assert cfg.budget.enabled is False and bg.guard_for(cfg) is None


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["budget"].update(monthly_limit_usd=0), "monthly_limit_usd"),
    (lambda r: r["budget"].update(cost_per_gpu_hour=-1), "cost_per_gpu_hour"),
    (lambda r: r["budget"].update(alert_thresholds=[90, 50]), "alert_thresholds"),
    (lambda r: r["budget"].update(alert_thresholds=[50, 150]), "alert_thresholds"),
    (lambda r: r["budget"].update(billing_timezone="Mars/Olympus"), "billing_timezone"),
])
def test_config_budget_validation(tmp_path, mutate, msg):
    with pytest.raises(ValueError, match=msg):
        config_loader.load_config(_write_cfg(tmp_path, mutate))


def test_gcp_client_status_and_duration(monkeypatch):
    """Настоящие protobuf-объекты google-cloud-run: успех, ещё идёт, провал, успех без условий."""
    run_v2 = pytest.importorskip("google.cloud.run_v2")
    import gcp_client
    # без google-cloud-storage модуль обнуляет и run_v2 — подставляем настоящий
    monkeypatch.setattr(gcp_client, "run_v2", run_v2)
    Cond = run_v2.Condition
    status = gcp_client.GcpClient._execution_status

    ok = run_v2.Execution(name="executions/ok", task_count=1, succeeded_count=1, conditions=[
        Cond(type_="Completed", state=Cond.State.CONDITION_SUCCEEDED)])
    ok.start_time = T0
    ok.completion_time = T0 + timedelta(seconds=420)
    st = status(ok)
    assert st["status"] == "SUCCEEDED" and st["duration_sec"] == 420

    running = run_v2.Execution(name="executions/run", task_count=1, conditions=[
        Cond(type_="Completed", state=Cond.State.CONDITION_PENDING)])
    running.start_time = T0
    assert status(running) is None

    failed = run_v2.Execution(name="executions/f", task_count=1, failed_count=1, conditions=[
        Cond(type_="Completed", state=Cond.State.CONDITION_FAILED, message="OOM")])
    failed.completion_time = T0
    st = status(failed)
    assert st["status"] == "FAILED" and st["reason"] == "OOM" and "duration_sec" not in st

    bare = run_v2.Execution(name="executions/b", task_count=1, succeeded_count=1)
    bare.completion_time = T0
    assert status(bare)["status"] == "SUCCEEDED"
