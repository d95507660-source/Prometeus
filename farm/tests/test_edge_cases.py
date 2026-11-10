"""tests/test_edge_cases.py — граничные случаи, найденные при подготовке к передаче проекта
(ruff/mypy-аудит): None вместо статуса, порт без значения, запись при сбое, ссылки и т.п.

Запуск:  python -m pytest -q tests/test_edge_cases.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import diversity  # noqa: E402
import job_queue  # noqa: E402

# --- diversity.atomic_write_text --------------------------------------------------

def test_atomic_write_keeps_old_file_on_failure(tmp_path, monkeypatch):
    f = tmp_path / "a.json"
    diversity.atomic_write_text(f, "old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(diversity.os, "replace", boom)
    with pytest.raises(OSError):
        diversity.atomic_write_text(f, "new")
    assert f.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]  # tmp убран


@pytest.mark.parametrize("bad", ["", "a/b", "../x", "x" * 65, None, 5, "a b"])
def test_validate_device_id_rejects(bad):
    with pytest.raises(ValueError):
        diversity.validate_device_id(bad)


def test_cohort_rank_adds_missing_device():
    assert diversity.cohort_rank(["b", "a"], "c") == (2, 3)
    assert diversity.cohort_rank([], "x") == (0, 1)


# --- job_queue -----------------------------------------------------------------------

def test_enqueue_rejects_zero_attempts(tmp_path):
    q = job_queue.JobQueue(tmp_path / "q.db")
    with pytest.raises(ValueError):
        q.enqueue("dev", "2026-01-01", 1, max_attempts=0)


def test_complete_rejects_unknown_status(tmp_path):
    q = job_queue.JobQueue(tmp_path / "q.db")
    with pytest.raises(ValueError):
        q.complete(1, "maybe")


def test_complete_ignores_job_that_is_not_running(tmp_path):
    q = job_queue.JobQueue(tmp_path / "q.db")
    q.enqueue("dev", "2026-01-01", 1)
    assert q.complete(1, "done") is False  # не была захвачена — результат не пишется
    assert q.jobs()[0].status == "pending"


def test_enqueue_updates_priority_only_while_pending(tmp_path):
    q = job_queue.JobQueue(tmp_path / "q.db")
    q.enqueue("dev", "2026-01-01", 1, priority=5)
    q.enqueue("dev", "2026-01-01", 1, priority=10)
    assert q.jobs()[0].priority == 10
    q.register_worker("w")
    j = q.claim("w")
    q.complete(j.id, "done", worker_id="w")
    q.enqueue("dev", "2026-01-01", 1, priority=5)
    assert q.jobs()[0].priority == 10 and q.jobs()[0].status == "done"


# --- devices_registry ---------------------------------------------------------------

def _registry(tmp_path, devices):
    from devices_registry import DevicesRegistry
    (tmp_path / "devices.json").write_text(json.dumps({"devices": devices}), encoding="utf-8")
    cfg = {"mimic": {"active_niches": ["fit"]},
           "devices_registry": {"require_username_with_niche_marker": True, "niche_markers": {"fit": ".lift"}}}
    return DevicesRegistry(cfg, tmp_path / "devices.json", tmp_path / "niches.json")


def test_registry_rejects_missing_port_and_bad_marker(tmp_path):
    reg = _registry(tmp_path, [
        {"device_id": "ok_dev", "niche": "fit", "adb": {"host": "10.0.0.1", "port": 5555},
         "accounts": {"tiktok": {"username": "a.lift"}}},
        {"device_id": "no_port", "niche": "fit", "adb": {"host": "10.0.0.2"},
         "accounts": {"tiktok": {"username": "b.lift"}}},
        {"device_id": "no_marker", "niche": "fit", "adb": {"host": "10.0.0.3", "port": 5555},
         "accounts": {"tiktok": {"username": "plain"}}},
        {"device_id": "dup_adb", "niche": "fit", "adb": {"host": "10.0.0.1", "port": 5555},
         "accounts": {"tiktok": {"username": "c.lift"}}},
    ])
    assert [d["device_id"] for d in reg.get_active_devices()] == ["ok_dev"]


# --- gcp_client: run_job не возвращает None -------------------------------------------

def test_run_job_fallback_never_returns_none(monkeypatch):
    run_v2 = pytest.importorskip("google.cloud.run_v2")
    import gcp_client
    monkeypatch.setattr(gcp_client, "run_v2", run_v2)
    client = gcp_client.GcpClient.__new__(gcp_client.GcpClient)
    client.project_id, client.region, client.job_name = "p", "us-central1", "j"
    still_running = run_v2.Execution(name="executions/x", task_count=1)

    class Op:
        metadata = None

        def result(self, timeout=None):
            return still_running

    client._jobs = SimpleNamespace(run_job=lambda request: Op())
    assert client.run_job({"A": "1"}, timeout_sec=1) == {"status": "UNKNOWN", "execution": "executions/x"}


# --- autoposter: без подключения не публикует ----------------------------------------

def test_autoposter_build_caption_keeps_disclosure_when_truncated():
    import autoposter
    cfg = {"monetization": {"disclosure": {"enabled": True, "platforms": ["tiktok"], "text": "AI-generated"}}}
    text = autoposter.build_caption_text("x" * 5000, ["#a"], cfg, "tiktok")
    assert len(text) <= autoposter.CAPTION_LIMITS["tiktok"] and text.endswith("AI-generated")
    yt = autoposter.build_caption_text("y" * 300, ["#a", "#b", "#c"], cfg, "youtube")
    assert len(yt) <= 100  # для YouTube пометка зависит от platforms — здесь её нет


# --- link_rotator + cpa_matcher: связка аккаунт → ссылка ------------------------------

def test_link_for_registry_account_id_is_valid():
    from link_rotator import get_tracking_link
    from revenue_tracker import accounts_from_registry
    acc = accounts_from_registry([{"device_id": "aria_fight", "accounts": {"tiktok": {}}}])[0]
    assert "utm_campaign=account_aria_fight_tiktok" in get_tracking_link("https://p.example", acc, 1, "tiktok")


def test_device_id_with_dash_is_not_a_valid_account_id():
    from link_rotator import get_tracking_link
    with pytest.raises(ValueError):
        get_tracking_link("https://p.example", "aria-fight_tiktok", 1, "tiktok")


# --- env: секреты не читаются при импорте --------------------------------------------

def test_importing_modules_does_not_require_secrets(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "TOGETHER_API_KEY", "LOCAL_WORKER_TOKEN", "MODEL_SERVER_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    import importlib
    for name in ("claude_common", "claude_analyst", "claude_builder", "local_worker", "budget_guard",
                 "revenue_tracker", "account_health", "forge_stage"):
        importlib.import_module(name)
    assert "ANTHROPIC_API_KEY" not in os.environ
