"""tests/test_orchestrator.py — встраивание account_health и очереди forge в orchestrator.

Запуск:  python -m pytest -q tests/test_orchestrator.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_loader  # noqa: E402
import forge_stage  # noqa: E402
import orchestrator as orch_mod  # noqa: E402
from job_queue import JobQueue  # noqa: E402

DAY = "2026-11-10"


class FakeCfg:
    def __init__(self, root: Path):
        self.root = root
        self.timezone = "UTC"
        self.forge = config_loader.ForgeConfig(videos_per_device_per_day=1)
        self.account_health = config_loader.AccountHealthConfig(enabled=False)
        self.uniqueness = {"platforms": ["tiktok"]}

    @property
    def queue_db_path(self) -> Path:
        return self.root / "queue.db"

    @property
    def videos_root(self) -> Path:
        return self.root / "videos"

    def path(self, key: str) -> Path:
        return self.root / key

    def today(self) -> str:
        return DAY


@dataclass
class FakeRegistry:
    devices: list

    def get_active_devices(self, niches=None, device_ids=None):
        return [d for d in self.devices if not device_ids or d["device_id"] in device_ids]


def make_orch(tmp_path: Path) -> orch_mod.Orchestrator:
    o = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    o.cfg = FakeCfg(tmp_path)
    o.registry = FakeRegistry([{"device_id": "aria_fight", "niche": "wrestling_female",
                                "accounts": {"tiktok": {"username": "aria.fight"}}}])
    return o


def test_account_health_runs_before_mimic():
    stages = orch_mod.Orchestrator.parse_stages("full")
    assert stages.index("account_health") < stages.index("mimic") < stages.index("forge")
    assert orch_mod.Orchestrator.parse_stages("health") == ["account_health"]


def test_forge_stage_uses_queue_and_reports_stats(tmp_path, monkeypatch):
    o = make_orch(tmp_path)

    def fake_stage(cfg, registry, resume, device_ids, day):
        q = JobQueue(cfg.queue_db_path)
        q.enqueue("aria_fight", day, 1)
        q.register_worker("w")
        j = q.claim("w")
        q.complete(j.id, "done", worker_id="w")
        return 0

    monkeypatch.setattr(forge_stage, "stage_forge", fake_stage)
    results = o.run("forge", None)
    r = results[0]
    assert r.ok and r.usable and r.payload["stats"]["done"] == 1
    assert orch_mod.exit_code_for(results) == 0


@pytest.mark.parametrize("code", [3, 130])
def test_forge_special_exit_codes_stop_cycle(tmp_path, monkeypatch, code):
    o = make_orch(tmp_path)
    monkeypatch.setattr(forge_stage, "stage_forge", lambda *a, **kw: code)
    called = []
    monkeypatch.setitem(orch_mod.STAGES, "ghost", lambda ctx: called.append(1) or ({}, True, True))
    results = o.run("forge,ghost", None)
    assert [r.name for r in results] == ["forge"] and not called
    assert orch_mod.exit_code_for(results) == code


def test_resume_flag_reaches_forge_stage(tmp_path, monkeypatch):
    o = make_orch(tmp_path)
    seen = {}

    def fake_stage(cfg, registry, resume, device_ids, day):
        seen["resume"] = resume
        return 0

    monkeypatch.setattr(forge_stage, "stage_forge", fake_stage)
    o.run("forge", None, resume=True)
    assert seen["resume"] is True


def test_account_health_stage_disabled_is_noop(tmp_path):
    results = make_orch(tmp_path).run("account_health", None)
    assert results[0].ok and results[0].payload == {"enabled": False}
