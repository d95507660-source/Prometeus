"""orchestrator.py — главный оркестратор конвейера.

Этапы выполняются строго по порядку (каждый зависит от предыдущего).
--parallel ускоряет работу ВНУТРИ этапа: несколько устройств одновременно (ghost, autoposter, pulse).
forge всегда идёт через очередь SQLite (forge_stage.py), параллельность — forge.max_parallel_jobs.
--resume — forge доделывает незавершённую очередь вместо постановки новых задач.

Коды выхода: 0 — все этапы ok; 1 — были ошибки; 2 — неверные аргументы;
             3 — forge: доля окончательно упавших задач > forge.stop_on_failure_ratio;
             130 — forge прерван (Ctrl+C), очередь сохранена.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config_loader import AppConfig, apply_env_overrides, load_config, load_dotenv_if_present
from devices_registry import DevicesRegistry

log = logging.getLogger(__name__)

DEFAULT_FULL_CYCLE = ["trend_analyzer", "hashtag_updater", "account_health", "mimic", "forge",
                      "ghost", "autoposter", "pulse"]
SCRIPT_ALIASES = {"poster": "autoposter", "trends": "trend_analyzer", "collect": "trends_collector",
                  "hashtags": "hashtag_updater", "gen": "forge", "uniq": "ghost", "test": "pulse",
                  "plan": "mimic", "health": "account_health"}
CRITICAL = {"forge", "ghost"}  # без их результата дальше идти бессмысленно
EXIT_FORGE_FAILURE_RATIO = 3
EXIT_INTERRUPTED = 130


@dataclass
class StageResult:
    name: str
    ok: bool                 # без единой ошибки
    usable: bool             # есть результат, с которым можно идти дальше
    elapsed_sec: float
    payload: Any = None
    error: str | None = None
    exit_code: int = 0       # особый код процесса (forge: 3 или 130), 0 — обычная логика


@dataclass
class Ctx:
    cfg: AppConfig
    devices: list[dict]
    parallel: bool
    day: str
    registry: Any = None
    resume: bool = False
    reset_budget: bool = False


def _map_devices(ctx: Ctx, fn: Callable[[dict], Any], workers: int) -> dict[str, Any]:
    def safe(d: dict):
        try:
            return d["device_id"], fn(d)
        except Exception as e:
            log.exception("%s: ошибка", d["device_id"])
            return d["device_id"], {"error": str(e)}
    if not ctx.parallel or workers <= 1 or len(ctx.devices) <= 1:
        return dict(safe(d) for d in ctx.devices)
    with ThreadPoolExecutor(max_workers=min(workers, len(ctx.devices))) as ex:
        return dict(ex.map(safe, ctx.devices))


def _has_errors(payload: dict) -> bool:
    return any(isinstance(v, dict) and v.get("error") for v in payload.values())


# ---------------------------------------------------------------------------
# Этапы: возвращают (payload, ok, usable)
# ---------------------------------------------------------------------------

def stage_trends_collector(ctx: Ctx):
    from trends_collector import collect_all
    p = collect_all(ctx.cfg)
    return p, all(v >= 0 for s in p.values() for v in s.values()), True


def stage_trend_analyzer(ctx: Ctx):
    from trend_analyzer import analyze_all
    p = analyze_all(ctx.cfg, os.getenv("APIFY_API_TOKEN", ""))
    ok = not any(v.get("error") for v in p.values())
    return {k: {"viral": len(v.get("items", [])), "error": v.get("error")} for k, v in p.items()}, ok, True


def stage_hashtag_updater(ctx: Ctx):
    from hashtag_updater import update_hashtags
    p = update_hashtags(ctx.cfg)
    return {k: len(v) for k, v in p.items()}, True, True


def stage_account_health(ctx: Ctx):
    from account_health import AccountHealth
    health = AccountHealth(ctx.cfg)
    if not health.enabled:
        return {"enabled": False}, True, True
    p = health.run(ctx.devices)
    # проблемы аккаунтов не ломают цикл: mimic/autoposter сами пропустят такие аккаунты
    return p, not any(v.get("last_error") for v in p.values()), True


def stage_mimic(ctx: Ctx):
    from mimic import Mimic
    p = Mimic(ctx.cfg, ctx.registry).run([d["device_id"] for d in ctx.devices])
    return p, bool(p), bool(p)


def stage_forge(ctx: Ctx):
    """forge через очередь SQLite: повторы, приоритет новых аккаунтов, --resume."""
    from forge_stage import (EXIT_BUDGET_PAUSED, EXIT_FAILURE_RATIO,
                             EXIT_INTERRUPTED as FS_INTERRUPTED, stage_forge)
    from job_queue import JobQueue

    code = stage_forge(ctx.cfg, ctx.registry, resume=ctx.resume,
                       device_ids=[d["device_id"] for d in ctx.devices], day=ctx.day,
                       reset_budget=ctx.reset_budget)
    queue = JobQueue(ctx.cfg.queue_db_path)
    stats = queue.stats(None if ctx.resume else ctx.day)
    payload = {"exit_code": code, "stats": stats,
               "failed": [{"device": j.device_id, "day": j.day, "source_idx": j.source_idx,
                           "attempts": j.attempts, "error": (j.error or "")[:300]}
                          for j in queue.jobs(None if ctx.resume else ctx.day, status="failed")]}
    if code == EXIT_FAILURE_RATIO:
        raise StageExit(payload, EXIT_FORGE_FAILURE_RATIO)
    if code == FS_INTERRUPTED:
        raise StageExit(payload, EXIT_INTERRUPTED)
    if code == EXIT_BUDGET_PAUSED:
        # цикл не останавливаем (usable=True): ghost/autoposter обработают уже готовые ролики,
        # pulse проверит устройства; ok=False — итоговый код процесса 1
        payload["budget_paused"] = True
        log.warning("forge: пауза по бюджету — дальше идут только уже готовые ролики")
        return payload, False, True
    return payload, stats["failed"] == 0 and stats["total"] > 0, stats["done"] > 0


def stage_ghost(ctx: Ctx):
    from diversity import niche_cohorts
    from ghost import uniquify_device
    all_platforms = ctx.cfg.uniqueness.get("platforms", ["tiktok", "instagram", "youtube"])
    # когорты по ВСЕМ активным устройствам — стиль канала не зависит от --device
    cohorts = niche_cohorts(ctx.registry.get_active_devices() if ctx.registry else ctx.devices)

    def one(d: dict):
        accounts = d.get("accounts") or {}
        platforms = [p for p in all_platforms
                     if p in accounts and accounts[p].get("enabled", True)]
        return uniquify_device(d["device_id"], ctx.cfg.videos_root, platforms, day=ctx.day,
                               cohort=cohorts.get(d.get("niche", "")),
                               style_index=d.get("style_index"))

    workers = max(1, (os.cpu_count() or 2) // 2)  # ffmpeg сам использует несколько ядер
    p = _map_devices(ctx, one, workers)
    produced = sum(len(v) for v in p.values() if isinstance(v, dict) and not v.get("error"))
    return p, not _has_errors(p), produced > 0


def stage_autoposter(ctx: Ctx):
    from autoposter import post_device
    p = _map_devices(ctx, lambda d: post_device(d, ctx.cfg, ctx.day), workers=len(ctx.devices))
    posted = sum(v.get("posted", 0) for v in p.values() if isinstance(v, dict))
    failed = sum(v.get("failed", 0) for v in p.values() if isinstance(v, dict))
    return p, not _has_errors(p) and failed == 0, posted > 0 or failed == 0


def stage_pulse(ctx: Ctx):
    from pulse import pulse_device
    p = _map_devices(ctx, pulse_device, workers=len(ctx.devices))
    ok = all(isinstance(v, dict) and v.get("online") and not v.get("warnings") for v in p.values())
    return p, ok, True


class StageExit(Exception):
    """Этап требует остановить цикл с особым кодом выхода процесса."""

    def __init__(self, payload: Any, exit_code: int):
        super().__init__(f"exit_code={exit_code}")
        self.payload = payload
        self.exit_code = exit_code


STAGES: dict[str, Callable[[Ctx], tuple[Any, bool, bool]]] = {
    "trends_collector": stage_trends_collector,
    "trend_analyzer": stage_trend_analyzer,
    "hashtag_updater": stage_hashtag_updater,
    "account_health": stage_account_health,
    "mimic": stage_mimic,
    "forge": stage_forge,
    "ghost": stage_ghost,
    "autoposter": stage_autoposter,
    "pulse": stage_pulse,
}


@contextmanager
def single_instance(lock_path: Path):
    """Не даёт запустить два оркестратора одновременно (например, из cron)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    try:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass  # Windows: без блокировки
        except BlockingIOError:
            raise SystemExit("оркестратор уже запущен (lock: %s)" % lock_path)
        yield
    finally:
        f.close()


class Orchestrator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.registry = DevicesRegistry.from_config(cfg)

    @staticmethod
    def parse_stages(script: str) -> list[str]:
        script = SCRIPT_ALIASES.get(script, script)
        names = DEFAULT_FULL_CYCLE if script == "full" else \
            [SCRIPT_ALIASES.get(s.strip(), s.strip()) for s in script.split(",") if s.strip()]
        unknown = [s for s in names if s not in STAGES]
        if unknown:
            raise ValueError(f"неизвестные этапы: {unknown}; доступны: {list(STAGES)}")
        return list(names)

    def _run_stage(self, name: str, ctx: Ctx) -> StageResult:
        t0 = time.time()
        try:
            payload, ok, usable = STAGES[name](ctx)
            return StageResult(name, ok, usable, time.time() - t0, payload=payload)
        except StageExit as e:
            return StageResult(name, False, False, time.time() - t0, payload=e.payload,
                               error=f"остановка с кодом {e.exit_code}", exit_code=e.exit_code)
        except Exception as e:
            log.exception("этап %s упал", name)
            return StageResult(name, False, False, time.time() - t0, error=str(e))

    def run(self, script: str, device_ids: list[str] | None, parallel: bool = False,
            dry_run: bool = False, day: str | None = None,
            resume: bool = False, reset_budget: bool = False) -> list[StageResult]:
        stages = self.parse_stages(script)
        devices = self.registry.get_active_devices(device_ids=device_ids)
        ctx = Ctx(self.cfg, devices, parallel, day or self.cfg.today(), registry=self.registry,
                  resume=resume, reset_budget=reset_budget)
        if dry_run:
            return self._dry_run(stages, ctx)
        if not devices and any(s not in ("trends_collector", "trend_analyzer", "hashtag_updater")
                               for s in stages):
            log.error("нет активных устройств — проверь devices.json")
        results: list[StageResult] = []
        for s in stages:
            log.info("=== этап %s ===", s)
            r = self._run_stage(s, ctx)
            log.info("этап %s: ok=%s usable=%s %.1fs %s", r.name, r.ok, r.usable,
                     r.elapsed_sec, r.error or "")
            results.append(r)
            if r.exit_code:
                log.error("этап %s потребовал остановки (код %d) — цикл остановлен", s, r.exit_code)
                break
            if s in CRITICAL and not r.usable:
                log.error("критичный этап %s не дал результата — цикл остановлен", s)
                break
        return results

    def _dry_run(self, stages: list[str], ctx: Ctx) -> list[StageResult]:
        log.info("[dry-run] день=%s устройств=%d этапы=%s", ctx.day, len(ctx.devices), stages)
        for d in ctx.devices:
            log.info("  %s ниша=%s голос=%s аккаунты=%s", d["device_id"], d["niche"],
                     d.get("voice_id"),
                     [p for p, a in (d.get("accounts") or {}).items() if a.get("enabled", True)])
        return [StageResult(s, True, True, 0.0, payload={"dry_run": True}) for s in stages]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    ap.add_argument("--script", default="full",
                    help="full или этапы через запятую: " + ",".join(STAGES))
    ap.add_argument("--device", action="append", default=None)
    ap.add_argument("--day", default=None, help="YYYY-MM-DD (по умолчанию сегодня)")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="forge: доделать незавершённую очередь (после сбоя или Ctrl+C)")
    ap.add_argument("--reset-budget", action="store_true",
                    help="forge: обнулить трату месяца в data/budget.json перед запуском")
    a = ap.parse_args()

    load_dotenv_if_present(Path(a.config).resolve().parent)
    cfg = apply_env_overrides(load_config(a.config, log_name="orchestrator"))
    orch = Orchestrator(cfg)
    try:
        orch.parse_stages(a.script)
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        sys.exit(2)
    with single_instance(cfg.path("logs") / "orchestrator.lock"):
        results = orch.run(a.script, a.device, parallel=a.parallel, dry_run=a.dry_run, day=a.day,
                           resume=a.resume, reset_budget=a.reset_budget)
    sys.stdout.write(json.dumps(summarize(results), ensure_ascii=False, indent=2, default=str) + "\n")
    sys.exit(exit_code_for(results))


def summarize(results: list[StageResult]) -> list[dict]:
    return [{"stage": r.name, "ok": r.ok, "usable": r.usable,
             "elapsed_sec": round(r.elapsed_sec, 2), "error": r.error, "result": r.payload}
            for r in results]


def exit_code_for(results: list[StageResult]) -> int:
    for r in results:
        if r.exit_code:
            return r.exit_code
    return 0 if results and all(r.ok for r in results) else 1


if __name__ == "__main__":
    main()
