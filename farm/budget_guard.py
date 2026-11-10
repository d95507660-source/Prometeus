"""budget_guard.py — месячный лимит расходов на генерацию (Cloud Run GPU).

Состояние: data/budget.json (запись атомарная, под блокировкой — безопасно из 8 потоков forge и
из нескольких процессов на одной машине)
  {"month": "2026-11", "spent_usd": 12.3, "reserved": {"<job>": {"usd": 0.4, "at": ts}},
   "recorded_jobs": [...], "alerts_sent": [50], "calibration": {"gpu_sec": 3600, "scenes": 30},
   "history": {"2026-10": 280.1}, "resets": [...]}

Как работает:
  * перед запуском Cloud Run Job forge_stage резервирует оценку (try_reserve). Проверка учитывает
    уже потраченное + резервы задач, которые сейчас выполняются, + новую оценку;
  * после задачи резерв снимается и записывается фактическая стоимость (record_spend);
  * оценка = сцены × среднее время на сцену × $/час / 3600. Среднее время на сцену учится по
    реальным задачам (включает холодный старт и загрузку моделей); до первых данных —
    budget.default_scene_gpu_sec;
  * лимит исчерпан → новые задачи остаются в очереди (pending). Новый месяц биллинга
    (budget.billing_timezone) или ручной сброс (--reset-budget) — и следующий запуск их доделает;
  * пороги alert_thresholds (50/90/100 %) — предупреждение в лог один раз за месяц на порог.

Ограничения: это оценка по времени GPU, а не счёт GCP. В реальный счёт входят ещё CPU/память
инстанса (если их нет в cost_per_gpu_hour), Cloud Storage, исходящий трафик, LLM-запросы.
Задачи, уже запущенные к моменту исчерпания, доработают — факт может немного превысить лимит.
Поэтому нужен и бюджет в самом GCP (ниже).

──────────────────────────────────────────────────────────────────────────────────────────────
Бюджет и оповещения в GCP (страховка поверх этого модуля)
──────────────────────────────────────────────────────────────────────────────────────────────
Console:  Billing → (ваш платёжный аккаунт) → Budgets & alerts → Create budget
  1. Scope: проект prometheus-prod-510404 (или весь платёжный аккаунт), все сервисы.
     Credits: если хотите видеть расход ДО вычета кредитов — снимите галочки кредитов
     [ПРОВЕРИТЬ ПО ДОКАМ: названия пунктов меняются].
  2. Amount: Specified amount → 300 USD (тот же monthly_limit_usd), Monthly.
  3. Actions → Threshold rules: 50 %, 90 %, 100 % от Actual; можно добавить 100 % от Forecasted —
     письмо придёт заранее, по прогнозу.
  4. Notifications: e-mail администраторам биллинга; при желании — Pub/Sub-топик
     (пригодится для автоматической реакции).
  5. Finish.
Важно: бюджет GCP только ОПОВЕЩАЕТ и сам расходы не останавливает. Автоостановка — отдельная
настройка: Pub/Sub → Cloud Run function, которая, например, удаляет триггер или отключает
биллинг проекта. Отключение биллинга останавливает ВСЕ сервисы проекта и может привести к удалению
ресурсов — настраивайте осознанно [ПРОВЕРИТЬ ПО ДОКАМ: «Disable billing usage with notifications»].

CLI (команду печатает `python budget_guard.py --print-gcp-budget-cmd`, сама ничего не выполняет):
  gcloud billing budgets create --billing-account=XXXXXX-XXXXXX-XXXXXX \\
      --display-name="prometheus monthly" --budget-amount=300USD \\
      --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
  [ПРОВЕРИТЬ ПО ДОКАМ: флаги `gcloud billing budgets create` в вашей версии gcloud]
  ID платёжного аккаунта: `gcloud billing accounts list` или переменная GCP_BILLING_ACCOUNT.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diversity import atomic_write_text

log = logging.getLogger("budget_guard")

RESERVATION_TTL_SEC = 4 * 3600      # резерв упавшего процесса не держит бюджет вечно
JOB_KEY_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
BILLING_ACCOUNT_RE = re.compile(r"^[0-9A-F]{6}-[0-9A-F]{6}-[0-9A-F]{6}$")
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _round(v: float) -> float:
    return round(float(v), 6)


def _check_amount(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError(f"{what}: ожидалось неотрицательное число, получено {v!r}")
    return float(v)


def _check_key(job_id: Any) -> str:
    s = str(job_id) if isinstance(job_id, int) and not isinstance(job_id, bool) else job_id
    if not isinstance(s, str) or not JOB_KEY_RE.fullmatch(s):
        raise ValueError(f"некорректный job_id: {job_id!r}")
    return s


class BudgetGuard:
    def __init__(self, cfg_budget: Any, state_path: str | Path,
                 now: Callable[[], datetime] | None = None):
        self.bc = cfg_budget
        self.path = Path(state_path)
        self._now = now or (lambda: datetime.now(timezone.utc))
        key = str(self.path.resolve())
        with _THREAD_LOCKS_GUARD:
            self._tlock = _THREAD_LOCKS.setdefault(key, threading.Lock())

    # --- служебное -------------------------------------------------------------

    @property
    def limit(self) -> float:
        return float(self.bc.monthly_limit_usd)

    @property
    def rate(self) -> float:
        return float(self.bc.cost_per_gpu_hour)

    def month_key(self) -> str:
        from zoneinfo import ZoneInfo
        return self._now().astimezone(ZoneInfo(self.bc.billing_timezone)).strftime("%Y-%m")

    def cost_for_seconds(self, gpu_sec: float) -> float:
        return _round(_check_amount(gpu_sec, "gpu_sec") * self.rate / 3600)

    @contextmanager
    def _locked(self) -> Iterator[dict]:
        """Поток + процесс: threading.Lock и flock на соседнем .lock-файле (на Windows — только поток)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tlock:
            with open(self.path.with_suffix(self.path.suffix + ".lock"), "a+") as lf:
                try:
                    import fcntl
                    fcntl.flock(lf, fcntl.LOCK_EX)
                except ImportError:
                    pass
                state = self._load()
                yield state
                self._save(state)

    def _fresh(self, month: str, spent: float = 0.0) -> dict:
        return {"month": month, "spent_usd": _round(spent), "reserved": {}, "recorded_jobs": [],
                "alerts_sent": [], "calibration": {"gpu_sec": 0.0, "scenes": 0},
                "history": {}, "resets": []}

    def _load(self) -> dict:
        month = self.month_key()
        if not self.path.exists():
            return self._fresh(month, float(self.bc.current_spent_usd))
        try:
            st = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(st, dict) or not isinstance(st.get("spent_usd"), (int, float)):
                raise ValueError("неожиданная структура")
        except Exception as e:
            # битый файл не обнуляет бюджет молча: считаем лимит исчерпанным до ручного разбора
            backup = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            self.path.replace(backup)
            log.error("budget.json повреждён (%s), копия %s — считаю бюджет исчерпанным; "
                      "проверьте и выполните --reset-budget", e, backup.name)
            return self._fresh(month, self.limit)
        for k, v in self._fresh(month).items():
            st.setdefault(k, v)
        if st["month"] != month:
            st["history"][st["month"]] = _round(st["spent_usd"])
            log.info("новый месяц биллинга %s: потрачено за %s — $%.2f, счётчик обнулён",
                     month, st["month"], st["spent_usd"])
            st.update(month=month, spent_usd=0.0, recorded_jobs=[], alerts_sent=[])
        now = time.time()
        stale = [k for k, r in st["reserved"].items() if now - float(r.get("at", 0)) > RESERVATION_TTL_SEC]
        for k in stale:
            log.warning("снимаю зависший резерв %s ($%.2f)", k, st["reserved"][k]["usd"])
            del st["reserved"][k]
        return st

    def _save(self, st: dict) -> None:
        st["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        atomic_write_text(self.path, json.dumps(st, ensure_ascii=False, indent=2))

    @staticmethod
    def _committed(st: dict) -> float:
        return float(st["spent_usd"]) + sum(float(r["usd"]) for r in st["reserved"].values())

    def _alerts(self, st: dict) -> None:
        pct = float(st["spent_usd"]) / self.limit * 100
        for t in self.bc.alert_thresholds:
            if pct >= t and t not in st["alerts_sent"]:
                st["alerts_sent"].append(t)
                log.warning("БЮДЖЕТ: потрачено %.0f%% месячного лимита ($%.2f из $%.2f, месяц %s)",
                            pct, st["spent_usd"], self.limit, st["month"])

    # --- API ---------------------------------------------------------------------

    def avg_scene_gpu_sec(self) -> float:
        with self._locked() as st:
            cal = st["calibration"]
        if cal.get("scenes", 0) > 0 and cal.get("gpu_sec", 0) > 0:
            return float(cal["gpu_sec"]) / int(cal["scenes"])
        return float(self.bc.default_scene_gpu_sec)

    def estimate_cost(self, n_scenes: int) -> float:
        """(сцены × среднее время на сцену × $/час) / 3600."""
        if isinstance(n_scenes, bool) or not isinstance(n_scenes, int) or n_scenes < 1:
            raise ValueError(f"n_scenes должен быть целым >= 1: {n_scenes!r}")
        return _round(n_scenes * self.avg_scene_gpu_sec() * self.rate / 3600)

    def can_spend(self, estimated_cost_usd: float) -> bool:
        est = _check_amount(estimated_cost_usd, "estimated_cost_usd")
        with self._locked() as st:
            return self._committed(st) + est <= self.limit + 1e-9

    def try_reserve(self, job_id: str, estimated_cost_usd: float) -> bool:
        """Атомарно: проверить лимит и зарезервировать оценку под задачу."""
        key, est = _check_key(job_id), _check_amount(estimated_cost_usd, "estimated_cost_usd")
        with self._locked() as st:
            if key in st["reserved"]:
                return True
            if self._committed(st) + est > self.limit + 1e-9:
                return False
            st["reserved"][key] = {"usd": _round(est), "at": time.time()}
            return True

    def release(self, job_id: str) -> None:
        """Снять резерв без траты (задача не запускала GPU)."""
        key = _check_key(job_id)
        with self._locked() as st:
            st["reserved"].pop(key, None)

    def record_spend(self, actual_cost_usd: float, job_id: str, *, gpu_sec: float | None = None,
                     scenes: int | None = None) -> float:
        """Записать фактическую стоимость задачи (повторный вызов с тем же job_id не удваивает).
        Возвращает потраченное за месяц."""
        key, cost = _check_key(job_id), _check_amount(actual_cost_usd, "actual_cost_usd")
        with self._locked() as st:
            st["reserved"].pop(key, None)
            if key in st["recorded_jobs"]:
                log.warning("трата по задаче %s уже записана — пропускаю", key)
                return float(st["spent_usd"])
            st["spent_usd"] = _round(float(st["spent_usd"]) + cost)
            st["recorded_jobs"].append(key)
            if gpu_sec is not None and scenes:
                st["calibration"]["gpu_sec"] = _round(float(st["calibration"]["gpu_sec"])
                                                      + _check_amount(gpu_sec, "gpu_sec"))
                st["calibration"]["scenes"] = int(st["calibration"]["scenes"]) + int(scenes)
            self._alerts(st)
            log.info("трата %s: $%.4f, за месяц $%.2f из $%.2f", key, cost, st["spent_usd"], self.limit)
            return float(st["spent_usd"])

    def reset(self, reason: str = "вручную") -> None:
        with self._locked() as st:
            st["resets"].append({"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "spent_before": st["spent_usd"], "reason": str(reason)[:200]})
            st["spent_usd"], st["alerts_sent"] = 0.0, []
        log.warning("бюджет месяца сброшен (%s)", reason)

    def status(self) -> dict:
        with self._locked() as st:
            committed = self._committed(st)
            return {"month": st["month"], "spent_usd": st["spent_usd"],
                    "reserved_usd": _round(committed - float(st["spent_usd"])),
                    "limit_usd": self.limit, "remaining_usd": _round(max(0.0, self.limit - committed)),
                    "percent": round(float(st["spent_usd"]) / self.limit * 100, 1),
                    "alerts_sent": list(st["alerts_sent"])}


def guard_for(cfg: Any) -> BudgetGuard | None:
    """BudgetGuard, если в конфиге есть включённый блок budget; иначе None (без ограничителя)."""
    bc = getattr(cfg, "budget", None)
    if bc is None or not getattr(bc, "enabled", False):
        return None
    return BudgetGuard(bc, cfg.budget_state_path)


# Функции из ТЗ — на конфиге проекта (config.json рядом с модулем)
_DEFAULT: BudgetGuard | None = None


def _default_guard() -> BudgetGuard:
    global _DEFAULT
    if _DEFAULT is None:
        from config_loader import load_config
        g = guard_for(load_config(Path(__file__).resolve().parent / "config.json", log_name="budget_guard"))
        if g is None:
            raise RuntimeError("в config.json нет включённого блока budget")
        _DEFAULT = g
    return _DEFAULT


def can_spend(estimated_cost_usd: float) -> bool:
    return _default_guard().can_spend(estimated_cost_usd)


def record_spend(actual_cost_usd: float, job_id: str) -> float:
    return _default_guard().record_spend(actual_cost_usd, job_id)


def gcp_budget_command(cfg_budget: Any, billing_account: str,
                       display_name: str = "prometheus monthly") -> list[str]:
    """Аргументы gcloud для бюджета с порогами из конфига (только формирует, не выполняет)."""
    if not BILLING_ACCOUNT_RE.fullmatch(billing_account or ""):
        raise ValueError("billing account должен быть вида 0X0X0X-0X0X0X-0X0X0X")
    if not re.fullmatch(r"[A-Za-z0-9 _\-]{1,60}", display_name):
        raise ValueError("display_name: буквы, цифры, пробел, _ и -")
    cmd = ["gcloud", "billing", "budgets", "create", f"--billing-account={billing_account}",
           f"--display-name={display_name}", f"--budget-amount={float(cfg_budget.monthly_limit_usd):g}USD"]
    cmd += [f"--threshold-rule=percent={t / 100:g}" for t in cfg_budget.alert_thresholds]
    return cmd


def main() -> None:
    import shlex

    from config_loader import cli_config

    def extra(ap):
        ap.add_argument("--status", action="store_true", help="показать состояние бюджета")
        ap.add_argument("--reset-budget", action="store_true", help="обнулить трату текущего месяца")
        ap.add_argument("--print-gcp-budget-cmd", action="store_true",
                        help="напечатать команду gcloud для бюджета GCP (не выполняет её)")

    cfg, a = cli_config("budget_guard", extra=extra)
    guard = guard_for(cfg)
    if guard is None:
        raise SystemExit("в config.json нет включённого блока budget")
    if a.print_gcp_budget_cmd:
        acc = os.getenv("GCP_BILLING_ACCOUNT", "").strip()
        if not acc:
            raise SystemExit("задай GCP_BILLING_ACCOUNT (gcloud billing accounts list)")
        sys.stdout.write(" ".join(shlex.quote(x) for x in gcp_budget_command(cfg.budget, acc)) + "\n")
        return
    if a.reset_budget:
        guard.reset("--reset-budget")
    sys.stdout.write(json.dumps(guard.status(), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
