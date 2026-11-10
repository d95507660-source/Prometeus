"""config_loader.py — загрузка config.json + niches.json, валидация, логирование.

Все относительные пути (data/..., models/..., logs/...) считаются от папки,
где лежит config.json, а не от текущей директории запуска.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_PATHS = {
    "models": "models",
    "videos": "data/videos",
    "post_plans": "data/post_plans",
    "forge_checkpoints": "data/forge_checkpoints",
    "trend_analysis": "data/trend_analysis",
    "trends_cache": "data/trends_cache",
    "stock_cache": "data/stock_cache",
    "stock_downloads": "data/stock_downloads",
    "hashtag_seeds": "data/hashtag_seeds.json",
    "analytics": "data/analytics",
    "affiliate": "data/affiliate",
    "cross_promo": "data/cross_promo",
    "logs": "logs",
    "accounts": "data/accounts",
    "devices": "devices.json",
    "niches": "niches.json",
}


@dataclass
class GcpConfig:
    project_id: str
    region: str = "us-central1"
    job_name: str = "prometheus-generate"
    work_bucket: str = "prometheus-work"
    models_bucket: str = "prometheus-models"
    mode: str = "cloud"                       # cloud — Cloud Run Job; local — local_worker на своём GPU


@dataclass
class LocalConfig:
    server_url: str = "http://127.0.0.1:8081"
    models_dir: str = "/models"
    vram_safety_mb: int = 2000                # свободной видеопамяти перед задачей не меньше этого
    max_concurrent_jobs: int = 1
    oom_max_frames: list[int] = field(default_factory=lambda: [81, 49])  # лесенка при нехватке VRAM
    queue_db: str = "data/local_queue.db"     # очередь local_worker (отдельно от очереди forge)
    output_dir: str = "data/local_outputs"
    poll_interval_sec: float = 5.0            # как часто forge спрашивает /status
    vram_wait_sec: float = 15.0               # пауза, если VRAM не хватает


@dataclass
class ForgeConfig:
    videos_per_device_per_day: int = 2
    video_duration_sec: list[int] = field(default_factory=lambda: [10, 15])
    scenes_per_video: list[int] = field(default_factory=lambda: [1, 2])
    scene_duration_sec: int = 5
    min_realism_score: int = 70
    hook_score_threshold: int = 60
    hook_max_retries: int = 2
    use_trend_analyzer: bool = True
    flux_enabled: bool = False
    flux_model: str = "black-forest-labs/FLUX.1-schnell"
    llm_model: str = "Qwen/Qwen2.5-72B-Instruct"
    language: str = "en"
    max_parallel_jobs: int = 4
    job_timeout_sec: int = 3600
    # очередь генерации (forge_stage.py / job_queue.py)
    queue_db: str = "data/queue.db"           # относительный путь — от папки config.json
    retry_max_attempts: int = 3               # всего попыток на задачу, включая первую
    retry_backoff_sec: float = 60.0           # пауза перед повтором: backoff * 2^(попытка-1)
    stop_on_failure_ratio: float = 0.2        # доля окончательно упавших задач → остановка, код 3
    new_account_days: int = 14                # аккаунты моложе — приоритет 10 (прогрев), иначе 5


@dataclass
class EconomicsConfig:
    cpa_commission: float = 0.0           # доля комиссии CPA-сети (0.1 = 10 %), если выплаты — до комиссии
    new_account_cost: float = 40.0        # во что обходится запуск одного аккаунта, $
    accounts_per_month_cap: int = 8       # не больше стольких новых аккаунтов в месяц
    gpu_cloud_threshold_usd: float = 300.0  # облачный GPU дороже этого в месяц — предупреждение
    forecast_months: int = 3
    forecast_history_months: int = 6      # сколько прошлых месяцев берётся для прогноза
    cpa_stats: str = "data/cpa_stats.json"
    software_sales: str = "data/software_sales.json"
    expenses: str = "data/expenses.json"


@dataclass
class BudgetConfig:
    enabled: bool = False                     # нет блока budget в config.json — ограничителя нет
    monthly_limit_usd: float = 300.0
    cost_per_gpu_hour: float = 0.672          # $/час инстанса Cloud Run с GPU [ПРОВЕРИТЬ по прайсу:
                                              # CPU и память тарифицируются отдельно]
    alert_thresholds: list[int] = field(default_factory=lambda: [50, 90, 100])
    current_spent_usd: float = 0.0            # стартовая трата месяца, если до включения уже тратили
    default_scene_gpu_sec: float = 120.0      # оценка до накопления статистики по реальным задачам
    billing_timezone: str = "America/Los_Angeles"  # месяц биллинга GCP [ПРОВЕРИТЬ ПО ДОКАМ]
    state_file: str = "data/budget.json"


@dataclass
class AccountHealthConfig:
    enabled: bool = False                     # нет блока в config.json — модуль выключен (как раньше)
    new_account_days: int = 7                 # столько дней новый аккаунт не публикует
    max_posts_per_day_progression: list[int] = field(default_factory=lambda: [1, 2, 3])
    shadowban_views_threshold: int = 50       # средние просмотры ниже → shadowban_suspect
    shadowban_check_hours: int = 48           # пост «созрел» для оценки через столько часов
    shadowban_max_post_age_hours: int = 168   # посты старше не учитываются
    shadowban_min_posts: int = 1              # минимум созревших постов для вывода
    ban_error_markers: list[str] = field(default_factory=lambda: [
        "suspended", "banned", "terminated", "disabled"])
    graph_api_version: str = "v21.0"          # Instagram Graph API [ПРОВЕРИТЬ ПО ДОКАМ]
    request_timeout_sec: float = 20.0


@dataclass
class AppConfig:
    project: str
    gcp: GcpConfig
    mimic: dict
    forge: ForgeConfig
    uniqueness: dict
    monetization: dict
    devices_registry: dict
    trend_analyzer: dict
    paths: dict
    trends_collector: dict
    niches: dict
    raw: dict
    root: Path
    config_path: Path
    timezone: str = "UTC"
    account_health: AccountHealthConfig = field(default_factory=AccountHealthConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)

    @property
    def gcp_project_id(self) -> str:
        return self.gcp.project_id

    def path(self, key: str) -> Path:
        """Абсолютный путь по ключу из paths (с учётом DEFAULT_PATHS)."""
        rel = self.paths.get(key) or DEFAULT_PATHS.get(key)
        if rel is None:
            raise KeyError(f"неизвестный ключ paths: {key}")
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def videos_root(self) -> Path:
        return self.path("videos")

    def today(self) -> str:
        return today_str(self.timezone)

    @property
    def queue_db_path(self) -> Path:
        p = Path(self.forge.queue_db)
        return p if p.is_absolute() else (self.root / p)

    def _rel(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    @property
    def local_queue_db_path(self) -> Path:
        return self._rel(self.local.queue_db)

    @property
    def local_output_dir(self) -> Path:
        return self._rel(self.local.output_dir)

    @property
    def budget_state_path(self) -> Path:
        p = Path(self.budget.state_file)
        return p if p.is_absolute() else (self.root / p)


def today_str(tz_name: str = "UTC") -> str:
    """Сегодняшняя дата YYYY-MM-DD в часовом поясе проекта."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Логирование: консоль + logs/<name>.log, повторный вызов ничего не дублирует
# ---------------------------------------------------------------------------

def setup_logging(name: str = "orchestrator", logs_dir: str | Path = "logs",
                  level: str | None = None) -> None:
    level = level or os.getenv("LOG_LEVEL") or "INFO"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not any(getattr(h, "_ags_console", False) for h in root.handlers):
        # убираем обработчики basicConfig, чтобы не было двойного вывода
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root.removeHandler(h)
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        ch._ags_console = True  # type: ignore[attr-defined]
        root.addHandler(ch)

    logs_dir = Path(logs_dir)
    target = (logs_dir / f"{name}.log").resolve()
    if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target
           for h in root.handlers):
        return
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(target, maxBytes=10 * 1024 * 1024, backupCount=3,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception as e:  # нет прав на запись и т.п. — работаем только в консоль
        sys.stderr.write(f"[config_loader] не удалось открыть лог-файл {target}: {e}\n")


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------

def _validate(cfg: AppConfig) -> None:
    errors: list[str] = []
    active = list(cfg.mimic.get("active_niches", []))
    unknown = [n for n in active if n not in cfg.niches]
    if unknown:
        errors.append(f"active_niches нет в niches.json: {sorted(unknown)}")

    markers = cfg.devices_registry.get("niche_markers", {})
    no_marker = [n for n in active
                 if not markers.get(n) and not cfg.niches.get(n, {}).get("marker")]
    if no_marker and cfg.devices_registry.get("require_username_with_niche_marker", True):
        errors.append(f"нет маркера для ниш: {no_marker}")

    f = cfg.forge
    if f.videos_per_device_per_day < 1:
        errors.append("forge.videos_per_device_per_day должен быть >= 1")
    if f.scene_duration_sec < 1:
        errors.append("forge.scene_duration_sec должен быть >= 1")
    if (len(f.scenes_per_video) != 2 or f.scenes_per_video[0] < 1
            or f.scenes_per_video[0] > f.scenes_per_video[1]):
        errors.append("forge.scenes_per_video должен быть [min, max], min >= 1")
    if not 0 <= f.min_realism_score <= 100 or not 0 <= f.hook_score_threshold <= 100:
        errors.append("пороги min_realism_score / hook_score_threshold должны быть 0..100")
    if f.hook_max_retries < 0:
        errors.append("forge.hook_max_retries должен быть >= 0")
    if not 1 <= f.max_parallel_jobs <= 64:
        errors.append("forge.max_parallel_jobs должен быть 1..64")
    if f.retry_max_attempts < 1:
        errors.append("forge.retry_max_attempts должен быть >= 1")
    if f.retry_backoff_sec < 0:
        errors.append("forge.retry_backoff_sec должен быть >= 0")
    if not 0 < f.stop_on_failure_ratio <= 1:
        errors.append("forge.stop_on_failure_ratio должен быть в (0, 1]")
    if f.new_account_days < 0:
        errors.append("forge.new_account_days должен быть >= 0")
    if not str(f.queue_db or "").strip() or "\x00" in str(f.queue_db):
        errors.append("forge.queue_db пустой или некорректный")

    ppd = int(cfg.mimic.get("posts_per_account_per_day", f.videos_per_device_per_day))
    if ppd != f.videos_per_device_per_day:
        log.warning("mimic.posts_per_account_per_day=%d не равен forge.videos_per_device_per_day=%d — "
                    "часть слотов останется без видео или часть видео без слотов",
                    ppd, f.videos_per_device_per_day)

    def _num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def _int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    if cfg.gcp.mode not in ("cloud", "local"):
        errors.append(f"gcp.mode должен быть cloud или local, получено {cfg.gcp.mode!r}")
    if cfg.gcp.mode == "cloud" and not cfg.gcp.project_id:
        errors.append("gcp.mode=cloud: нужен gcp.project_id (или GCP_PROJECT_ID)")
    e = cfg.economics
    if not _num(e.cpa_commission) or not 0 <= e.cpa_commission < 1:
        errors.append("economics.cpa_commission должен быть в [0, 1)")
    if not _num(e.new_account_cost) or e.new_account_cost <= 0:
        errors.append("economics.new_account_cost должен быть > 0")
    if not _int(e.accounts_per_month_cap) or e.accounts_per_month_cap < 0:
        errors.append("economics.accounts_per_month_cap должен быть целым >= 0")
    if not _num(e.gpu_cloud_threshold_usd) or e.gpu_cloud_threshold_usd < 0:
        errors.append("economics.gpu_cloud_threshold_usd должен быть >= 0")
    if not _int(e.forecast_months) or not 1 <= e.forecast_months <= 24:
        errors.append("economics.forecast_months должен быть 1..24")
    if not _int(e.forecast_history_months) or not 1 <= e.forecast_history_months <= 36:
        errors.append("economics.forecast_history_months должен быть 1..36")
    for key in ("cpa_stats", "software_sales", "expenses"):
        if not str(getattr(e, key) or "").strip():
            errors.append(f"economics.{key} пустой")

    lc = cfg.local
    if not re.fullmatch(r"https?://[A-Za-z0-9.\-]{1,253}(:\d{1,5})?/?", str(lc.server_url)):
        errors.append("local.server_url должен быть вида http://host:port")
    elif lc.server_url.startswith("http://") and not re.match(
            r"http://(127\.0\.0\.1|localhost|\[::1\])(:|/|$)", lc.server_url):
        log.warning("local.server_url без https на внешнем адресе: токен воркера уйдёт открытым текстом")
    if not str(lc.models_dir or "").strip():
        errors.append("local.models_dir пустой")
    if not _int(lc.vram_safety_mb) or lc.vram_safety_mb < 0:
        errors.append("local.vram_safety_mb должен быть целым >= 0")
    if not _int(lc.max_concurrent_jobs) or not 1 <= lc.max_concurrent_jobs <= 8:
        errors.append("local.max_concurrent_jobs должен быть 1..8")
    ladder = lc.oom_max_frames
    if (not isinstance(ladder, list) or not ladder
            or not all(_int(x) and x >= 5 and (x - 1) % 4 == 0 for x in ladder)
            or ladder != sorted(set(ladder), reverse=True)):
        errors.append("local.oom_max_frames: убывающий список чисел вида 4k+1 (>= 5), например [81, 49]")
    for key in ("poll_interval_sec", "vram_wait_sec"):
        if not _num(getattr(lc, key)) or getattr(lc, key) <= 0:
            errors.append(f"local.{key} должен быть > 0")
    for key in ("queue_db", "output_dir"):
        if not str(getattr(lc, key) or "").strip():
            errors.append(f"local.{key} пустой")

    b = cfg.budget

    if not _num(b.monthly_limit_usd) or b.monthly_limit_usd <= 0:
        errors.append("budget.monthly_limit_usd должен быть > 0")
    if not _num(b.cost_per_gpu_hour) or b.cost_per_gpu_hour <= 0:
        errors.append("budget.cost_per_gpu_hour должен быть > 0")
    if not _num(b.current_spent_usd) or b.current_spent_usd < 0:
        errors.append("budget.current_spent_usd должен быть >= 0")
    if not _num(b.default_scene_gpu_sec) or b.default_scene_gpu_sec <= 0:
        errors.append("budget.default_scene_gpu_sec должен быть > 0")
    th = b.alert_thresholds
    if (not isinstance(th, list) or not th
            or not all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 100 for x in th)
            or th != sorted(set(th))):
        errors.append("budget.alert_thresholds: возрастающий список целых 1..100 без повторов")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(str(b.billing_timezone))
    except Exception:
        errors.append(f"budget.billing_timezone: неизвестный часовой пояс {b.billing_timezone!r}")
    if not str(b.state_file or "").strip() or "\x00" in str(b.state_file):
        errors.append("budget.state_file пустой или некорректный")

    h = cfg.account_health
    prog = h.max_posts_per_day_progression
    if (not isinstance(prog, list) or not prog
            or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in prog)):
        errors.append("account_health.max_posts_per_day_progression: непустой список целых >= 1")
    if h.new_account_days < 0:
        errors.append("account_health.new_account_days должен быть >= 0")
    if h.shadowban_views_threshold < 0:
        errors.append("account_health.shadowban_views_threshold должен быть >= 0")
    if h.shadowban_check_hours < 1 or h.shadowban_max_post_age_hours < h.shadowban_check_hours:
        errors.append("account_health: нужно 1 <= shadowban_check_hours <= shadowban_max_post_age_hours")
    if h.shadowban_min_posts < 1:
        errors.append("account_health.shadowban_min_posts должен быть >= 1")
    if not re.fullmatch(r"v\d{1,3}\.\d{1,2}", str(h.graph_api_version)):
        errors.append("account_health.graph_api_version должен быть вида v21.0")
    if not h.ban_error_markers or not all(isinstance(m, str) and m.strip() for m in h.ban_error_markers):
        errors.append("account_health.ban_error_markers: непустой список строк")

    ppv = cfg.monetization.get("ppv", {})
    if ppv and not (0 < ppv.get("price_min", 1) <= ppv.get("price_max", 1)):
        errors.append("monetization.ppv: нужно 0 < price_min <= price_max")
    if ppv and not (0 < ppv.get("custom_min", 1) <= ppv.get("custom_max", 1)):
        errors.append("monetization.ppv: нужно 0 < custom_min <= custom_max")

    if errors:
        raise ValueError("ошибки в конфиге:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise ValueError(f"битый JSON в {path}: {e}") from e


def load_config(config_path: str | Path = "config.json",
                niches_path: str | Path | None = None,
                log_name: str | None = None) -> AppConfig:
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path)
    root = config_path.parent

    paths = {**DEFAULT_PATHS, **(raw.get("paths") or {})}
    niches_file = Path(niches_path) if niches_path else Path(paths["niches"])
    if not niches_file.is_absolute():
        niches_file = root / niches_file
    niches: dict = {}
    if niches_file.exists():
        niches = (_read_json(niches_file) or {}).get("niches", {})
    else:
        log.warning("niches.json не найден: %s", niches_file)

    logs_dir = Path(paths["logs"])
    if not logs_dir.is_absolute():
        logs_dir = root / logs_dir
    setup_logging(log_name or "orchestrator", logs_dir)

    gcp_raw = raw.get("gcp") or {}
    # project_id обязателен только для облака (проверка в _validate)
    econ_raw = raw.get("economics") or {}
    unknown_econ = set(econ_raw) - set(EconomicsConfig.__dataclass_fields__)
    if unknown_econ:
        log.warning("неизвестные ключи economics в config.json (игнорирую): %s", sorted(unknown_econ))
    local_raw = raw.get("local") or {}
    unknown_local = set(local_raw) - set(LocalConfig.__dataclass_fields__)
    if unknown_local:
        log.warning("неизвестные ключи local в config.json (игнорирую): %s", sorted(unknown_local))

    forge_raw = raw.get("forge") or {}
    unknown_forge = set(forge_raw) - set(ForgeConfig.__dataclass_fields__)
    if unknown_forge:
        log.warning("неизвестные ключи forge в config.json (игнорирую): %s", sorted(unknown_forge))

    budget_raw = dict(raw.get("budget") or {})
    unknown_budget = set(budget_raw) - set(BudgetConfig.__dataclass_fields__)
    if unknown_budget:
        log.warning("неизвестные ключи budget в config.json (игнорирую): %s", sorted(unknown_budget))
    if "budget" in raw:
        budget_raw.setdefault("enabled", True)  # блок есть — ограничитель включён

    health_raw = raw.get("account_health") or {}
    unknown_health = set(health_raw) - set(AccountHealthConfig.__dataclass_fields__)
    if unknown_health:
        log.warning("неизвестные ключи account_health в config.json (игнорирую): %s",
                    sorted(unknown_health))

    cfg = AppConfig(
        project=raw.get("project", "ai-girls-studio"),
        gcp=GcpConfig(
            project_id=str(gcp_raw.get("project_id") or os.getenv("GCP_PROJECT_ID") or ""),
            region=gcp_raw.get("region", "us-central1"),
            job_name=gcp_raw.get("job_name", "prometheus-generate"),
            work_bucket=gcp_raw.get("work_bucket", "prometheus-work"),
            models_bucket=gcp_raw.get("models_bucket", "prometheus-models"),
            mode=gcp_raw.get("mode", "cloud"),
        ),
        economics=EconomicsConfig(**{k: v for k, v in econ_raw.items()
                                     if k in EconomicsConfig.__dataclass_fields__}),
        local=LocalConfig(**{k: v for k, v in local_raw.items()
                             if k in LocalConfig.__dataclass_fields__}),
        mimic=raw.get("mimic", {}),
        forge=ForgeConfig(**{k: v for k, v in forge_raw.items()
                             if k in ForgeConfig.__dataclass_fields__}),
        uniqueness=raw.get("uniqueness", {}),
        monetization=raw.get("monetization", {}),
        devices_registry=raw.get("devices_registry", {}),
        trend_analyzer=raw.get("trend_analyzer", {}),
        paths=paths,
        trends_collector=raw.get("trends_collector", {}),
        niches=niches,
        raw=raw,
        root=root,
        config_path=config_path,
        timezone=raw.get("timezone", "UTC"),
        account_health=AccountHealthConfig(**{k: v for k, v in health_raw.items()
                                              if k in AccountHealthConfig.__dataclass_fields__}),
        budget=BudgetConfig(**{k: v for k, v in budget_raw.items()
                               if k in BudgetConfig.__dataclass_fields__}),
    )
    _validate(cfg)
    log.debug("config loaded: project=%s niches=%d root=%s", cfg.project, len(niches), root)
    return cfg


def apply_env_overrides(cfg: AppConfig) -> AppConfig:
    env_map = {
        "GCP_PROJECT_ID": "project_id",
        "GCP_REGION": "region",
        "GCP_JOB_NAME": "job_name",
        "GCS_WORK_BUCKET": "work_bucket",
        "GCS_MODELS_BUCKET": "models_bucket",
    }
    for env, attr in env_map.items():
        val = os.getenv(env)
        if val:
            setattr(cfg.gcp, attr, val)
    return cfg


def load_dotenv_if_present(root: str | Path | None = None) -> None:
    """Подхватывает .env, если установлен python-dotenv (не перезаписывает уже заданные переменные)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_file = Path(root or Path(__file__).resolve().parent) / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def add_common_args(ap) -> None:
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))


def cli_config(log_name: str, argv: list[str] | None = None, extra=None):
    """Общий разбор аргументов для всех CLI: --config + свои аргументы через extra(ap)."""
    import argparse
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    if extra:
        extra(ap)
    a = ap.parse_args(argv)
    load_dotenv_if_present(Path(a.config).resolve().parent)
    cfg = apply_env_overrides(load_config(a.config, log_name=log_name))
    return cfg, a


if __name__ == "__main__":
    c, _ = cli_config("config_loader")
    sys.stdout.write(json.dumps({"project": c.project, "root": str(c.root), "gcp": c.gcp.__dict__,
                                 "forge": c.forge.__dict__, "niches": list(c.niches.keys()),
                                 "timezone": c.timezone, "today": c.today()},
                                indent=2, ensure_ascii=False) + "\n")
