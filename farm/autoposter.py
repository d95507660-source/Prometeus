"""autoposter.py — публикация готовых роликов на телефоны через ADB + uiautomator2.

Как работает:
  1. ролик копируется на телефон (/sdcard/DCIM/AGS/...) и регистрируется в медиатеке;
  2. приложение открывается через стандартное «Поделиться» (intent SEND) сразу с этим видео;
  3. шаги интерфейса (кнопки «Далее», поле описания, «Опубликовать») ищутся по тексту,
     а не по координатам — набор шагов можно переопределить в config.json → autoposter.flows;
  4. успех засчитывается, только если кнопка публикации нажата и экран публикации закрылся.
Повторно один и тот же ролик на ту же платформу не публикуется (posted.json рядом с роликом).

Допуск к публикации (account_health, если включён в config.json):
  публикуем только аккаунты со статусом active без shadowban_suspect; число постов на площадку
  за день — не больше max_posts_per_day_progression для текущего дня после активации.

ВАЖНО: тексты кнопок зависят от версии приложения и языка телефона — проверь поток
на каждом приложении вручную (python autoposter.py --device X --dry-run показывает план).
"""
from __future__ import annotations

import json
import logging
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from diversity import atomic_write_text, validate_day, validate_device_id

log = logging.getLogger(__name__)

PACKAGES = {"tiktok": "com.zhiliaoapp.musically",
            "instagram": "com.instagram.android",
            "youtube": "com.google.android.youtube"}
CAPTION_LIMITS = {"tiktok": 2200, "instagram": 2200, "youtube": 100}
REMOTE_DIR = "/sdcard/DCIM/AGS"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-.]{1,120}$")

# Шаги по умолчанию. Тип шага:
#   click_any   — нажать первую найденную кнопку из списка текстов (optional: можно не найти)
#   set_caption — вставить описание в первое поле ввода
#   post_any    — финальная кнопка; после неё ждём, что она исчезла
DEFAULT_FLOWS: dict[str, list[dict[str, Any]]] = {
    "tiktok": [
        {"click_any": ["Next", "Далее"], "timeout": 25},
        {"click_any": ["Next", "Далее"], "timeout": 10, "optional": True},
        {"set_caption": True, "timeout": 15},
        {"post_any": ["Post", "Опубликовать"], "timeout": 15},
    ],
    "instagram": [
        {"click_any": ["Reels", "Reel"], "timeout": 15, "optional": True},
        {"click_any": ["Next", "Далее"], "timeout": 25},
        {"click_any": ["Next", "Далее"], "timeout": 10, "optional": True},
        {"set_caption": True, "timeout": 15},
        {"post_any": ["Share", "Поделиться", "Опубликовать"], "timeout": 15},
    ],
    "youtube": [
        {"click_any": ["Next", "Далее"], "timeout": 25, "optional": True},
        {"set_caption": True, "timeout": 20},
        {"post_any": ["Upload Short", "Upload", "Загрузить"], "timeout": 15},
    ],
}


class PostError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Текст подписи
# ---------------------------------------------------------------------------

def build_caption_text(description: str, hashtags: list[str], cfg: dict, platform: str,
                       cta: str = "") -> str:
    disc = cfg.get("monetization", {}).get("disclosure", {})
    disclosure = ""
    if disc.get("enabled") and platform in disc.get("platforms", []):
        disclosure = disc.get("text", "AI-generated character")
    tags = [t if t.startswith("#") else f"#{t}" for t in (hashtags or []) if t]
    tags = list(dict.fromkeys(tags))
    limit = CAPTION_LIMITS.get(platform, 2200)

    if platform == "youtube":  # у YouTube это заголовок: одна строка, до 100 символов
        tail = f" | {disclosure}" if disclosure else ""
        head = " ".join(x for x in [description.strip(), *tags[:2]] if x)
        head = head[: max(0, limit - len(tail))].rstrip()
        return (head + tail).strip()

    parts = [description.strip(), cta.strip(), " ".join(tags), disclosure]
    text = "\n\n".join(p for p in parts if p)
    if len(text) > limit:  # дисклеймер не обрезаем никогда
        keep = limit - len(disclosure) - 2
        text = text[:keep].rstrip() + ("\n\n" + disclosure if disclosure else "")
    return text


# ---------------------------------------------------------------------------
# ADB
# ---------------------------------------------------------------------------

def adb(serial: str, *args: str, timeout: int = 60, check: bool = True) -> str:
    cmd = ["adb", "-s", serial, *args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise PostError(f"adb {' '.join(args[:3])}: таймаут {timeout}s") from e
    if check and p.returncode != 0:
        raise PostError(f"adb {' '.join(args[:3])}: {(p.stderr or p.stdout)[-400:]}")
    return p.stdout


def adb_shell(serial: str, command: str, timeout: int = 60) -> str:
    """Одна строка shell-команды; все подставляемые значения должны быть заранее экранированы."""
    return adb(serial, "shell", command, timeout=timeout)


def connect(host: str, port: int) -> str:
    serial = f"{host}:{port}"
    subprocess.run(["adb", "connect", serial], capture_output=True, text=True, timeout=20)
    state = subprocess.run(["adb", "-s", serial, "get-state"], capture_output=True,
                           text=True, timeout=15).stdout.strip()
    if state != "device":
        raise PostError(f"{serial} недоступен (state={state or 'нет ответа'})")
    return serial


def human_delay(min_s: float = 1.5, max_s: float = 4.5) -> None:
    time.sleep(random.uniform(min_s, max_s))


def push_video(serial: str, video: Path, remote_name: str) -> str:
    if not SAFE_NAME.match(remote_name):
        raise PostError(f"небезопасное имя файла: {remote_name!r}")
    remote = f"{REMOTE_DIR}/{remote_name}"
    adb_shell(serial, f"mkdir -p {REMOTE_DIR}")
    adb(serial, "push", str(video), remote, timeout=300)
    adb_shell(serial, "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
                      f"-d {shlex.quote('file://' + remote)}")
    return remote


def media_uri(serial: str, remote_name: str, wait_sec: int = 20) -> str:
    """content:// URI ролика в медиатеке — через него приложение получает доступ к файлу."""
    where = shlex.quote(f"_display_name='{remote_name}'")
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        out = adb_shell(serial, "content query --uri content://media/external/video/media "
                                f"--projection _id --where {where}")
        ids = re.findall(r"_id=(\d+)", out)
        if ids:
            return f"content://media/external/video/media/{ids[-1]}"
        time.sleep(2)
    raise PostError(f"ролик {remote_name} не появился в медиатеке")


def open_share(serial: str, package: str, uri: str) -> None:
    adb_shell(serial, "am start -a android.intent.action.SEND -t video/mp4 "
                      f"--eu android.intent.extra.STREAM {shlex.quote(uri)} "
                      f"--grant-read-uri-permission -p {shlex.quote(package)}")


def cleanup_remote(serial: str, days: int = 2) -> None:
    try:
        adb_shell(serial, f"find {REMOTE_DIR} -name '*.mp4' -mtime +{int(days)} -delete", timeout=30)
    except Exception as e:
        log.debug("очистка %s: %s", serial, e)


# ---------------------------------------------------------------------------
# UI-шаги (uiautomator2)
# ---------------------------------------------------------------------------

def _u2(serial: str):
    import uiautomator2 as u2
    return u2.connect(serial)


def _find_any(d, texts: list[str], timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for t in texts:
            for sel in (d(text=t), d(description=t)):
                if sel.exists:
                    return sel
        time.sleep(0.7)
    return None


def run_flow(d, steps: list[dict], caption: str) -> None:
    for i, step in enumerate(steps, 1):
        timeout = float(step.get("timeout", 15))
        if "click_any" in step:
            sel = _find_any(d, step["click_any"], timeout)
            if sel is None:
                if step.get("optional"):
                    continue
                raise PostError(f"шаг {i}: не найдена кнопка {step['click_any']}")
            sel.click()
            human_delay(1.5, 3.5)
        elif step.get("set_caption"):
            field = d(className="android.widget.EditText")
            if not field.wait(timeout=timeout):
                raise PostError(f"шаг {i}: не найдено поле описания")
            field.click()
            human_delay(0.8, 1.6)
            field.set_text(caption)  # юникод и переносы строк — без shell, без инъекций
            human_delay(1.0, 2.0)
            if step.get("hide_keyboard"):  # включай, если клавиатура закрывает кнопку публикации
                d.press("back")
                human_delay(0.8, 1.5)
        elif "post_any" in step:
            sel = _find_any(d, step["post_any"], timeout)
            if sel is None:
                raise PostError(f"шаг {i}: не найдена кнопка публикации {step['post_any']}")
            sel.click()
            # подтверждение: кнопка публикации должна исчезнуть
            gone_deadline = time.time() + 30
            while time.time() < gone_deadline:
                if _find_any(d, step["post_any"], 1.0) is None:
                    return
                time.sleep(1.5)
            raise PostError("кнопка публикации не исчезла — публикация не подтверждена")
        else:
            raise PostError(f"шаг {i}: неизвестный тип {step}")
    raise PostError("в потоке нет шага post_any")


def post_video(serial: str, platform: str, video: Path, caption: str, remote_name: str,
               flows: dict[str, list[dict]], screenshots_dir: Path) -> bool:
    package = PACKAGES[platform]
    d = _u2(serial)
    try:
        push_video(serial, video, remote_name)
        uri = media_uri(serial, remote_name)
        d.app_stop(package)
        human_delay(1, 2)
        open_share(serial, package, uri)
        human_delay(4, 7)
        run_flow(d, flows.get(platform) or DEFAULT_FLOWS[platform], caption)
        human_delay(3, 6)
        return True
    except Exception as e:
        log.error("%s %s: %s", serial, platform, e)
        try:  # снимок экрана для разбора, но успехом это НЕ считается
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            d.screenshot(str(screenshots_dir / f"{remote_name}.png"))
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Публикация по устройству
# ---------------------------------------------------------------------------

def _posted_file(source_dir: Path) -> Path:
    return source_dir / "posted.json"


def _load_posted(source_dir: Path) -> dict:
    f = _posted_file(source_dir)
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except Exception:
        return {}


def _save_posted(source_dir: Path, data: dict) -> None:
    atomic_write_text(_posted_file(source_dir), json.dumps(data, ensure_ascii=False, indent=2))


def _posted_today(day_dir: Path) -> dict[str, int]:
    """Сколько успешных публикаций на каждую площадку уже было за этот день."""
    counts: dict[str, int] = {}
    for source_dir in day_dir.glob("source_*"):
        if not source_dir.is_dir():
            continue
        for platform, rec in _load_posted(source_dir).items():
            if isinstance(rec, dict) and rec.get("ok"):
                counts[platform] = counts.get(platform, 0) + 1
    return counts


def post_device(device: dict, cfg_app, day: str | None = None, dry_run: bool = False,
                health: Any = None) -> dict:
    """cfg_app — AppConfig из config_loader. health — AccountHealth (по умолчанию из конфига)."""
    from account_health import health_for
    from forge import read_checkpoint
    from mimic import get_slot

    cfg = cfg_app.raw
    device_id = validate_device_id(device["device_id"])
    day = validate_day(day or cfg_app.today())
    if health is None:
        health = health_for(cfg_app)
    day_dir = cfg_app.videos_root / device_id / day
    if not day_dir.exists():
        return {"device": device_id, "posted": 0, "error": f"нет роликов за {day}"}

    # допуск аккаунтов — до подключения к телефону: незачем будить устройство впустую
    blocked: dict[str, str] = {}
    allowed_today: dict[str, int] = {}
    if health is not None:
        for platform, acc in (device.get("accounts") or {}).items():
            if not acc.get("enabled", True) or platform not in PACKAGES:
                continue
            ok, reason = health.can_post(device_id, platform, day, device=device)
            if ok:
                allowed_today[platform] = health.posts_allowed(device_id, platform, day, device=device)
            else:
                blocked[platform] = reason
                log.info("публикация %s/%s пропущена: %s", device_id, platform, reason)
        if not allowed_today:
            return {"device": device_id, "day": day, "posted": 0, "failed": 0, "blocked": blocked}
    posted_today = _posted_today(day_dir)

    ap_cfg = cfg.get("autoposter", {})
    flows = {**DEFAULT_FLOWS, **(ap_cfg.get("flows") or {})}
    # сколько роликов за один запуск на площадку; меньше posts_per_day — если запускаешь
    # автопостинг несколько раз в день по расписанию, чтобы посты шли с интервалом
    max_per_platform = int(ap_cfg.get("max_posts_per_platform_per_run",
                                      cfg.get("mimic", {}).get("posts_per_account_per_day", 2)))
    screenshots = cfg_app.path("logs") / "autoposter_screens"

    serial = None
    if not dry_run:
        serial = connect(device["adb"]["host"], int(device["adb"]["port"]))
        cleanup_remote(serial)

    posted, failed, planned = 0, 0, []
    per_platform: dict[str, int] = {}
    source_dirs = sorted((p for p in day_dir.glob("source_*") if p.is_dir()),
                         key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    for source_dir in source_dirs:
        try:
            src_idx = int(source_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            log.warning("пропускаю папку с неожиданным именем: %s", source_dir)
            continue
        slot = get_slot(cfg_app, device_id, day, src_idx)
        ckpt = read_checkpoint(cfg_app, device_id, day, src_idx)
        description = ckpt.get("hook") or slot.get("hook") or ""
        hashtags = list(dict.fromkeys((slot.get("hashtags") or []) + (ckpt.get("hashtags") or [])))[:5]
        cta = slot.get("cta", "")
        done = _load_posted(source_dir)

        for platform, acc in (device.get("accounts") or {}).items():
            if not acc.get("enabled", True) or platform not in PACKAGES:
                continue
            if platform in blocked:
                continue
            if done.get(platform, {}).get("ok"):
                continue
            if per_platform.get(platform, 0) >= max_per_platform:
                continue
            if health is not None and posted_today.get(platform, 0) >= allowed_today.get(platform, 0):
                continue  # дневной лимит прогрессии исчерпан
            video = source_dir / f"{platform}.mp4"
            if not video.exists():
                continue
            caption = build_caption_text(description, hashtags, cfg, platform, cta)
            if dry_run:
                planned.append({"source": source_dir.name, "platform": platform,
                                "account": acc.get("username"), "caption": caption})
                per_platform[platform] = per_platform.get(platform, 0) + 1
                posted_today[platform] = posted_today.get(platform, 0) + 1
                continue
            remote = f"ags_{device_id}_{day}_{src_idx}_{platform}.mp4"
            if serial is None:  # не dry-run — подключение выполнено выше
                raise PostError("нет подключения к устройству")
            ok = post_video(serial, platform, video, caption, remote, flows, screenshots)
            log.info("публикация %s/%s/%s → %s: %s", device_id, day, source_dir.name, platform,
                     "OK" if ok else "ОШИБКА")
            done[platform] = {"ok": ok, "ts": time.time(), "account": acc.get("username")}
            _save_posted(source_dir, done)
            per_platform[platform] = per_platform.get(platform, 0) + 1
            if ok:
                posted += 1
                posted_today[platform] = posted_today.get(platform, 0) + 1
                if health is not None:
                    health.record_post(device_id, platform)
            else:
                failed += 1
            human_delay(8, 20)
        if not dry_run and source_dir is not source_dirs[-1] and (posted or failed):
            human_delay(30, 90)  # пауза между разными роликами
    res = {"device": device_id, "day": day, "posted": posted, "failed": failed}
    if blocked:
        res["blocked"] = blocked
    if dry_run:
        res["planned"] = planned
    return res


if __name__ == "__main__":
    from config_loader import cli_config
    from devices_registry import DevicesRegistry

    def extra(ap):
        ap.add_argument("--device", action="append", default=None)
        ap.add_argument("--day", default=None)
        ap.add_argument("--dry-run", action="store_true")

    cfg_app, a = cli_config("autoposter", extra=extra)
    for dev in DevicesRegistry.from_config(cfg_app).get_active_devices(device_ids=a.device):
        try:
            out = post_device(dev, cfg_app, a.day, a.dry_run)
        except Exception as e:
            log.exception("autoposter %s упал", dev.get("device_id"))
            out = {"device": dev["device_id"], "error": str(e)}
        sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
