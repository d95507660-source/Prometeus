"""audience_analyzer.py — сбор ТОЛЬКО текста публичных комментариев под последними роликами профиля.

Что сохраняется: текст комментария и число лайков. Не сохраняются: имя автора, его ID,
аватар, ссылка на профиль. Дополнительно из текста вычищаются @упоминания, ссылки, e-mail и
телефоны — в комментариях люди часто оставляют личные данные.

Результат: data/audience_insights/<account_username>.json
  {"account": ..., "platform": "tiktok", "collected_at": ISO, "items": [
      {"video_index": 0, "comment_text": "...", "likes": 12}, ...]}

ВАЖНО: селекторы элементов интерфейса ниже — заготовка [ПРОВЕРИТЬ на вашей версии приложения].
Названия resource-id/описаний меняются от версии к версии и зависят от языка телефона.
Снимите дерево экрана:  python audience_analyzer.py --serial X --dump profile.xml
и поправьте SELECTORS или передайте свои через selectors=... .
Сбор автоматизацией интерфейса может противоречить правилам площадки — для своих аккаунтов
надёжнее официальные API (YouTube commentThreads, Instagram Graph /comments).
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diversity import atomic_write_text

log = logging.getLogger(__name__)

SERIAL_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")
MAX_VIDEOS_LIMIT = 9          # в сетке профиля без прокрутки видно ~9 роликов
MAX_COMMENTS_PER_VIDEO = 60
MAX_SCROLLS_PER_VIDEO = 8
MAX_COMMENT_LEN = 500

# [ПРОВЕРИТЬ] — заготовка для TikTok; подберите по дампу дерева экрана.
SELECTORS: dict[str, dict[str, Any]] = {
    "profile_ready": {"descriptionContains": "Followers"},
    "grid_item": {"className": "android.widget.FrameLayout", "descriptionContains": "video"},
    "comments_button": {"descriptionContains": "omment"},
    "comment_row": {"resourceIdMatches": r".*:id/comment_item.*"},
    "comment_text": {"resourceIdMatches": r".*:id/comment_content.*"},
    "comment_likes": {"resourceIdMatches": r".*:id/like_count.*"},
}

_MENTION = re.compile(r"@[\w.]{1,40}")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")


def anonymize(text: str) -> str:
    """Убирает из текста комментария личные данные; переносы строк → пробел."""
    t = _EMAIL.sub("[email]", text or "")
    t = _URL.sub("[link]", t)
    t = _MENTION.sub("@user", t)
    t = _PHONE.sub("[phone]", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:MAX_COMMENT_LEN]


def parse_count(text: str | None) -> int:
    """'12' → 12, '1.2K' → 1200, '3,4 тыс.' → 3400, '2M' → 2 000 000; непонятное → 0."""
    s = (text or "").strip().lower().replace(" ", "").replace(",", ".")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k|m|тыс\.?|млн\.?)?", s)
    if not m:
        return 0
    mult = {"k": 1_000, "тыс": 1_000, "тыс.": 1_000, "m": 1_000_000, "млн": 1_000_000,
            "млн.": 1_000_000}.get(m.group(2) or "", 1)
    return int(float(m.group(1)) * mult)


def _connect(serial: str):
    import uiautomator2 as u2
    return u2.connect(serial)


def _text(obj: Any) -> str | None:
    try:
        return obj.get_text() if obj.exists else None
    except Exception:
        return None


def _collect_comments(d: Any, video_index: int, sel: dict[str, dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for _ in range(MAX_SCROLLS_PER_VIDEO):
        rows = d(**sel["comment_row"])
        new = 0
        for i in range(rows.count):
            row = rows[i]
            raw = _text(row.child(**sel["comment_text"]))
            if not raw:
                continue
            text = anonymize(raw)
            if not text or text in seen:
                continue
            seen.add(text)
            new += 1
            out.append({"video_index": video_index, "comment_text": text,
                        "likes": parse_count(_text(row.child(**sel["comment_likes"])))})
            if len(out) >= MAX_COMMENTS_PER_VIDEO:
                return out
        if new == 0:
            break  # прокрутка больше ничего не добавила
        d.swipe_ext("up", scale=0.6)
        time.sleep(0.8)  # дать списку догрузиться
    return out


def save_insights(items: list[dict], account_username: str, platform: str = "tiktok",
                  data_dir: Path = Path("data")) -> Path:
    path = Path(data_dir) / "audience_insights" / f"{account_username}.json"
    atomic_write_text(path, json.dumps({
        "account": account_username, "platform": platform,
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": items}, ensure_ascii=False, indent=2))
    return path


def collect_public_text_data(device_serial: str, account_username: str, max_videos: int = 5,
                             *, selectors: dict[str, dict] | None = None,
                             data_dir: Path = Path("data"), device: Any = None) -> list[dict]:
    """Текст комментариев под последними max_videos роликами публичного профиля.
    device — подключённый uiautomator2-объект (для тестов); иначе подключаемся по serial."""
    if not isinstance(device_serial, str) or not SERIAL_RE.fullmatch(device_serial):
        raise ValueError(f"некорректный device_serial: {device_serial!r}")
    if not isinstance(account_username, str) or not USERNAME_RE.fullmatch(account_username):
        raise ValueError(f"некорректный account_username: {account_username!r}")
    if not 1 <= int(max_videos) <= MAX_VIDEOS_LIMIT:
        raise ValueError(f"max_videos должен быть 1..{MAX_VIDEOS_LIMIT}")
    sel = {**SELECTORS, **(selectors or {})}

    started = time.monotonic()
    try:
        d = device if device is not None else _connect(device_serial)
        d.open_url(f"https://www.tiktok.com/@{account_username}")  # username проверен регуляркой
        if not d(**sel["profile_ready"]).wait(timeout=20):
            raise RuntimeError("профиль не загрузился (селектор profile_ready не найден)")
    except Exception as e:
        log.error("устройство %s недоступно или профиль @%s не открыт: %s",
                  device_serial, account_username, e)
        return []

    items: list[dict] = []
    for vi in range(int(max_videos)):
        try:
            grid = d(**sel["grid_item"])
            if vi >= grid.count:
                log.info("@%s: в сетке только %d роликов", account_username, grid.count)
                break
            grid[vi].click()
            button = d(**sel["comments_button"])
            if not button.wait(timeout=10):
                log.warning("@%s ролик %d: кнопка комментариев не найдена", account_username, vi)
            else:
                button.click()
                if d(**sel["comment_row"]).wait(timeout=10):
                    items.extend(_collect_comments(d, vi, sel))
                d.press("back")  # закрыть комментарии
            d.press("back")  # вернуться в профиль
            d(**sel["profile_ready"]).wait(timeout=10)
        except Exception as e:  # частичный результат лучше, чем ничего
            log.error("@%s ролик %d: %s — останавливаю сбор", account_username, vi, e)
            break

    path = save_insights(items, account_username, data_dir=data_dir)
    log.info("@%s: собрано комментариев %d с %d роликов за %.1f с → %s", account_username,
             len(items), len({i['video_index'] for i in items}), time.monotonic() - started, path)
    return items


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="сбор текста публичных комментариев")
    ap.add_argument("--serial", required=True)
    ap.add_argument("--username")
    ap.add_argument("--max-videos", type=int, default=5)
    ap.add_argument("--dump", metavar="FILE.xml", help="сохранить дерево текущего экрана и выйти")
    a = ap.parse_args()
    if a.dump:
        if not SERIAL_RE.fullmatch(a.serial):
            raise SystemExit("некорректный --serial")
        atomic_write_text(Path(a.dump), _connect(a.serial).dump_hierarchy())
        log.info("дерево экрана → %s", a.dump)
        return
    if not a.username:
        raise SystemExit("нужен --username")
    items = collect_public_text_data(a.serial, a.username, a.max_videos)
    sys.stdout.write(json.dumps({"collected": len(items)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
