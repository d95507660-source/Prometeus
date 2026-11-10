"""cpa_matcher.py — подбор партнёрского оффера под идею ролика.

Файл офферов (data/cpa_offers.json) — список или {"offers": [...]}:
  [{"id": 1, "keywords": ["tracker", "habit"], "payout": 2.5, "url": "https://...",
    "enabled": true}]
  enabled необязателен (по умолчанию true).

content_idea — ответ LLM; для сопоставления берутся поля
  topic, title, hook, key_points (список или строка), keywords, hashtags.

Совпадение — по целому слову без учёта регистра; «tracker» совпадает и с «trackers»
(окончания s/es). Ключ из нескольких слов («habit tracker») ищется как фраза.
Лучший оффер — максимальный payout; при равенстве — больше совпавших слов, затем меньший id.
Ответ — копия оффера с полем matched_keywords.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_OFFERS_FILE = "data/cpa_offers.json"
IDEA_TEXT_FIELDS = ("topic", "title", "hook", "key_points", "keywords", "hashtags")
OFFER_ID_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
MAX_KEYWORD_LEN = 80


def _idea_text(content_idea: dict) -> str:
    parts: list[str] = []
    for key in IDEA_TEXT_FIELDS:
        v = content_idea.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (list, tuple)):
            parts.extend(str(x) for x in v if isinstance(x, (str, int, float)))
    # «#habittracker» → «habittracker», «habit_tracker» → «habit tracker»
    text = " ".join(parts).lower().replace("#", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _keyword_pattern(keyword: str) -> re.Pattern[str] | None:
    kw = re.sub(r"\s+", " ", keyword.strip().lower())
    if not kw or len(kw) > MAX_KEYWORD_LEN:
        return None
    # \w в Python 3 — юникод, поэтому кириллица работает так же
    return re.compile(r"(?<!\w)" + re.escape(kw).replace(r"\ ", r"\s+") + r"(?:s|es)?(?!\w)")


def _valid_offer(o: Any) -> dict | None:
    """Проверенная копия оффера или None (причина — в лог)."""
    if not isinstance(o, dict):
        log.warning("оффер пропущен: не объект (%r)", o)
        return None
    oid = o.get("id")
    if isinstance(oid, bool) or not (
            (isinstance(oid, int) and oid >= 0) or (isinstance(oid, str) and OFFER_ID_RE.fullmatch(oid))):
        log.warning("оффер пропущен: некорректный id %r", oid)
        return None
    if o.get("enabled", True) is False:
        return None
    payout = o.get("payout")
    if isinstance(payout, bool) or not isinstance(payout, (int, float)) \
            or not math.isfinite(payout) or payout < 0:
        log.warning("оффер %s пропущен: некорректный payout %r", oid, payout)
        return None
    url = o.get("url")
    if not isinstance(url, str) or not re.match(r"^https?://[^\s]+$", url):
        log.warning("оффер %s пропущен: url должен начинаться с http(s)://", oid)
        return None
    kws = o.get("keywords")
    if not isinstance(kws, list) or not any(isinstance(k, str) and k.strip() for k in kws):
        log.warning("оффер %s пропущен: нет ключевых слов", oid)
        return None
    return {**o, "keywords": [k for k in kws if isinstance(k, str) and k.strip()]}


def load_offers(offers_file: str | Path = DEFAULT_OFFERS_FILE) -> list[dict]:
    """Проверенные офферы. Нет файла / битый JSON / не тот формат → [] и запись в лог."""
    path = Path(offers_file)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.error("файл офферов не найден: %s", path)
        return []
    except json.JSONDecodeError as e:
        log.error("битый JSON в %s: %s", path, e)
        return []
    except OSError as e:
        log.error("файл офферов не читается %s: %s", path, e)
        return []
    if isinstance(data, dict):
        data = data.get("offers")
    if not isinstance(data, list):
        log.error("%s: ожидался список офферов или {\"offers\": [...]}", path)
        return []
    offers = [v for v in (_valid_offer(o) for o in data) if v is not None]
    ids = [o["id"] for o in offers]
    dupes = sorted({str(i) for i in ids if ids.count(i) > 1})
    if dupes:
        log.warning("%s: повторяющиеся id офферов %s", path, dupes)
    return offers


def match_keywords(content_idea: dict, keywords: list[str]) -> list[str]:
    text = _idea_text(content_idea)
    if not text:
        return []
    out = []
    for kw in keywords:
        pat = _keyword_pattern(kw)
        if pat is not None and pat.search(text):
            out.append(kw)
    return out


def find_offer(content_idea: dict, offers_file: str | Path = DEFAULT_OFFERS_FILE) -> dict | None:
    """Лучший подходящий оффер или None."""
    if not isinstance(content_idea, dict):
        log.error("find_offer: content_idea должен быть dict, получено %s", type(content_idea).__name__)
        return None
    candidates = []
    for offer in load_offers(offers_file):
        matched = match_keywords(content_idea, offer["keywords"])
        if matched:
            candidates.append({**offer, "matched_keywords": matched})
    # максимальный payout → больше совпадений → меньший id (числовые id раньше строковых)
    best_offer = min(candidates, key=lambda o: (
        -float(o["payout"]), -len(o["matched_keywords"]),
        (0, o["id"], "") if isinstance(o["id"], int) else (1, 0, o["id"])), default=None)
    if best_offer is None:
        log.info("оффер не найден для идеи %r", _idea_text(content_idea)[:120])
    else:
        log.info("оффер %s (payout %.2f) по словам %s", best_offer["id"], best_offer["payout"],
                 best_offer["matched_keywords"])
    return best_offer
