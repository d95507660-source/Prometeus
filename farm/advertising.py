"""advertising.py — CPA-офферы в планах постов и обязательная маркировка рекламы.

Используется mimic.py (подбор оффера при планировании) и autoposter.py (маркировка при публикации).

Рекламный слот плана mimic:
  slot["ad"] = {"offer_id", "name", "advertiser", "erid", "label", "links": {platform: url}}
  slot["description"] = {platform: "<метка>\\n<ссылка>"}
Метка: «Реклама. <рекламодатель>. erid: <токен>» (erid — если есть в оффере).

Оффер в data/cpa_offers.json может содержать для маркировки:
  "advertiser": "ООО «Пример», ИНН 0000000000"  — кто рекламодатель (иначе brand → name → «Оффер <id>»)
  "brand": "FitApp", "name": "...", "erid": "2Vtz..."  — токен ОРД
ВАЖНО [ПРОВЕРИТЬ у юриста/ОРД]: erid выдаётся на конкретный креатив, а не на оффер целиком; здесь
один erid на оффер — упрощение. Состав метки (наименование/ИНН рекламодателя) и порядок передачи
статистики в ОРД — по 38-ФЗ «О рекламе» и требованиям вашего ОРД.

Маркеры площадок (advertising.default_ad_markers): начинающиеся с «#» — в подпись,
остальные — тексты переключателей в приложении (Paid partnership и т.п.), их нажимает autoposter.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from cpa_matcher import best_offer_from, load_offers
from link_rotator import account_id_for, check_erid, get_tracking_link

log = logging.getLogger(__name__)

MAX_ADVERTISER_LEN = 120
CPA_LINK_RE = re.compile(r"https?://\S*utm_content=offer_[A-Za-z0-9_]+\S*")


class AdError(ValueError):
    """Рекламный пост нельзя публиковать (нет erid при require_erid, кривые данные оффера)."""


def _clean(text: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def advertiser_of(offer: dict) -> str:
    return (_clean(offer.get("advertiser"), MAX_ADVERTISER_LEN)
            or _clean(offer.get("brand"), MAX_ADVERTISER_LEN)
            or _clean(offer.get("name"), MAX_ADVERTISER_LEN)
            or f"Оффер {offer.get('id')}")


def ad_label(prefix: str, advertiser: str, erid: str | None) -> str:
    label = f"{prefix.strip()}. {advertiser}."
    return f"{label} erid: {erid}" if erid else label


def hashtag_markers(cfg_ad: Any, platform: str) -> list[str]:
    return [m.strip() for m in cfg_ad.default_ad_markers.get(platform, []) if m.strip().startswith("#")]


def toggle_markers(cfg_ad: Any, platform: str) -> list[str]:
    return [m.strip() for m in cfg_ad.default_ad_markers.get(platform, []) if not m.strip().startswith("#")]


class AdPlanner:
    """Подбор оффера под слот плана. Офферы читаются один раз на запуск mimic."""

    def __init__(self, cfg_ad: Any, offers: list[dict]):
        self.cfg = cfg_ad
        self.offers = offers

    @classmethod
    def from_config(cls, cfg: Any) -> AdPlanner | None:
        ad = getattr(cfg, "advertising", None)
        if ad is None or not ad.enabled:
            return None
        offers = load_offers(cfg.ad_offers_path)
        if not offers:
            log.warning("advertising включён, но офферов нет (%s) — посты без CPA", cfg.ad_offers_path)
        return cls(ad, offers)

    def eligible(self, device: dict) -> bool:
        t = device.get("account_type")
        if t is None:
            return False  # без явного типа CPA не ставим: реклама на «профессиональном» аккаунте хуже
        return t in self.cfg.account_types

    def ad_for(self, device: dict, platforms: list[str], hook: str, hashtags: list[str]) -> dict | None:
        if not self.offers or not platforms or not self.eligible(device):
            return None
        offer = best_offer_from({"topic": hook, "hook": hook, "hashtags": hashtags}, self.offers)
        if offer is None:
            return None
        erid = offer.get("erid")
        if erid is not None:
            try:
                check_erid(erid)
            except ValueError as e:
                log.error("оффер %s: %s — оффер пропущен", offer["id"], e)
                return None
        elif self.cfg.require_erid:
            # план всё равно получает оффер: autoposter не опубликует такой пост и покажет причину
            log.warning("оффер %s без erid при require_erid=true — пост не будет опубликован", offer["id"])
        advertiser = advertiser_of(offer)
        label = ad_label(self.cfg.label_prefix, advertiser, erid)
        links = {p: get_tracking_link(offer["url"], account_id_for(device["device_id"], p), offer["id"], p,
                                      erid=erid) for p in platforms}
        return {"offer_id": offer["id"], "name": _clean(offer.get("name") or offer.get("brand"), 120),
                "advertiser": advertiser, "erid": erid, "label": label, "links": links,
                "matched_keywords": offer.get("matched_keywords", [])}


def detect_ad(slot: dict, platform: str) -> dict | None:
    """Рекламные данные слота. Если ad нет, но в description есть CPA-ссылка (utm_content=offer_…) —
    пост всё равно рекламный: возвращаем блок без рекламодателя и erid."""
    ad = slot.get("ad")
    if isinstance(ad, dict) and ad.get("links"):
        return ad
    desc = slot.get("description")
    text = desc.get(platform, "") if isinstance(desc, dict) else str(desc or "")
    m = CPA_LINK_RE.search(text)
    if m:
        return {"offer_id": None, "advertiser": None, "erid": None, "label": None,
                "links": {platform: m.group(0)}}
    return None


def validate_for_publish(ad: dict, cfg_ad: Any, platform: str) -> tuple[str, str | None]:
    """(метка, ссылка) для публикации или AdError, если публиковать нельзя."""
    erid = ad.get("erid")
    if cfg_ad.require_erid and not erid:
        raise AdError("рекламный пост без erid (advertising.require_erid=true) — не публикуется")
    if erid:
        check_erid(erid)
    label = ad.get("label") or ad_label(cfg_ad.label_prefix, ad.get("advertiser") or "рекламодатель не указан",
                                        erid)
    if not label.startswith(cfg_ad.label_prefix):
        raise AdError(f"метка рекламы должна начинаться с «{cfg_ad.label_prefix}»: {label!r}")
    return label, (ad.get("links") or {}).get(platform)
