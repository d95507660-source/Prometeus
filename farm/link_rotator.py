"""link_rotator.py — уникальная ссылка с UTM-метками для пары (аккаунт, оффер).

get_tracking_link("https://partner.example/offer?ref=42", "aria_fight", 7, "tiktok") →
  https://partner.example/offer?ref=42&utm_source=tiktok&utm_medium=video
      &utm_campaign=account_aria_fight&utm_content=offer_7

Существующие параметры ссылки и якорь (#...) сохраняются; utm_* из исходной ссылки
перезаписываются нашими. Ссылки с логином в адресе (https://user:pass@host) и с пробелами /
управляющими символами отклоняются.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
PLATFORM_RE = re.compile(r"^[a-z0-9_]{1,20}$")
UTM_KEYS = ("utm_source", "utm_medium", "utm_campaign", "utm_content")
MAX_URL_LEN = 2048


def _check_id(value: object, what: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{what}: ожидалась строка или число, получено bool")
    s = str(value) if isinstance(value, int) else value
    if not isinstance(s, str) or not ID_RE.fullmatch(s):
        raise ValueError(f"{what} может содержать только буквы, цифры и _ (1–64 символа): {value!r}")
    return s


def _check_base_url(base_url: object) -> str:
    if not isinstance(base_url, str) or not base_url.lower().startswith(("http://", "https://")):
        raise ValueError(f"base_url должен начинаться с http:// или https://: {base_url!r}")
    if len(base_url) > MAX_URL_LEN or re.search(r"[\s\x00-\x1f\x7f]", base_url):
        raise ValueError("base_url содержит пробелы/управляющие символы или слишком длинный")
    parts = urlsplit(base_url)
    if not parts.hostname:
        raise ValueError(f"в base_url нет домена: {base_url!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("base_url с логином/паролем в адресе не допускается")
    return base_url


def get_tracking_link(base_url: str, account_id: str, offer_id: int, platform: str) -> str:
    url = _check_base_url(base_url)
    acc = _check_id(account_id, "account_id")
    oid = _check_id(offer_id, "offer_id")
    if not isinstance(platform, str) or not PLATFORM_RE.fullmatch(platform):
        raise ValueError(f"platform: только a-z, 0-9 и _ (1–20 символов): {platform!r}")
    if url.lower().startswith("http://"):
        log.warning("ссылка оффера %s без https: %s", oid, urlsplit(url).hostname)

    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in UTM_KEYS]
    query += [("utm_source", platform), ("utm_medium", "video"),
              ("utm_campaign", f"account_{acc}"), ("utm_content", f"offer_{oid}")]
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", urlencode(query),
                       parts.fragment))
