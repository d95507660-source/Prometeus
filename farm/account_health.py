"""account_health.py — состояние аккаунтов, допуск к публикации, проверка охвата и бана.

Только чтение через официальные API. Никаких действий в интерфейсе приложений.

Состояние: data/accounts/<device_id>_<platform>.json
  status            new | active | restricted | banned
  created_at        YYYY-MM-DD — из devices.json (accounts.<p>.created_at или created_at устройства),
                    иначе дата первого появления; если API показывает посты старше — берём дату
                    самого раннего поста (уже работающие аккаунты не блокируются на N дней)
  first_post_at     ISO-время первого поста (из API или от autoposter)
  shadowban_suspect true — средние просмотры созревших постов < порога, публикация на паузе

Правила:
  new → active        через new_account_days дней после created_at
  active ↔ restricted при shadowban_suspect (снимается, если повторная проверка ≥ порога)
  banned              API вернул «suspended/banned/…» (ban_error_markers) или ранее найденный
                      канал YouTube исчез. Статус липкий: снять — вручную (--set DEV:PLATFORM:active)
  Ошибки авторизации, квоты, сети статус НЕ меняют — пишутся в last_error.
  Лимит постов в день: max_posts_per_day_progression[k], k — номер дня после активации.

Охват «за 48 часов»: API отдаёт просмотры за всё время, поэтому оцениваются посты возрастом
от shadowban_check_hours до shadowban_max_post_age_hours. Их просмотры ≥ просмотров за первые 48 ч,
так что «< 50» — консервативный сигнал (ложных срабатываний меньше, чем пропусков).

Учётные данные (только из окружения, в лог не пишутся):
  TIKTOK_TOKEN_<DEVICE_ID>     — user access token TikTok (scope video.list)  [ПРОВЕРИТЬ ПО ДОКАМ]
  INSTAGRAM_TOKEN_<DEVICE_ID>  — токен Instagram Graph API (instagram_basic, instagram_manage_insights)
                                 [ПРОВЕРИТЬ ПО ДОКАМ]
  YOUTUBE_API_KEY              — ключ YouTube Data API v3 (публичная статистика)
  <DEVICE_ID> — device_id в верхнем регистре, «-» → «_».
  В devices.json: accounts.instagram.ig_user_id, accounts.youtube.channel_id.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from diversity import atomic_write_text, validate_device_id

log = logging.getLogger("account_health")

STATUSES = ("new", "active", "restricted", "banned")
PLATFORMS = ("tiktok", "instagram", "youtube")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
YT_CHANNEL_RE = re.compile(r"^UC[A-Za-z0-9_\-]{22}$")
IG_USER_RE = re.compile(r"^\d{5,25}$")
MEDIA_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
UNLIMITED = 10 ** 6

# [ПРОВЕРИТЬ ПО ДОКАМ] — адреса и поля сверены по памяти, не по живому API
TIKTOK_VIDEO_LIST_URL = "https://open.tiktokapis.com/v2/video/list/"
GRAPH_BASE = "https://graph.facebook.com"
YT_BASE = "https://www.googleapis.com/youtube/v3"

HttpFn = Callable[..., tuple[int, Any]]


def requests_http(method: str, url: str, *, headers: dict | None = None, params: dict | None = None,
                  json_body: dict | None = None, timeout: float = 20.0) -> tuple[int, Any]:
    """(HTTP-код, JSON-ответ или {"_text": ...}). Исключения сети пробрасываются."""
    import requests
    r = requests.request(method, url, headers=headers or {}, params=params, json=json_body,
                         timeout=timeout)
    try:
        body = r.json()
    except ValueError:
        body = {"_text": r.text[:500]}
    return r.status_code, body


# ---------------------------------------------------------------------------
# Результаты API
# ---------------------------------------------------------------------------

@dataclass
class PostStat:
    post_id: str
    published_at: datetime
    views: int | None


@dataclass
class FetchResult:
    posts: list[PostStat] = field(default_factory=list)
    error_kind: str | None = None  # banned | not_found | auth | not_configured | error
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_kind is None


def _err_message(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        parts = [str(err.get(k)) for k in ("message", "code", "type", "error_subcode")
                 if err.get(k) not in (None, "", "ok")]
        for e in err.get("errors") or []:  # YouTube: errors[].reason
            if isinstance(e, dict) and e.get("reason"):
                parts.append(str(e["reason"]))
        return " ".join(parts)
    if isinstance(err, str):
        return err
    return str(body.get("_text") or "")


def classify_error(http_status: int, message: str, ban_markers: list[str]) -> str:
    low = (message or "").lower()
    if any(m in low for m in ban_markers):
        return "banned"
    if http_status == 401 or "invalid_token" in low or "access_token_invalid" in low \
            or "oauth" in low:
        return "auth"
    return "error"


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(s.replace("Z", "+0000"), fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Клиенты площадок
# ---------------------------------------------------------------------------

class _Client:
    platform = ""

    def __init__(self, http: HttpFn, timeout: float, ban_markers: list[str]):
        self.http = http
        self.timeout = timeout
        self.ban_markers = ban_markers

    def _call(self, method: str, url: str, **kw) -> tuple[int, Any] | FetchResult:
        try:
            return self.http(method, url, timeout=self.timeout, **kw)
        except Exception as e:  # сеть, DNS, таймаут — статус аккаунта не трогаем
            return FetchResult(error_kind="error", error=f"{type(e).__name__}: {e}"[:300])

    def _fail(self, status: int, body: Any) -> FetchResult:
        msg = _err_message(body) or f"HTTP {status}"
        return FetchResult(error_kind=classify_error(status, msg, self.ban_markers), error=msg[:500])

    def fetch(self, account: dict, credential: str | None,
              want_views: Callable[[PostStat], bool]) -> FetchResult:
        raise NotImplementedError


class TikTokClient(_Client):
    """TikTok Display API v2: POST /v2/video/list/ [ПРОВЕРИТЬ ПО ДОКАМ: поля, лимит max_count,
    коды ошибок для заблокированного аккаунта]."""
    platform = "tiktok"

    def fetch(self, account, credential, want_views):
        if not credential:
            return FetchResult(error_kind="not_configured", error="нет токена TikTok")
        res = self._call("POST", TIKTOK_VIDEO_LIST_URL,
                         headers={"Authorization": f"Bearer {credential}",
                                  "Content-Type": "application/json"},
                         params={"fields": "id,create_time,view_count"},
                         json_body={"max_count": 20})
        if isinstance(res, FetchResult):
            return res
        status, body = res
        err = body.get("error") if isinstance(body, dict) else None
        code = str(err.get("code", "ok")) if isinstance(err, dict) else "ok"
        if status != 200 or code != "ok":
            return self._fail(status, body)
        posts = []
        for v in ((body.get("data") or {}).get("videos") or []):
            vid, ts = str(v.get("id", "")), _int_or_none(v.get("create_time"))
            if not MEDIA_ID_RE.fullmatch(vid) or ts is None:
                continue
            posts.append(PostStat(vid, datetime.fromtimestamp(ts, timezone.utc),
                                  _int_or_none(v.get("view_count"))))
        return FetchResult(posts=posts)


class InstagramClient(_Client):
    """Instagram Graph API: GET /{ig-user-id}/media, GET /{media-id}/insights?metric=views
    [ПРОВЕРИТЬ ПО ДОКАМ: название метрики (views/plays/video_views) и форма ответа insights
    для Reels в текущей версии Graph API]."""
    platform = "instagram"

    def __init__(self, http, timeout, ban_markers, graph_version: str = "v21.0"):
        super().__init__(http, timeout, ban_markers)
        self.base = f"{GRAPH_BASE}/{graph_version}"

    @staticmethod
    def _views_from_insights(body: Any) -> int | None:
        for item in (body.get("data") or []) if isinstance(body, dict) else []:
            if not isinstance(item, dict):
                continue
            tv = item.get("total_value")
            if isinstance(tv, dict) and _int_or_none(tv.get("value")) is not None:
                return _int_or_none(tv.get("value"))
            values = item.get("values") or []
            if values and isinstance(values[0], dict):
                return _int_or_none(values[0].get("value"))
        return None

    def fetch(self, account, credential, want_views):
        uid = str(account.get("ig_user_id") or "")
        if not IG_USER_RE.fullmatch(uid):
            return FetchResult(error_kind="not_configured", error="нет корректного ig_user_id")
        if not credential:
            return FetchResult(error_kind="not_configured", error="нет токена Instagram")
        res = self._call("GET", f"{self.base}/{uid}/media",
                         params={"fields": "id,timestamp,media_product_type", "limit": 25,
                                 "access_token": credential})
        if isinstance(res, FetchResult):
            return res
        status, body = res
        if status != 200 or (isinstance(body, dict) and body.get("error")):
            return self._fail(status, body)
        posts = []
        for m in (body.get("data") or []):
            mid, ts = str(m.get("id", "")), _parse_iso(m.get("timestamp"))
            if not MEDIA_ID_RE.fullmatch(mid) or ts is None:
                continue
            post = PostStat(mid, ts, None)
            if want_views(post):  # insights — отдельный запрос, тянем только для нужных постов
                ins = self._call("GET", f"{self.base}/{mid}/insights",
                                 params={"metric": "views", "access_token": credential})
                if not isinstance(ins, FetchResult) and ins[0] == 200:
                    post.views = self._views_from_insights(ins[1])
                else:
                    log.warning("instagram insights %s: %s", mid,
                                ins.error if isinstance(ins, FetchResult) else _err_message(ins[1]))
            posts.append(post)
        return FetchResult(posts=posts)


class YouTubeClient(_Client):
    """YouTube Data API v3: channels.list → playlistItems.list (uploads) → videos.list
    [ПРОВЕРИТЬ ПО ДОКАМ: отдаёт ли channels.list пустой items для закрытого канала]."""
    platform = "youtube"

    def fetch(self, account, credential, want_views):
        cid = str(account.get("channel_id") or "")
        if not YT_CHANNEL_RE.fullmatch(cid):
            return FetchResult(error_kind="not_configured", error="нет корректного channel_id")
        if not credential:
            return FetchResult(error_kind="not_configured", error="нет YOUTUBE_API_KEY")
        res = self._call("GET", f"{YT_BASE}/channels",
                         params={"part": "contentDetails,status", "id": cid, "key": credential})
        if isinstance(res, FetchResult):
            return res
        status, body = res
        if status != 200:
            return self._fail(status, body)
        items = body.get("items") or []
        if not items:
            return FetchResult(error_kind="not_found", error=f"канал {cid} не найден")
        uploads = (((items[0].get("contentDetails") or {}).get("relatedPlaylists") or {})
                   .get("uploads") or "")
        if not MEDIA_ID_RE.fullmatch(uploads):
            return FetchResult(posts=[])
        res = self._call("GET", f"{YT_BASE}/playlistItems",
                         params={"part": "contentDetails", "playlistId": uploads,
                                 "maxResults": 15, "key": credential})
        if isinstance(res, FetchResult):
            return res
        status, body = res
        if status != 200:
            return self._fail(status, body)
        posts: dict[str, PostStat] = {}
        for it in body.get("items") or []:
            cd = it.get("contentDetails") or {}
            vid, ts = str(cd.get("videoId", "")), _parse_iso(cd.get("videoPublishedAt"))
            if MEDIA_ID_RE.fullmatch(vid) and ts is not None:
                posts[vid] = PostStat(vid, ts, None)
        need = [p.post_id for p in posts.values() if want_views(p)]
        if need:
            res = self._call("GET", f"{YT_BASE}/videos",
                             params={"part": "statistics", "id": ",".join(need), "key": credential})
            if isinstance(res, FetchResult):
                return res
            status, body = res
            if status != 200:
                return self._fail(status, body)
            for it in body.get("items") or []:
                p = posts.get(str(it.get("id", "")))
                if p is not None:
                    p.views = _int_or_none((it.get("statistics") or {}).get("viewCount"))
        return FetchResult(posts=list(posts.values()))


def credential_env_name(platform: str, device_id: str) -> str:
    if platform == "youtube":
        return "YOUTUBE_API_KEY"
    return f"{platform.upper()}_TOKEN_{validate_device_id(device_id).upper().replace('-', '_')}"


# ---------------------------------------------------------------------------
# Лог в logs/account_health.log
# ---------------------------------------------------------------------------

def _ensure_file_log(logs_dir: Path) -> None:
    target = (Path(logs_dir) / "account_health.log").resolve()
    if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target
           for h in log.handlers):
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(target, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)
        if log.level == logging.NOTSET:
            log.setLevel(logging.INFO)
    except OSError as e:
        sys.stderr.write(f"[account_health] лог-файл {target} не открыт: {e}\n")


# ---------------------------------------------------------------------------
# Состояние и правила
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AccountHealth:
    def __init__(self, cfg: Any, http: HttpFn | None = None, clients: dict | None = None,
                 now: Callable[[], datetime] | None = None):
        self.cfg = cfg
        self.hc = cfg.account_health
        self.dir = Path(cfg.path("accounts"))
        self._now = now or (lambda: datetime.now(timezone.utc))
        http = http or requests_http
        markers = [m.lower() for m in self.hc.ban_error_markers]
        timeout = float(self.hc.request_timeout_sec)
        self.clients = clients or {
            "tiktok": TikTokClient(http, timeout, markers),
            "instagram": InstagramClient(http, timeout, markers, self.hc.graph_api_version),
            "youtube": YouTubeClient(http, timeout, markers),
        }
        _ensure_file_log(cfg.path("logs"))

    @property
    def enabled(self) -> bool:
        return bool(self.hc.enabled)

    def now(self) -> datetime:
        return self._now().astimezone(timezone.utc)

    def today(self) -> date_cls:
        try:
            from zoneinfo import ZoneInfo
            return self.now().astimezone(ZoneInfo(getattr(self.cfg, "timezone", "UTC"))).date()
        except Exception:
            return self.now().date()

    # --- файлы -----------------------------------------------------------------

    def state_path(self, device_id: str, platform: str) -> Path:
        validate_device_id(device_id)
        if platform not in PLATFORMS:
            raise ValueError(f"неизвестная площадка: {platform!r}")
        return self.dir / f"{device_id}_{platform}.json"

    @staticmethod
    def _created_from_registry(device: dict | None, account: dict | None) -> str | None:
        for src in (account or {}, device or {}):
            v = str(src.get("created_at") or "")[:10]
            if DAY_RE.fullmatch(v):
                try:
                    date_cls.fromisoformat(v)
                    return v
                except ValueError:
                    pass
        return None

    def _default_state(self, device_id: str, platform: str, device: dict | None) -> dict:
        account = ((device or {}).get("accounts") or {}).get(platform) or {}
        created = self._created_from_registry(device, account)
        return {"device_id": device_id, "platform": platform,
                "username": account.get("username"), "status": "new", "status_reason": "",
                "created_at": created or self.today().isoformat(),
                "created_source": "devices.json" if created else "first_seen",
                "first_post_at": None, "last_post_at": None, "shadowban_suspect": False,
                "avg_views_48h": None, "verified": False, "last_check_at": None,
                "last_error": None, "updated_at": _iso(self.now())}

    def load_state(self, device_id: str, platform: str, device: dict | None = None) -> dict:
        p = self.state_path(device_id, platform)
        if p.exists():
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(st, dict) and st.get("status") in STATUSES:
                    # devices.json — источник истины для даты создания, если она там появилась
                    created = self._created_from_registry(
                        device, ((device or {}).get("accounts") or {}).get(platform))
                    if created and st.get("created_source") != "devices.json":
                        st["created_at"], st["created_source"] = created, "devices.json"
                    return st
                raise ValueError("неожиданная структура")
            except Exception as e:
                backup = p.with_name(f"{p.name}.corrupt-{int(time.time())}")
                p.replace(backup)
                log.error("%s %s: состояние повреждено (%s), копия %s — создаю заново (status=new)",
                          device_id, platform, e, backup.name)
        st = self._default_state(device_id, platform, device)
        self.save_state(st)
        return st

    def save_state(self, st: dict) -> None:
        st["updated_at"] = _iso(self.now())
        atomic_write_text(self.state_path(st["device_id"], st["platform"]),
                          json.dumps(st, ensure_ascii=False, indent=2))

    # --- правила ---------------------------------------------------------------

    def _activation_day(self, st: dict) -> date_cls:
        return date_cls.fromisoformat(st["created_at"]) + timedelta(days=int(self.hc.new_account_days))

    def refresh_status(self, st: dict, day: date_cls | None = None) -> bool:
        """Пересчёт статуса по дате и флагам. True — что-то изменилось."""
        day = day or self.today()
        before = (st["status"], st.get("status_reason"))
        if st["status"] == "banned":
            return False
        age_ok = day >= self._activation_day(st)
        if st.get("shadowban_suspect"):
            if age_ok:
                st["status"], st["status_reason"] = "restricted", "shadowban_suspect: публикация на паузе"
        elif st["status"] in ("new", "restricted"):
            if age_ok:
                st["status"], st["status_reason"] = "active", \
                    f"прошло {self.hc.new_account_days} дн. с {st['created_at']}"
            else:
                st["status"], st["status_reason"] = "new", f"активация {self._activation_day(st)}"
        changed = before != (st["status"], st.get("status_reason"))
        if changed:
            log.info("device=%s account=%s статус %s → %s (%s)", st["device_id"], st["platform"],
                     before[0], st["status"], st["status_reason"])
        return changed

    def _state(self, device_id: str, platform: str, device: dict | None,
               day: date_cls | None) -> dict:
        st = self.load_state(device_id, platform, device)
        if self.refresh_status(st, day):
            self.save_state(st)
        return st

    @staticmethod
    def _day(day: str | date_cls | None) -> date_cls | None:
        if day is None or isinstance(day, date_cls):
            return day
        if not DAY_RE.fullmatch(day):
            raise ValueError(f"некорректная дата: {day!r}")
        return date_cls.fromisoformat(day)

    def posts_allowed(self, device_id: str, platform: str, day: str | date_cls | None = None,
                      device: dict | None = None) -> int:
        """Сколько постов на площадку можно сделать за день."""
        if not self.enabled:
            return UNLIMITED
        d = self._day(day) or self.today()
        st = self._state(device_id, platform, device, d)
        if st["status"] != "active" or st.get("shadowban_suspect"):
            return 0
        prog = list(self.hc.max_posts_per_day_progression)
        k = max(0, (d - self._activation_day(st)).days)
        return int(prog[min(k, len(prog) - 1)])

    def can_post(self, device_id: str, platform: str, day: str | date_cls | None = None,
                 device: dict | None = None) -> tuple[bool, str]:
        if not self.enabled:
            return True, "account_health выключен"
        st = self._state(device_id, platform, device, self._day(day))
        if st["status"] == "active" and not st.get("shadowban_suspect"):
            return True, "active"
        return False, f"{st['status']}: {st.get('status_reason') or ''}".strip()

    def plan_platforms(self, device: dict) -> list[str]:
        """Площадки устройства для плана mimic: без banned и shadowban_suspect
        (новые оставляем — план на будущие дни, запрет публикации проверит autoposter)."""
        device_id = device["device_id"]
        out = []
        for p, acc in (device.get("accounts") or {}).items():
            if not acc.get("enabled", True) or p not in PLATFORMS:
                continue
            if not self.enabled:
                out.append(p)
                continue
            st = self._state(device_id, p, device, None)
            if st["status"] != "banned" and not st.get("shadowban_suspect"):
                out.append(p)
        return out

    def device_usable(self, device: dict) -> bool:
        """False — все аккаунты устройства в бане: исключить из forge/ghost/autoposter."""
        if not self.enabled:
            return True
        accounts = [p for p, a in (device.get("accounts") or {}).items()
                    if a.get("enabled", True) and p in PLATFORMS]
        return any(self._state(device["device_id"], p, device, None)["status"] != "banned"
                   for p in accounts)

    def record_post(self, device_id: str, platform: str, when: datetime | None = None) -> None:
        if not self.enabled:
            return
        st = self.load_state(device_id, platform)
        ts = _iso(when or self.now())
        st["first_post_at"] = st.get("first_post_at") or ts
        st["last_post_at"] = ts
        self.save_state(st)

    def set_status(self, device_id: str, platform: str, status: str, reason: str = "вручную") -> dict:
        if status not in STATUSES:
            raise ValueError(f"статус должен быть одним из {STATUSES}")
        st = self.load_state(device_id, platform)
        st["status"], st["status_reason"] = status, reason
        if status == "active":
            st["shadowban_suspect"] = False
        self.save_state(st)
        log.warning("device=%s account=%s статус вручную → %s", device_id, platform, status)
        return st

    # --- проверка через API ----------------------------------------------------

    def check_account(self, device: dict, platform: str) -> dict:
        device_id = validate_device_id(device["device_id"])
        account = (device.get("accounts") or {}).get(platform) or {}
        st = self.load_state(device_id, platform, device)
        started = time.monotonic()
        if st["status"] == "banned":
            log.info("device=%s account=%s action=check skip=banned", device_id, platform)
            return st
        client = self.clients.get(platform)
        if client is None:
            return st
        now = self.now()
        lo = now - timedelta(hours=int(self.hc.shadowban_max_post_age_hours))
        hi = now - timedelta(hours=int(self.hc.shadowban_check_hours))

        def want_views(p: PostStat) -> bool:
            return lo <= p.published_at <= hi

        res = client.fetch(account, os.getenv(credential_env_name(platform, device_id)), want_views)
        st["last_check_at"] = _iso(now)
        if res.error_kind == "banned":
            st["status"], st["status_reason"] = "banned", f"API: {res.error}"
            st["last_error"] = res.error
            log.error("device=%s account=%s БАН по ответу API: %s", device_id, platform, res.error)
        elif res.error_kind == "not_found":
            st["last_error"] = res.error
            if st.get("verified"):
                st["status"], st["status_reason"] = "banned", f"API: {res.error} (ранее был доступен)"
                log.error("device=%s account=%s аккаунт исчез из API — banned", device_id, platform)
            else:
                log.warning("device=%s account=%s %s — проверь идентификатор в devices.json",
                            device_id, platform, res.error)
        elif not res.ok:
            st["last_error"] = f"{res.error_kind}: {res.error}"
            log.warning("device=%s account=%s проверка не выполнена (%s): %s — статус не меняю",
                        device_id, platform, res.error_kind, res.error)
        else:
            st["verified"], st["last_error"] = True, None
            self._apply_posts(st, res.posts)
        self.refresh_status(st)
        self.save_state(st)
        log.info("device=%s account=%s action=check status=%s suspect=%s avg_views=%s "
                 "posts=%d duration=%.2fs", device_id, platform, st["status"],
                 st.get("shadowban_suspect"), st.get("avg_views_48h"), len(res.posts),
                 time.monotonic() - started)
        return st

    def _apply_posts(self, st: dict, posts: list[PostStat]) -> None:
        if not posts:
            return
        earliest = min(p.published_at for p in posts)
        known_first = _parse_iso(st.get("first_post_at"))
        if known_first is None or known_first > earliest:
            st["first_post_at"] = _iso(earliest)
        # аккаунт уже вёлся до подключения — не держим его «новым»
        if st.get("created_source") == "first_seen" \
                and earliest.date() < date_cls.fromisoformat(st["created_at"]):
            st["created_at"], st["created_source"] = earliest.date().isoformat(), "api_earliest_post"
            log.info("device=%s account=%s created_at по самому раннему посту: %s",
                     st["device_id"], st["platform"], st["created_at"])
        now = self.now()
        lo = now - timedelta(hours=int(self.hc.shadowban_max_post_age_hours))
        hi = now - timedelta(hours=int(self.hc.shadowban_check_hours))
        matured = [p for p in posts if lo <= p.published_at <= hi and p.views is not None]
        if len(matured) < int(self.hc.shadowban_min_posts):
            return  # не на чем судить — флаг не трогаем
        avg = sum(p.views or 0 for p in matured) / len(matured)
        st["avg_views_48h"] = round(avg, 1)
        threshold = int(self.hc.shadowban_views_threshold)
        if avg < threshold and not st.get("shadowban_suspect"):
            st["shadowban_suspect"] = True
            log.warning("device=%s account=%s shadowban_suspect: средние просмотры %.1f < %d "
                        "(%d постов) — публикация на паузе", st["device_id"], st["platform"],
                        avg, threshold, len(matured))
        elif avg >= threshold and st.get("shadowban_suspect"):
            st["shadowban_suspect"] = False
            log.info("device=%s account=%s охват восстановился (%.1f ≥ %d) — пауза снята",
                     st["device_id"], st["platform"], avg, threshold)

    def missing_credentials(self, devices: list[dict]) -> list[str]:
        out = []
        for d in devices:
            for p, acc in (d.get("accounts") or {}).items():
                if acc.get("enabled", True) and p in PLATFORMS \
                        and not os.getenv(credential_env_name(p, d["device_id"])):
                    out.append(f"{d['device_id']}/{p}: {credential_env_name(p, d['device_id'])}")
        return out

    def run(self, devices: list[dict]) -> dict[str, dict]:
        if not self.enabled:
            log.info("account_health выключен (account_health.enabled=false)")
            return {}
        missing = self.missing_credentials(devices)
        if missing:
            log.warning("нет учётных данных (проверка охвата/бана для них пропускается): %s",
                        "; ".join(missing))
        out: dict[str, dict] = {}
        for d in devices:
            for p, acc in (d.get("accounts") or {}).items():
                if not acc.get("enabled", True) or p not in PLATFORMS:
                    continue
                try:
                    st = self.check_account(d, p)
                except Exception as e:
                    log.exception("device=%s account=%s проверка упала: %s", d.get("device_id"), p, e)
                    continue
                out[f"{d['device_id']}/{p}"] = {k: st.get(k) for k in (
                    "status", "shadowban_suspect", "avg_views_48h", "created_at", "last_error")}
        counts: dict[str, int] = {}
        for v in out.values():
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        log.info("account_health: %s, shadowban_suspect=%d", counts,
                 sum(1 for v in out.values() if v["shadowban_suspect"]))
        return out


def health_for(cfg: Any) -> AccountHealth | None:
    """AccountHealth, если модуль включён в конфиге; иначе None (старое поведение)."""
    hc = getattr(cfg, "account_health", None)
    if hc is None or not getattr(hc, "enabled", False):
        return None
    return AccountHealth(cfg)


def stage_account_health(cfg: Any, registry: Any = None, health: AccountHealth | None = None) -> int:
    """Этап orchestrator. Всегда 0: проблемы аккаунтов — не повод останавливать цикл."""
    if registry is None:
        from devices_registry import DevicesRegistry
        registry = DevicesRegistry.from_config(cfg)
    health = health or AccountHealth(cfg)
    health.run(registry.get_active_devices())
    return 0


def main() -> None:
    from config_loader import cli_config

    def extra(ap):
        ap.add_argument("--device", action="append", default=None)
        ap.add_argument("--set", action="append", default=[], metavar="DEVICE:PLATFORM:STATUS",
                        help="выставить статус вручную, например aria_fight:tiktok:active")

    cfg, a = cli_config("account_health", extra=extra)
    health = AccountHealth(cfg)
    for item in a.set:
        try:
            dev, plat, status = item.split(":")
        except ValueError:
            raise SystemExit(f"--set ожидает DEVICE:PLATFORM:STATUS, получено {item!r}") from None
        health.set_status(dev, plat, status)
    if a.set:
        return
    from devices_registry import DevicesRegistry
    devices = DevicesRegistry.from_config(cfg).get_active_devices(device_ids=a.device)
    sys.stdout.write(json.dumps(health.run(devices), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
